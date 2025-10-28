import os
import torch
from magic_pdf.config.constants import *
from magic_pdf.model.sub_modules.model_init import AtomModelSingleton
from magic_pdf.model.model_list import AtomicModel
from transformers import LayoutLMv3ForTokenClassification
from loguru import logger
import yaml
from qwen_vl_utils import process_vision_info
from PIL import Image
import requests
from typing import List, Union
from openai import OpenAI
import torch
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor


class TaichuOCR:
    def __init__(self, config_path):
        current_file_path = os.path.abspath(__file__)

        current_dir = os.path.dirname(current_file_path)

        root_dir = os.path.dirname(current_dir)

        with open(config_path, 'r', encoding='utf-8') as f:
            self.configs = yaml.load(f, Loader=yaml.FullLoader)
        logger.info('using configs: {}'.format(self.configs))

        self.device = self.configs.get('device', 'cpu')
        logger.info('using device: {}'.format(self.device))

        bf16_supported = False
        if self.device.startswith("cuda"):
            bf16_supported = torch.cuda.is_bf16_supported()
        elif self.device.startswith("mps"):
            bf16_supported = True

        models_dir = self.configs.get(
            'models_dir', os.path.join(root_dir, 'model_weight')
        )

        logger.info('using models_dir: {}'.format(models_dir))
        if not os.path.exists(models_dir):
            raise FileNotFoundError(
                f"Model directory '{models_dir}' not found. "
                "Please run 'python download_model.py' to download the required models."
            )

        self.layout_config = self.configs.get('layout_config')
        self.layout_model_name = self.layout_config.get(
            'model', MODEL_NAME.DocLayout_YOLO
        )

        layout_model_path = os.path.join(models_dir, self.configs['weights'][self.layout_model_name])
        if not os.path.exists(layout_model_path):
            raise FileNotFoundError(
                f"Layout model file not found at '{layout_model_path}'. "
                "Please run 'python download_model.py' to download the required models."
            )

        atom_model_manager = AtomModelSingleton()
        if self.layout_model_name == MODEL_NAME.DocLayout_YOLO:
            self.layout_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.Layout,
                layout_model_name=MODEL_NAME.DocLayout_YOLO,
                doclayout_yolo_weights=layout_model_path,
                device=self.device,
            )
        logger.info(f'layout model loaded: {self.layout_model_name}')

        layout_reader_config = self.layout_config.get('reader')
        self.layout_reader_name = layout_reader_config.get('name')
        if self.layout_reader_name == 'layoutreader':
            layoutreader_model_dir = os.path.join(models_dir, self.configs['weights'][self.layout_reader_name])
            if os.path.exists(layoutreader_model_dir):
                model = LayoutLMv3ForTokenClassification.from_pretrained(
                    layoutreader_model_dir
                )
            else:
                logger.warning(
                    'local layoutreader model not exists, use online model from huggingface'
                )
                model = LayoutLMv3ForTokenClassification.from_pretrained(
                    'hantian/layoutreader'
                )

            if bf16_supported:
                model.to(self.device).eval().bfloat16()
            else:
                model.to(self.device).eval()
        else:
            logger.error('model name not allow')
        self.layoutreader_model = model
        logger.info(f'layoutreader model loaded: {self.layout_reader_name}')

        self.chat_config = self.configs.get('chat_config', {})
        chat_backend = self.chat_config.get('backend', 'lmdeploy')
        chat_path = self.chat_config.get('weight_path', 'model_weight/Recognition')
        if chat_backend == 'lmdeploy':
            logger.info('Use LMDeploy as backend')
            self.chat_model = \
                (chat_path)
        elif chat_backend == 'vllm':
            logger.info('Use vLLM as backend')
            self.chat_model = TaichuChat_vLLM(chat_path)
        elif chat_backend == 'transformers':
            logger.info('Use transformers as backend')
            batch_size = self.chat_config.get('batch_size', 5)
            self.chat_model = TaichuChat_transformers(chat_path, batch_size, device=self.device)
        elif chat_backend == 'api':
            logger.info('Use API as backend')
            api_config = self.configs.get('api_config', {})
            if not api_config:
                raise ValueError("API configuration is required for API backend.")
            self.chat_model = TaichuChat_OpenAIAPI(
                url=api_config.get('url'),
                model_name=api_config.get('model_name'),
                api_key=api_config.get('api_key', None)
            )
        logger.info(f'VLM loaded: {self.chat_model.model_name}')



class TaichuChat_vLLM:
    def __init__(self, model_path):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vLLM is not installed. Please install it following ")
        self.model_name = os.path.basename(model_path)
        self.pipe = LLM(model=model_path,
                        max_seq_len_to_capture=10240,
                        mm_processor_kwargs={'use_fast': True},
                        gpu_memory_utilization=self._auto_gpu_mem_ratio(0.9))
        self.gen_config = SamplingParams(max_tokens=4096, temperature=0, repetition_penalty=1.05)

    def _auto_gpu_mem_ratio(self, ratio):
        mem_free, mem_total = torch.cuda.mem_get_info()
        ratio = ratio * mem_free / mem_total
        return ratio

    def batch_inference(self, images, questions):
        placeholder = "<|image_pad|>"
        prompts = [
            ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
             f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
             f"{question}<|im_end|>\n"
             "<|im_start|>assistant\n") for question in questions
        ]
        inputs = [{
            "prompt": prompts[i],
            "multi_modal_data": {
                "image": images[i],
            }
        } for i in range(len(prompts))]
        outputs = self.pipe.generate(inputs, sampling_params=self.gen_config)
        return [o.outputs[0].text for o in outputs]


class TaichuChat_transformers:
    def __init__(self, model_path: str, max_batch_size: int = 10, max_new_tokens=4096, device: str = None):
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError("transformers is not installed. Please install it following")
        self._setup_global_cuda_environment()
        self.model_name = os.path.basename(model_path)
        self.max_batch_size = max_batch_size
        self.max_new_tokens = max_new_tokens

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        #打印 GPU 使用情况
        self._print_gpu_info()

        bf16_supported = False
        if self.device.startswith("cuda"):
            bf16_supported = torch.cuda.is_bf16_supported()
        elif self.device.startswith("mps"):
            bf16_supported = True

        logger.info(f"Loading Qwen2.5VL model from: {model_path}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Max batch size: {self.max_batch_size}")

        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory={
                    0: "20GiB",
                    1: "20GiB",
                    2: "20GiB",
                    3: "20GiB"
                },
                quantization_config=quantization_config,
                low_cpu_mem_usage=True
            )

            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.processor.tokenizer.padding_side = "left"

            self.model.eval()
            logger.info("Qwen2.5VL model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def _print_gpu_info(self):
        import torch
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
            memory_reserved = torch.cuda.memory_reserved(i) / 1024 / 1024
            print(f"GPU {i} (物理 GPU {2 + i}) 内存使用: 已分配 {memory_allocated:.2f} MB, 已预留 {memory_reserved:.2f} MB")

    @staticmethod
    def _setup_global_cuda_environment():
        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,7"
        os.environ["TRITON_CUDA_NUM_STAGES"] = "2"
        os.environ["TRITON_CUDA_NUM_WARPS"] = "4"

    def load_image(self, image_source: Union[str, Image.Image]) -> Image.Image:
        if isinstance(image_source, str):
            if image_source.startswith('http'):
                response = requests.get(image_source)
                return Image.open(response.content).convert('RGB')
            else:
                return Image.open(image_source).convert('RGB')
        elif isinstance(image_source, Image.Image):
            return image_source.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image_source)}")

    def prepare_messages(self, images: List[Union[str, Image.Image]], questions: List[str]) -> List[List[dict]]:
        if len(images) != len(questions):
            raise ValueError("Images and questions must have the same length")

        all_messages = []
        for image, question in zip(images, questions):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image if isinstance(image, str) else image,
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]
            all_messages.append(messages)

        return all_messages

    def batch_inference(self, images: List[Union[str, Image.Image]], questions: List[str]) -> List[str]:
        if len(images) != len(questions):
            raise ValueError("Images and questions must have the same length")

        results = []
        total_items = len(images)

        for i in range(0, total_items, self.max_batch_size):
            batch_end = min(i + self.max_batch_size, total_items)
            batch_images = images[i:batch_end]
            batch_questions = questions[i:batch_end]

            logger.info(
                f"Processing batch {i // self.max_batch_size + 1}/{(total_items - 1) // self.max_batch_size + 1} "
                f"(items {i + 1}-{batch_end})")

            try:
                batch_results = self._process_batch(batch_images, batch_questions)
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Batch processing failed for items {i + 1}-{batch_end}: {e}")
                logger.info("Falling back to single processing...")
                for img, q in zip(batch_images, batch_questions):
                    try:
                        single_result = self._process_single(img, q)
                        results.append(single_result)
                    except Exception as single_e:
                        logger.error(f"Single processing also failed: {single_e}")
                        results.append(f"Error: {str(single_e)}")

            if self.device == 'cuda':
                torch.cuda.empty_cache()

        return results

    def _process_batch(self, batch_images: List[Union[str, Image.Image]], batch_questions: List[str]) -> List[str]:
        all_messages = self.prepare_messages(batch_images, batch_questions)

        texts = []
        image_inputs = []

        for messages in all_messages:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)

            image_inputs.append(process_vision_info(messages)[0])

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.1,
                repetition_penalty=1.05,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return [text.strip() for text in output_texts]

    def _process_single(self, image: Union[str, Image.Image], question: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.1,
                repetition_penalty=1.05,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text.strip()

    def single_inference(self, image: Union[str, Image.Image], question: str) -> str:
        return self._process_single(image, question)


class TaichuChat_OpenAIAPI:
    def __init__(self, url: str, model_name: str, api_key: str = None):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=url
        )
        if not self.validate_connection():
            raise ValueError("Invalid API URL or API key. Please check your configuration.")

    def validate_connection(self) -> bool:
        """
        Validate the effectiveness of API URL and key
        """
        try:
            # Try to get model list to validate connection
            response = self.client.models.list()
            logger.info("API connection validation successful")
            return True
        except Exception as e:
            logger.error(f"API connection validation failed: {e}")
            return False

    def img2base64(self, image: Image.Image):
        """
        Convert a PIL Image to a Base64 encoded string.
        """
        import io
        import base64

        buffered = io.BytesIO()

        try:
            if hasattr(image, 'format') and image.format:
                img_format = image.format
            else:
                # Default to PNG if format is not specified
                img_format = "PNG"

            image.save(buffered, format=img_format)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return img_base64, img_format.lower()

        except Exception as e:
            raise ValueError(f"Failed to convert image to base64: {e}")

    def batch_inference(self, images: List[Union[str, Image.Image]], questions: List[str]) -> List[str]:
        results = []
        for image, question in zip(images, questions):
            try:
                if isinstance(image, Image.Image):
                    img, img_type = self.img2base64(image)
                else:
                    img, img_type = image, 'png'

                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/{img_type};base64,{img}"
                        },
                        {
                            "type": "input_text",
                            "text": question
                        }
                    ],
                }]
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages
                )
                results.append(response.choices[0].message.content)
            except Exception as e:
                results.append(f"Error: {e}")
        return results