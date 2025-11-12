# -*- coding: utf-8 -*-
"""
Dual-Mode Minimal Inference (FastVisionModel / Qwen2.5-VL)
Supports 8GB Models, JPG/PNG Compatible
"""
import torch
from PIL import Image
from typing import Literal

# Import model-specific dependencies
try:
    from unsloth import FastVisionModel
    from transformers import AutoProcessor, AutoTokenizer
except ImportError as e:
    raise ImportError(
        "Please install required dependencies: pip install unsloth transformers accelerate pillow torch") from e

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor
except ImportError:
    print("Install Qwen dependencies: pip install transformers>=4.40.0 accelerate>=0.30.0")


def simple_inference(
        model_path: str,
        image_path: str,
        prompt: str,
        model_type: Literal["fastvision", "qwen2.5-vl"] = "fastvision"
) -> str:
    """
    Dual-mode minimal inference supporting FastVisionModel and Qwen2_5_VLForConditionalGeneration

    Args:
        model_path: Path to the pre-trained/merged model
        image_path: Path to input image (JPG/PNG)
        prompt: Inference prompt text
        model_type: Model architecture type - "fastvision" or "qwen2.5-vl"

    Returns:
        Decoded inference result
    """
    # Validate model type
    if model_type not in ["fastvision", "qwen2.5-vl"]:
        raise ValueError(f"Invalid model_type: {model_type}. Choose 'fastvision' or 'qwen2.5-vl'")

    # Core configuration (shared across models)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_seq_length = 1024
    max_new_tokens = 4096
    target_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    print(f"Loading {model_type} model from: {model_path}")

    # --------------------------
    # Model-specific Loading
    # --------------------------
    if model_type == "fastvision":
        # FastVisionModel (from unsloth) loading
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            load_in_8bit=False,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
        model = model.to(target_dtype).eval()

    else:  # model_type == "qwen2.5-vl"
        # Qwen2.5-VL loading
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        processor = Qwen2VLProcessor.from_pretrained(
            model_path,
            tokenizer=tokenizer,
            trust_remote_code=True
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=target_dtype,
            device_map=device,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()

    # Clear GPU cache if available
    if device == "cuda":
        torch.cuda.empty_cache()

    # --------------------------
    # Shared Preprocessing
    # --------------------------
    # Load and process image
    image = Image.open(image_path).convert("RGB")

    # Build chat template (compatible with both models)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "dummy"}}  # Dummy URL (actual image passed via processor)
            ]
        }
    ]

    # Prepare input text with chat template
    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Prepare model inputs
    inputs = processor(
        text=[text_input],
        images=[image],
        padding=True,
        return_tensors="pt",
        torch_dtype=target_dtype
    ).to(device)

    # --------------------------
    # Inference Generation
    # --------------------------
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 1.0,
    }
    # Add Qwen-specific generation params if needed
    if model_type == "qwen2.5-vl":
        generation_kwargs["num_beams"] = 1

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    # --------------------------
    # Result Decoding
    # --------------------------
    prompt_length = inputs.input_ids.shape[1]
    result = tokenizer.decode(
        generated_ids[0][prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    print(f"{model_type} inference completed successfully")
    return result


# --------------------------
# Usage Examples
# --------------------------
if __name__ == "__main__":
    # Core configuration - modify these parameters as needed
    MODEL_PATH = "./model/s1_parser"
    TEST_IMAGE = "/data/test/parse_15.jpg"
    PROMPT = "Please recognize the LaTeX formula in the image and output the complete LaTeX code without additional explanations."

    # --------------------------
    # Example 1: FastVisionModel (default)
    # --------------------------
    print("\n" + "=" * 60)
    print("Example 1: Inference with FastVisionModel")
    print("=" * 60)
    try:
        result_fastvision = simple_inference(
            model_path=MODEL_PATH,
            image_path=TEST_IMAGE,
            prompt=PROMPT,
            model_type="fastvision"
        )
        print("Inference Result:")
        print("-" * 40)
        print(result_fastvision)
    except Exception as e:
        print(f"FastVisionModel inference failed: {str(e)}")

    # --------------------------
    # Example 2: Qwen2.5-VL
    # --------------------------
    print("\n" + "=" * 60)
    print("Example 2: Inference with Qwen2.5-VL")
    print("=" * 60)
    try:
        result_qwen = simple_inference(
            model_path=MODEL_PATH,
            image_path=TEST_IMAGE,
            prompt=PROMPT,
            model_type="qwen2.5-vl"
        )
        print("Inference Result:")
        print("-" * 40)
        print(result_qwen)
    except Exception as e:
        print(f"Qwen2.5-VL inference failed: {str(e)}")
