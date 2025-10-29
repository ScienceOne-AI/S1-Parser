# -*- coding: utf-8 -*-
"""
LaTeX OCR 模型的 SFT 训练脚本
基于 Unsloth 框架优化，支持视觉-语言多模态模型微调
"""
import os
import json
import gc
import logging
import argparse
from typing import Dict, List, Generator

import torch
from PIL import Image
from datasets import Dataset
from transformers import AutoTokenizer, TextStreamer
from torchvision import transforms

import unsloth
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LaTeX OCR 模型 SFT 训练脚本")

    # 路径配置
    parser.add_argument("--base_model_dir", type=str,
                        default="/base_model/",
                        help="基础模型路径")
    parser.add_argument("--data_json", type=str,
                        default="/data/mergedata/latex_ocr_train.json",
                        help="训练数据JSON文件路径")
    parser.add_argument("--output_dir", type=str,
                        default="./output/OCR-sft-tuned",
                        help="模型输出目录")
    parser.add_argument("--offload_folder", type=str,
                        default="./offload",
                        help="模型卸载缓存目录")

    # 模型配置
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="最大序列长度")
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA 秩参数")
    parser.add_argument("--load_in_8bit", action="store_true", default=True,
                        help="是否使用8bit量化加载模型")
    parser.add_argument("--load_in_4bit", action="store_true", default=False,
                        help="是否使用4bit量化加载模型")

    # 训练配置
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="每个设备的训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--warmup_steps", type=int, default=50,
                        help="预热步数")
    parser.add_argument("--max_steps", type=int, default=1500,
                        help="最大训练步数")
    parser.add_argument("--save_steps", type=int, default=200000,
                        help="模型保存步数")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志打印步数")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减系数")
    parser.add_argument("--label_smoothing_factor", type=float, default=0.1,
                        help="标签平滑系数")

    return parser.parse_args()


class LaTeXOCRConfig:
    """LaTeX OCR 模型配置类"""

    def __init__(self, args: argparse.Namespace):
        # 路径配置
        self.base_model_dir = args.base_model_dir
        self.data_json = args.data_json
        self.output_dir = args.output_dir
        self.offload_folder = args.offload_folder

        # 模型参数
        self.max_seq_length = args.max_seq_length
        self.lora_rank = args.lora_rank
        self.load_in_8bit = args.load_in_8bit
        self.load_in_4bit = args.load_in_4bit
        self.gpu_memory_utilization = 0.6

        # 训练参数
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.learning_rate = args.learning_rate
        self.warmup_steps = args.warmup_steps
        self.max_steps = args.max_steps
        self.save_steps = args.save_steps
        self.logging_steps = args.logging_steps
        self.weight_decay = args.weight_decay
        self.label_smoothing_factor = args.label_smoothing_factor

        # 文本模板配置
        self.system_prompt = "You are a LaTeX OCR assistant, whose goal is to read photos input by users and convert them into LaTeX message."
        self.solution_start = "<SOLUTION>"
        self.solution_end = "</SOLUTION>"

        # 初始化输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.offload_folder, exist_ok=True)

    def to_dict(self) -> Dict:
        """转换为字典，用于保存配置"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


def get_image_transforms() -> transforms.Compose:
    """获取图像预处理转换管道"""
    return transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomRotation(degrees=(-5, 5), expand=False, fill=(255, 255, 255)),
        transforms.RandomResizedCrop(448, scale=(0.85, 1.0), ratio=(0.7, 1.3)),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.02 if torch.rand(1).item() < 0.2 else x)
    ])


def build_training_dataset(
        data_path: str,
        tokenizer: AutoTokenizer,
        config: LaTeXOCRConfig
) -> Dataset:
    """
    构建训练数据集

    Args:
        data_path: 数据JSON文件路径
        tokenizer: 分词器
        config: 模型配置对象

    Returns:
        处理后的训练数据集
    """
    img_transform = get_image_transforms()

    def data_generator() -> Generator[Dict, None, None]:
        """数据生成器"""
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        logger.info(f"加载原始数据，共 {len(raw_data)} 条样本")

        for idx, sample in enumerate(raw_data):
            try:
                # 解析样本数据
                img_path = sample["conversations"][0]["value"]
                formula = sample["conversations"][1]["value"]

                # 加载并预处理图像
                image = Image.open(img_path).convert("RGB")
                image = img_transform(image)

                # 构建对话消息
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": config.system_prompt},
                    ]
                }]

                # 应用聊天模板
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # 构建完整训练文本
                full_text = f"{prompt_text}{config.solution_start}{formula}{config.solution_end}"

                yield {
                    "text": full_text,
                    "prompt": prompt_text,
                    "answer": formula,
                    "image_path": img_path
                }

            except Exception as e:
                logger.warning(f"处理样本 {idx} 失败（路径: {img_path}）: {str(e)}，已跳过")
                continue

    # 构建数据集
    dataset = Dataset.from_generator(data_generator)
    logger.info(f"数据集构建完成，有效样本数: {len(dataset)}")
    return dataset


def load_model_and_tokenizer(config: LaTeXOCRConfig) -> tuple[FastVisionModel, AutoTokenizer]:
    """
    加载模型和分词器

    Args:
        config: 模型配置对象

    Returns:
        加载并配置好的模型和分词器
    """
    logger.info(f"开始加载模型: {config.base_model_dir}")

    # 加载基础模型
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=config.base_model_dir,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
        gpu_memory_utilization=config.gpu_memory_utilization,
        offload_folder=config.offload_folder
    )

    # 配置LoRA
    model = FastVisionModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=[
            # 文本解码器（LLM部分）
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
            # 视觉编码器
            "vision_encoder.layers.0.attention.qkv", "vision_encoder.layers.0.attention.proj",
            "vision_encoder.layers.1.attention.qkv", "vision_encoder.layers.1.attention.proj",
            "vision_encoder.layers.2.attention.qkv", "vision_encoder.layers.2.attention.proj",
            "vision_encoder.layers.3.attention.qkv", "vision_encoder.layers.3.attention.proj",
            "vision_encoder.conv1", "vision_encoder.conv2",
            "vision_encoder.fc1", "vision_encoder.fc2"
        ],
        lora_alpha=config.lora_rank * 2,
        lora_dropout=0.15,
        bias="lora_only",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        inference_mode=False,
    )

    # 训练优化设置
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # 训练时禁用缓存

    # 配置分词器
    tokenizer.chat_template = (
            "{% if messages[0]['role'] =='system' %}"
            "{{ messages[0]['content'] + eos_token }}"
            "{% set loop_messages = messages[1:] %}"
            "{% else %}"
            "{{ '" + config.system_prompt + "' + eos_token }}"
                                            "{% set loop_messages = messages %}"
                                            "{% endif %}"
                                            "{% for message in loop_messages %}"
                                            "{% if message['role'] == 'user' %}{{ message['content'] }}"
                                            "{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}"
                                            "{% endif %}"
                                            "{% endfor %}"
                                            "{% if add_generation_prompt %}{{ '" + config.solution_start + "' }}{% endif %}"
    )

    logger.info(f"模型加载完成，可训练参数: {model.print_trainable_parameters()}")
    return model, tokenizer


def run_sft_training(
        config: LaTeXOCRConfig,
        model: FastVisionModel,
        tokenizer: AutoTokenizer,
        train_dataset: Dataset
) -> None:
    """
    运行SFT训练

    Args:
        config: 模型配置对象
        model: 待训练模型
        tokenizer: 分词器
        train_dataset: 训练数据集
    """
    logger.info("开始SFT训练...")

    # 配置训练参数
    training_args = SFTConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        optim="adamw_8bit",
        weight_decay=config.weight_decay,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="none",
        label_smoothing_factor=config.label_smoothing_factor,
        fp16=torch.cuda.is_available(),  # 自动启用FP16加速
        max_seq_length=config.max_seq_length,
    )

    # 初始化训练器
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        args=training_args,
        peft_config=model.peft_config,  # 传递LoRA配置
    )

    # 开始训练
    trainer.train()

    # 保存模型
    logger.info(f"保存模型至 {config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # 保存训练配置
    with open(os.path.join(config.output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 初始化配置
    config = LaTeXOCRConfig(args)
    logger.info(f"训练配置: {json.dumps(config.to_dict(), ensure_ascii=False, indent=2)}")

    # 检查CUDA
    if not torch.cuda.is_available():
        logger.warning("未检测到CUDA设备，将使用CPU训练（速度可能极慢）")

    try:
        # 加载模型和分词器
        model, tokenizer = load_model_and_tokenizer(config)

        # 构建数据集
        train_dataset = build_training_dataset(config.data_json, tokenizer, config)

        # 运行SFT训练
        run_sft_training(config, model, tokenizer, train_dataset)

        logger.info("SFT训练已完成！")

    except Exception as e:
        logger.error(f"训练过程出错: {str(e)}", exc_info=True)
    finally:
        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
