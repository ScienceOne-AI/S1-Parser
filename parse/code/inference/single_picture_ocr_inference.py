# -*- coding: utf-8 -*-
"""
LaTeX OCR 图片转LaTeX工具
基于VL模型+LoRA微调实现，专注于单张化学/数学公式的OCR识别与LaTeX转换

使用说明：
1. 安装依赖：pip install transformers peft torch
2. 基础使用：python script_name.py --image_path ./pic/middle_school_chemistry.png
3. 自定义路径：python script_name.py --base_model_dir /path/to/base/model --lora_model_dir /path/to/lora/model --image_path /path/to/image.png

命令行参数说明：
--base_model_dir: 基础模型路径
--lora_model_dir: LoRA微调模型路径
--image_path: 待处理图片路径（默认：./pic/middle_school_chemistry.png）
--max_new_tokens: 生成文本最大长度（默认：1024）
--resize_height: 图片缩放高度（默认：100）
--resize_width: 图片缩放宽度（默认：500）
"""

import os
import sys
import argparse
import logging
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel, LoraConfig, TaskType

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ===================== LoRA配置 =====================
def get_lora_config() -> LoraConfig:
    """获取LoRA微调配置（保持原有配置不变）"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=True,
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
    )


# ===================== 命令行参数解析 =====================
def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="BedRockOCR VL LaTeX OCR 图片转LaTeX工具")

    # 模型路径配置
    parser.add_argument("--base_model_dir", type=str,
                        default="/model/",
                        help="基础模型路径")
    parser.add_argument("--lora_model_dir", type=str,
                        default="/model/output/chemistry-sft-tuned-2025-11-06",
                        help="LoRA微调模型路径")

    # 图片相关配置
    parser.add_argument("--image_path", type=str,
                        default="./pic/middle_school_chemistry.png",
                        help="待处理图片路径")
    parser.add_argument("--resize_height", type=int,
                        default=100,
                        help="图片缩放高度")
    parser.add_argument("--resize_width", type=int,
                        default=500,
                        help="图片缩放宽度")

    # 生成配置
    parser.add_argument("--max_new_tokens", type=int,
                        default=1024,
                        help="生成文本最大长度")

    return parser.parse_args()


# ===================== 模型加载 =====================
def load_model_and_processor(args: argparse.Namespace) -> tuple[AutoModelForVision2Seq, AutoProcessor]:
    """
    加载基础模型、LoRA适配器和处理器

    Args:
        args: 命令行参数

    Returns:
        tuple: (加载好的模型, 处理器)
    """
    # 加载基础模型
    logger.info(f"正在加载基础模型：{args.base_model_dir}")
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.enable_input_require_grads()

    # 加载LoRA适配器
    logger.info(f"正在加载LoRA适配器：{args.lora_model_dir}")
    lora_config = get_lora_config()
    model = PeftModel.from_pretrained(model, model_id=args.lora_model_dir, config=lora_config)

    # 加载处理器
    logger.info(f"正在加载处理器：{args.base_model_dir}")
    processor = AutoProcessor.from_pretrained(args.base_model_dir)

    return model, processor


# ===================== 输入准备 =====================
def prepare_model_inputs(
        args: argparse.Namespace,
        processor: AutoProcessor
) -> dict:
    """
    准备模型输入数据

    Args:
        args: 命令行参数
        processor: 模型处理器

    Returns:
        dict: 模型输入字典（已移至CUDA）
    """
    # 提示词
    prompt = "You are a LaTeX OCR assistant. Your goal is to read images provided by users and convert them into standard LaTeX"

    # 构建消息格式
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": args.image_path,
                    "resized_height": args.resize_height,
                    "resized_width": args.resize_width,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # 处理输入数据
    logger.info(f"正在处理图片输入：{args.image_path}")
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    return inputs.to("cuda")


# ===================== 结果生成 =====================
def generate_latex_result(
        args: argparse.Namespace,
        model: AutoModelForVision2Seq,
        processor: AutoProcessor,
        inputs: dict
) -> str:
    """
    生成LaTeX结果

    Args:
        args: 命令行参数
        model: 加载好的模型
        processor: 模型处理器
        inputs: 模型输入数据

    Returns:
        str: 生成的LaTeX代码
    """
    logger.info("正在生成LaTeX结果...")
    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    # 裁剪输入部分，只保留生成内容
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # 解码生成结果
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]


# ===================== 图片校验 =====================
def validate_image_path(image_path: str) -> None:
    """
    校验图片文件是否存在

    Args:
        image_path: 图片路径

    Raises:
        FileNotFoundError: 图片文件不存在时抛出
    """
    if not os.path.exists(image_path):
        logger.error(f"图片文件不存在：{image_path}")
        raise FileNotFoundError(f"图片文件不存在：{image_path}")
    logger.info(f"图片文件校验通过：{image_path}")


# ===================== 主函数 =====================
def main():
    """主函数：执行完整的图片转LaTeX流程"""
    try:
        # 解析命令行参数
        args = parse_args()

        # 校验图片路径
        validate_image_path(args.image_path)

        # 加载CUDA内存优化配置
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        logger.info("CUDA内存优化配置已启用")

        # 加载模型和处理器
        model, processor = load_model_and_processor(args)

        # 准备输入数据
        inputs = prepare_model_inputs(args, processor)

        # 生成LaTeX结果
        latex_result = generate_latex_result(args, model, processor, inputs)

        # 打印结果
        print("\n" + "=" * 80)
        print("图片转换后的LaTeX内容：")
        print("=" * 80)
        print(latex_result)
        print("=" * 80)
        logger.info("LaTeX转换完成！")

    except Exception as e:
        logger.error(f"程序执行出错：{str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
