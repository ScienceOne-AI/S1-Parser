# -*- coding: utf-8 -*-
"""
OCR 数据集合并工具
用于合并多个OCR相关数据集，生成统一格式的JSON数据集
支持train/val/test划分，自动打乱和采样
"""

import pandas as pd
import json
import os
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


# ========== 配置项（可根据实际情况修改） ==========
@dataclass
class DatasetConfig:
    """数据集配置类"""
    # 输出目录（默认：当前目录下的output文件夹）
    output_dir: str = "./output"
    # 采样比例（默认：使用90%的数据）
    sample_ratio: float = 0.9
    # 随机种子（保证结果可复现）
    random_seed: int = 42
    # 数据集路径配置
    datasets: Dict[str, Dict[str, str]] = None


# 初始化默认配置（移除隐私路径，使用占位符提示用户配置）
DEFAULT_CONFIG = DatasetConfig(
    datasets={
        "ocr_message": {  # 印刷体OCR数据集
            "train": "path/to/ocr_message/train.csv",
            "validation": "path/to/ocr_message/validation.csv",
            "test": "path/to/ocr_message/test.csv"
        },
        "handwrite": {  # 手写LaTeX数据集
            "train": "path/to/ocr_handwrite/train.csv",
            "validation": "path/to/ocr_handwrite/validation.csv",
            "test": "path/to/ocr_handwrite/test.csv"
        },
        "ocr_drug": {  # 文本OCR数据集
            "train": "path/to/ocr_drug/train.csv",
            "validation": "path/to/ocr_drug/validation.csv",
            "test": "path/to/ocr_drug/test.csv"
        }
    }
)


# ========== 工具函数 ==========
def setup_random_seed(seed: int) -> None:
    """设置随机种子以保证结果可复现"""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """加载CSV数据集，包含基础的异常处理"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"读取CSV失败: {csv_path}, 错误: {str(e)}") from e

    # 验证必要的列是否存在
    required_columns = ["image_path", "text"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV文件缺少必要列: {missing_cols}, 必需列: {required_columns}")

    return df


def create_conversation_samples(
        df: pd.DataFrame,
        dataset_prefix: str,
        split_type: str,
        start_idx: int = 0
) -> List[Dict[str, Any]]:
    """
    将CSV数据转换为对话格式样本

    Args:
        df: 输入DataFrame
        dataset_prefix: 数据集前缀（用于区分不同来源）
        split_type: 数据划分类型（train/val/test）
        start_idx: 起始索引（用于生成唯一ID）

    Returns:
        对话格式的样本列表
    """
    samples = []
    for idx in range(len(df)):
        sample_id = f"{dataset_prefix}_{split_type}_{start_idx + idx + 1}"
        samples.append({
            "id": sample_id,
            "conversations": [
                {
                    "role": "user",
                    "value": str(df.iloc[idx]["image_path"]).strip()
                },
                {
                    "role": "assistant",
                    "value": str(df.iloc[idx]["text"]).strip()
                }
            ]
        })
    return samples


def merge_datasets(
        config: DatasetConfig,
        split_type: str,
        output_path: str
) -> None:
    """
    合并单个划分（train/val/test）的所有数据集

    Args:
        config: 配置对象
        split_type: 数据划分类型
        output_path: 输出文件路径
    """
    all_samples = []
    current_idx = 0

    # 遍历所有数据集
    for dataset_name, dataset_paths in config.datasets.items():
        # 获取当前划分的CSV路径
        csv_path = dataset_paths.get(split_type)
        if not csv_path:
            print(f"警告: {dataset_name} 数据集缺少 {split_type} 划分，跳过")
            continue

        # 加载数据
        print(f"正在加载 {dataset_name} - {split_type} 数据集: {csv_path}")
        df = load_csv_data(csv_path)

        # 转换为对话格式
        samples = create_conversation_samples(
            df=df,
            dataset_prefix=dataset_name,
            split_type=split_type,
            start_idx=current_idx
        )

        all_samples.extend(samples)
        current_idx += len(df)
        print(f"已加载 {len(samples)} 个样本")

    # 数据处理：打乱和采样
    print(f"\n合并后总样本数: {len(all_samples)}")
    setup_random_seed(config.random_seed)
    random.shuffle(all_samples)

    # 按比例采样
    sample_count = int(len(all_samples) * config.sample_ratio)
    selected_samples = all_samples[:sample_count]

    # 再次打乱确保随机性
    random.shuffle(selected_samples)

    # 保存为JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(selected_samples, f, ensure_ascii=False, indent=2)
        print(f"成功保存到: {output_path}")
        print(f"最终选择样本数: {len(selected_samples)}\n")
    except Exception as e:
        raise RuntimeError(f"保存JSON失败: {output_path}, 错误: {str(e)}") from e


def main(config: Optional[DatasetConfig] = None) -> None:
    """主函数：执行所有数据集的合并"""
    # 使用默认配置或自定义配置
    if config is None:
        config = DEFAULT_CONFIG

    # 设置随机种子
    setup_random_seed(config.random_seed)

    # 定义输出路径
    output_paths = {
        "train": os.path.join(config.output_dir, "json", "ocr_train.json"),
        "validation": os.path.join(config.output_dir, "json", "ocr_val.json"),
        "test": os.path.join(config.output_dir, "json", "ocr_test.json")
    }

    # 处理所有划分
    for split_type in ["train", "validation", "test"]:
        print(f"===== 开始处理 {split_type} 划分 =====")
        merge_datasets(
            config=config,
            split_type=split_type,
            output_path=output_paths[split_type]
        )

    print("===== 所有数据集处理完成！=====")
    print(f"输出目录: {config.output_dir}")


if __name__ == "__main__":
    # 用户需要在此处修改配置，或通过代码传入自定义配置
    # 示例：使用自定义路径配置
    # custom_config = DatasetConfig(
    #     output_dir="/path/to/output",
    #     sample_ratio=0.9,
    #     datasets={
    #         "small": {
    #             "train": "/path/to/small/train.csv",
    #             "validation": "/path/to/small/val.csv",
    #             "test": "/path/to/small/test.csv"
    #         },
    #         # ... 其他数据集配置
    #     }
    # )
    # main(custom_config)
    main()
