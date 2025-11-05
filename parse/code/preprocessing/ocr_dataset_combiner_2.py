# -*- coding: utf-8 -*-
"""
OCR 自建数据追加工具
====================

将自建图片+LaTeX 标签追加到已有训练集，生成统一对话格式 JSON。

主要特性：
- 配置化（DatasetConfig）
- 随机种子固定，结果可复现
- 完善的日志与异常处理
- 自动创建输出目录
- 追加后全局打乱并重新编号 index

使用方法：
1. 修改 DEFAULT_CONFIG 中的路径
2. python ocr_dataset_combiner.py
"""

import json
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


# ---------- 配置 ----------
@dataclass
class DatasetConfig:
    """自建数据追加配置"""
    image_dir: str = "/data/data/pic"
    label_file: str = "/data//data/label.json"
    existing_train_json: str = "/data/mergedata/json/latex_ocr_train.json"
    output_json: str = "/data/mergedata/json/latex_ocr_train_add.json"
    random_seed: int = 42
    overwrite: bool = True
    verbose: bool = True


# ---------- 工具 ----------
def set_seed(seed: int) -> None:
    random.seed(seed)


def log(msg: str, cfg: DatasetConfig) -> None:
    if cfg.verbose:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}]  {msg}")


def validate(cfg: DatasetConfig) -> None:
    if not Path(cfg.image_dir).is_dir():
        raise FileNotFoundError(f"图片目录不存在: {cfg.image_dir}")
    if not Path(cfg.label_file).is_file():
        raise FileNotFoundError(f"标签文件不存在: {cfg.label_file}")
    if Path(cfg.existing_train_json).is_file() and not cfg.overwrite:
        raise FileExistsError(f"已有训练集且 overwrite=False: {cfg.existing_train_json}")


def load_labels(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sorted_image_files(dir_: Path) -> List[str]:
    files = [f for f in dir_.iterdir() if f.suffix.lower() in (".png", ".jpg")]
    files.sort(key=lambda x: int(re.search(r"(\d+)", x.stem).group(1)))
    return [f.name for f in files]


def build_conversations(image_files: List[str],
                        labels: List[str],
                        image_dir: str,
                        prefix: str = "self_message") -> List[Dict[str, Any]]:
    if len(image_files) != len(labels):
        raise ValueError(f"图片数({len(image_files)}) ≠ 标签数({len(labels)})")
    conversations = []
    for idx, (img, lab) in enumerate(zip(image_files, labels), 1):
        conversations.append({
            "id": f"{prefix}_{idx}",
            "conversations": [
                {"role": "user", "value": str(Path(image_dir) / img)},
                {"role": "assistant", "value": str(lab)}
            ]
        })
    return conversations


def load_existing(path: str) -> List[Dict[str, Any]]:
    if not Path(path).is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: List[Dict[str, Any]], path: str, cfg: DatasetConfig) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log(f"已保存 -> {path}  （共 {len(data)} 条）", cfg)


# ---------- 主流程 ----------
def main(cfg: DatasetConfig | None = None) -> None:
    if cfg is None:
        cfg = DatasetConfig()

    log("===== OCR 自建数据追加开始 =====", cfg)
    validate(cfg)
    set_seed(cfg.random_seed)

    labels = load_labels(cfg.label_file)
    image_files = sorted_image_files(Path(cfg.image_dir))
    log(f"加载 {len(image_files)} 张图片，{len(labels)} 条标签", cfg)

    new_samples = build_conversations(image_files, labels, cfg.image_dir)
    existing = load_existing(cfg.existing_train_json)
    merged = existing + new_samples

    random.shuffle(merged)
    for idx, item in enumerate(merged, 1):
        item["index"] = idx

    save_json(merged, cfg.output_json, cfg)
    log("===== 全部完成 =====", cfg)


if __name__ == "__main__":
    main()
