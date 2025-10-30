# -*- coding: utf-8 -*-
"""
Group Relative Policy Optimization（GRPO）训练框架用于LaTeX OCR优化
基于强化学习的公式识别模型微调，解决梯度爆炸和数值不稳定问题
"""
import os
import json
import re
import gc
import logging
import argparse
import torch
import numpy as np
from PIL import Image
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sacrebleu import sentence_bleu
from Levenshtein import ratio
from transformers import AutoTokenizer, GenerationConfig
from unsloth import FastVisionModel
from trl import GRPOConfig, GRPOTrainer
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


# ============= 日志配置 =============
def setup_logging():
    """配置日志输出格式和级别"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


# ============= 命令行参数解析 =============
def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LaTeX OCR模型 GRPO 训练脚本")

    # 路径配置
    parser.add_argument("--base-model-dir", type=str,
                        default="/data/Recognition",
                        help="基础模型路径")
    parser.add_argument("--sft-output-dir", type=str,
                        default="./output/sft-tuned/",
                        help="SFT模型输出路径")
    parser.add_argument("--grpo-output-dir", type=str,
                        default="./output/grpo-rl-optimized",
                        help="GRPO模型输出路径")
    parser.add_argument("--data-json", type=str,
                        default="/data/mergedata/json/latex_ocr_train.json",
                        help="训练数据JSON文件路径")
    parser.add_argument("--offload-folder", type=str, default="./offload",
                        help="模型卸载缓存目录")

    # 模型参数
    parser.add_argument("--max-seq-length", type=int, default=1024,
                        help="最大序列长度")
    parser.add_argument("--max-new-tokens", type=int, default=160,
                        help="生成文本最大长度")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA秩参数")
    parser.add_argument("--resize-h", type=int, default=448,
                        help="图像缩放高度")
    parser.add_argument("--resize-w", type=int, default=448,
                        help="图像缩放宽度")
    parser.add_argument("--load-in-4bit", action="store_true", default=True,
                        help="是否使用4bit量化加载模型")

    # 训练参数
    parser.add_argument("--batch-size", type=int, default=2,
                        help="每设备训练批次大小")
    parser.add_argument("--gradient-accumulation", type=int, default=2,
                        help="梯度累积步数")
    parser.add_argument("--epochs", type=int, default=5,
                        help="训练轮数")
    parser.add_argument("--learning-rate", type=float, default=2e-6,
                        help="学习率")
    parser.add_argument("--max-grad-norm", type=float, default=0.3,
                        help="梯度裁剪最大范数")
    parser.add_argument("--warmup-steps", type=int, default=200,
                        help="学习率预热步数")
    parser.add_argument("--eval-steps", type=int, default=2,
                        help="评估步数间隔")
    parser.add_argument("--logging-steps", type=int, default=2,
                        help="日志输出步数间隔")

    # 生成参数
    parser.add_argument("--temperature", type=float, default=1.5,
                        help="生成温度参数")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="top-p采样参数")
    parser.add_argument("--num-generations", type=int, default=4,
                        help="每样本生成数量")

    return parser.parse_args()


# ============= 配置参数 =============
class Config:
    """训练配置参数容器"""

    def __init__(self, args: argparse.Namespace):
        # 路径配置
        self.base_model_dir = args.base_model_dir
        self.sft_output_dir = args.sft_output_dir
        self.grpo_output_dir = args.grpo_output_dir
        self.data_json = args.data_json
        self.offload_folder = args.offload_folder

        # 模型参数
        self.max_seq_length = args.max_seq_length
        self.max_new_tokens = args.max_new_tokens
        self.lora_rank = args.lora_rank
        self.resize_h = args.resize_h
        self.resize_w = args.resize_w
        self.load_in_4bit = args.load_in_4bit

        # 训练参数
        self.batch_size = args.batch_size
        self.gradient_accumulation = args.gradient_accumulation
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.max_grad_norm = args.max_grad_norm
        self.warmup_steps = args.warmup_steps
        self.eval_steps = args.eval_steps
        self.logging_steps = args.logging_steps

        # 生成参数
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.num_generations = args.num_generations

        # 提示词模板
        self.system_prompt = "You are a LaTeX OCR assistant. Convert images to LaTeX."
        self.solution_start = "<SOLUTION>"
        self.solution_end = "</SOLUTION>"
        self.chat_template = (
                "{% if messages[0]['role'] != 'system' %}{{ '" + self.system_prompt + "' }}{% endif %}"
                                                                                      "{% for message in messages %}"
                                                                                      "{% if message['role'] == 'user' %}{{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}"
                                                                                      "{% endfor %}"
                                                                                      "{% if add_generation_prompt %}{{ '" + self.solution_start + "' }}{% endif %}"
        )


# ============= 模型工具 =============
def load_model(config: Config, logger: logging.Logger, sft_path: str = None, as_base: bool = False):
    """加载模型并配置LoRA"""
    logger.info(f"加载模型: {config.base_model_dir}, 4bit量化: {config.load_in_4bit}")

    # 加载基础模型
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=config.base_model_dir,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        offload_folder=config.offload_folder
    )
    tokenizer.chat_template = config.chat_template

    if as_base:  # 纯推理模式（不加载LoRA）
        model.config.use_cache = False
        logger.info("基础模型加载完成（不包含LoRA）")
        return model, tokenizer

    # 配置LoRA适配器（包含视觉和文本层）
    model = FastVisionModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=[
            # 文本层
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
            # 视觉层
            "vision_model.encoder.layers.2.self_attn.q_proj",
            "vision_model.encoder.layers.2.self_attn.v_proj",
            "vision_model.encoder.layers.5.self_attn.q_proj",
            "vision_model.encoder.layers.5.self_attn.v_proj",
            "vision_model.encoder.layers.8.self_attn.q_proj",
            "vision_model.encoder.layers.8.self_attn.v_proj",
            "vision_model.fc_norm",
            "vision_model.post_layernorm"
        ],
        lora_alpha=config.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407
    )

    # 加载SFT权重
    if sft_path:
        logger.info(f"加载SFT权重: {sft_path}")
        model.load_adapter(sft_path, adapter_name="sft")
        model.set_adapter("sft")

    model.config.use_cache = False
    logger.info("训练模型加载完成（包含LoRA配置）")
    return model, tokenizer


# ============= 数据处理 =============
def build_dataset(config: Config, tokenizer, logger: logging.Logger):
    """构建训练和验证数据集"""
    logger.info(f"加载数据集: {config.data_json}")

    # 加载原始数据
    with open(config.data_json, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    train_raw, eval_raw = train_test_split(raw_data, test_size=0.1, random_state=42)
    logger.info(f"原始数据拆分完成 - 训练集: {len(train_raw)}, 验证集: {len(eval_raw)}")

    # 图像预处理
    img_transform = transforms.Compose([
        transforms.Resize((config.resize_h, config.resize_w),
                          antialias=True,
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomAffine(degrees=3, translate=(0.03, 0.03)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def _generate_dataset(subset, subset_name: str):
        """生成数据集的内部函数"""

        def generator():
            valid_count = 0
            for sample in subset:
                # 数据格式校验
                if not isinstance(sample, dict) or "conversations" not in sample:
                    continue
                convs = sample["conversations"]
                if len(convs) < 2:
                    continue

                # 提取数据
                try:
                    img_path = convs[0]["value"]
                    formula = convs[1]["value"].strip()
                    if not formula:
                        continue

                    # 处理图像
                    image = Image.open(img_path).convert("RGB")
                    image = img_transform(image)
                except Exception as e:
                    logger.warning(f"数据处理失败: {e}, 跳过样本")
                    continue

                # 构建提示词
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": config.system_prompt},
                    ]
                }]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                full_text = prompt + f"{config.solution_start}{formula}{config.solution_end}"

                valid_count += 1
                yield {"text": full_text, "prompt": prompt, "answer": formula}

            logger.info(f"{subset_name}数据集处理完成 - 有效样本: {valid_count}")

        return Dataset.from_generator(generator)

    return (
        _generate_dataset(train_raw, "训练"),
        _generate_dataset(eval_raw, "验证")
    )


# ============= 奖励函数 =============
class RewardCalculator:
    """奖励计算工具类"""

    @staticmethod
    def _math_symbol_match(pred: str, gt: str) -> float:
        """计算数学符号匹配度（Jaccard相似度）"""
        math_symbols = r'\\frac|\\sum|\\int|\\sqrt|\\alpha|\\beta|\\gamma|\\pi|\\cdot|\\times|\\leq|\\geq|\\neq|\\equiv|\\partial|\\nabla'
        pred_syms = set(re.findall(math_symbols, pred))
        gt_syms = set(re.findall(math_symbols, gt))

        if not gt_syms:
            return 0.8
        intersection = len(pred_syms & gt_syms)
        union = len(pred_syms | gt_syms)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _latex_syntax_score(pred: str) -> float:
        """评估LaTeX语法正确性"""
        score = 0.0

        # 数学环境检查
        math_envs = [(r'\(', r'\)'), (r'\[', r'\]'), (r'\\begin\{equation\}', r'\\end\{equation\}')]
        has_valid_env = any(env_start in pred and env_end in pred for env_start, env_end in math_envs)
        score += 0.4 if has_valid_env else 0.0

        # 括号平衡检查
        brace_types = {'{': '}', '[': ']', '(': ')'}
        balance_score = 1.0
        for open_b, close_b in brace_types.items():
            if pred.count(open_b) != pred.count(close_b):
                balance_score -= 0.1
        score += 0.3 * max(balance_score, 0.0)

        # 无效命令检查
        invalid_cmd = re.findall(r'(?<!\\)(?<!\w)[a-zA-Z](?!\w)', pred)
        score += 0.3 * (1.0 - min(len(invalid_cmd) / 5, 1.0))

        return min(score, 1.0)

    @classmethod
    def dense_reward(cls, completions, prompts, answer, **kwargs):
        """密集奖励计算主函数"""
        rewards = []
        for pred, gt in zip(completions, answer):
            # 提取预测内容
            pred = pred.get("content", "").strip() if isinstance(pred, dict) else str(pred).strip()
            gt = gt.strip()

            # 完全匹配奖励
            if pred == gt:
                rewards.append(8.0)
                continue

            # 短文本惩罚
            min_len = max(3, len(gt) // 3)
            if len(pred) < min_len:
                rewards.append(-0.05)
                continue

            # 综合得分计算
            lev_score = ratio(pred, gt)
            syntax_score = cls._latex_syntax_score(pred)
            symbol_score = cls._math_symbol_match(pred, gt)

            # 基础得分
            base_score = lev_score * 0.2 + syntax_score * 0.4 + symbol_score * 0.4

            # 分段奖励（增强生成过程反馈）
            step = max(len(gt) // 5, 2)
            sub_scores = []
            for i in range(step, min(len(pred), len(gt) * 2) + step, step):
                sub_pred = pred[:i]
                sub_lev = ratio(sub_pred, gt[:min(i, len(gt))])
                sub_syntax = cls._latex_syntax_score(sub_pred)
                sub_scores.append((sub_lev * 0.5 + sub_syntax * 0.5) * 1.2)

            # 最终奖励
            segment_score = np.mean(sub_scores) if sub_scores else 0.0
            final_reward = (base_score * 0.6 + segment_score * 0.4) ** 0.6 * 4.0 - 0.8
            rewards.append(float(np.clip(final_reward, -0.5, 6.0)))

        # 奖励归一化
        rewards = np.array(rewards)
        return np.clip(rewards, -1.0, 8.0).tolist()


# ============= 评估工具 =============
def make_compute_metrics(base_model, base_tokenizer, max_new_tokens: int, logger: logging.Logger):
    """创建评估指标计算函数"""
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    logger.info("评估函数初始化完成，基础模型已设置为推理模式")

    # 生成配置
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=2.0,
        do_sample=True,
        pad_token_id=base_tokenizer.pad_token_id,
        eos_token_id=base_tokenizer.eos_token_id,
        early_stopping=True
    )

    def _compute_bleu(model, tokenizer, dataset, model_name: str):
        """计算BLEU分数"""
        logger.info(f"开始{model_name}模型BLEU评估...")
        preds, refs = [], []
        for batch in tqdm(DataLoader(dataset, batch_size=1), desc=f"{model_name}评估"):
            prompt = batch["prompt"][0]
            ref = batch["answer"][0]

            # 生成预测
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, generation_config=gen_cfg)

            # 解码结果
            pred = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
            preds.append(pred)
            refs.append(ref)

        # 计算BLEU
        bleu_score = np.mean([
            sentence_bleu(p, [r], tokenize='char').score / 100
            for p, r in zip(preds, refs)
        ])
        logger.info(f"{model_name}模型BLEU分数: {bleu_score:.4f}")
        return bleu_score

    def compute_metrics(eval_preds):
        """评估指标主函数"""
        grpo_bleu = _compute_bleu(
            compute_metrics.grpo_model,
            compute_metrics.grpo_tokenizer,
            compute_metrics.eval_ds,
            "GRPO"
        )
        base_bleu = _compute_bleu(
            base_model,
            base_tokenizer,
            compute_metrics.eval_ds,
            "基础"
        )

        logger.info(f"BLEU对比 | 基础: {base_bleu:.4f}, GRPO: {grpo_bleu:.4f}, 差值: {grpo_bleu - base_bleu:+.4f}")
        return {"eval_bleu": grpo_bleu, "eval_loss": 1.0 - grpo_bleu}

    return compute_metrics


# ============= 训练流程 =============
class StableGRPOTrainer(GRPOTrainer):
    """增强稳定性的GRPO训练器"""

    def __init__(self, logger: logging.Logger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        try:
            # 计算基础损失
            loss = super().compute_loss(model, inputs, return_outputs=False, **kwargs)

            # 清理异常值
            if hasattr(self, "kl_meter") and self.kl_meter.val is not None:
                self.kl_meter.val = torch.nan_to_num(self.kl_meter.val, nan=0.0, posinf=0.0, neginf=0.0)

            # 梯度裁剪和损失保护
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.args.max_grad_norm)
            loss = torch.nan_to_num(loss, nan=2.0, posinf=5.0, neginf=5.0)
            loss = torch.clamp(loss, 0.01, 10.0)

            torch.cuda.empty_cache()
            return loss.requires_grad_() if not loss.requires_grad else loss

        except Exception as e:
            self.logger.error(f"损失计算异常: {e}, 使用安全损失值")
            return torch.tensor(1.0, requires_grad=True, device=model.device)


def run_training():
    """主训练流程"""
    # 初始化日志和参数
    logger = setup_logging()
    args = parse_args()
    config = Config(args)

    # 创建输出目录
    os.makedirs(config.grpo_output_dir, exist_ok=True)
    os.makedirs(config.offload_folder, exist_ok=True)
    logger.info(f"输出目录: {config.grpo_output_dir}, 卸载缓存: {config.offload_folder}")

    # 加载基础模型（用于评估）
    base_model, base_tokenizer = load_model(config, logger, as_base=True)

    # 加载带LoRA的训练模型
    model, tokenizer = load_model(config, logger, sft_path=config.sft_output_dir)

    # 构建数据集
    train_ds, eval_ds = build_dataset(config, tokenizer, logger)
    logger.info(f"数据集准备完成 - 训练集: {len(train_ds)}, 验证集: {len(eval_ds)}")

    # 配置训练参数
    training_args = GRPOConfig(
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        optim="adamw_8bit",
        save_steps=200000,  # 大步数避免频繁保存
        temperature=config.temperature,
        top_p=config.top_p,
        num_generations=config.num_generations,
        max_completion_length=config.max_new_tokens,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        warmup_steps=config.warmup_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=True,
        logging_steps=config.logging_steps,
        report_to="none",
        output_dir=config.grpo_output_dir,
        lr_scheduler_type="cosine",
    )

    # 准备评估函数
    compute_metrics_fn = make_compute_metrics(base_model, base_tokenizer, config.max_new_tokens, logger)
    compute_metrics_fn.eval_ds = eval_ds
    compute_metrics_fn.grpo_model = model
    compute_metrics_fn.grpo_tokenizer = tokenizer

    # 初始化训练器
    trainer = StableGRPOTrainer(
        logger=logger,
        model=model,
        processing_class=tokenizer,
        reward_funcs=[RewardCalculator.dense_reward],
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics_fn,
        data_collator=lambda data: {
            "prompt": [d["prompt"] for d in data],
            "answer": [d["answer"] for d in data],
        },
        kl_penalty="kl",
        kl_coef=0.02,
        adaptive_kl=True,
        target_kl=0.08,
    )

    # 开始训练
    logger.info("启动GRPO训练...")
    trainer.train()

    # 保存模型
    model.save_pretrained(config.grpo_output_dir)
    tokenizer.save_pretrained(config.grpo_output_dir)
    logger.info(f"模型已保存至: {config.grpo_output_dir}")

    # 清理资源
    del model, tokenizer, base_model, base_tokenizer, train_ds, eval_ds
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("训练完成，资源已清理")


if __name__ == "__main__":
    run_training()
