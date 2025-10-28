
# 📄 S1-Parser: Efficient Multimodal Document Parsing via Dynamic Structure-Semantics Fusion 

<p align="center">
          🔗 <a href="https://github.com/ScienceOne-AI/S1-Parser">Codebase</a>&nbsp&nbsp  
</p>


**S1-Parser** is a highly efficient multimodal text parsing tool designed to enable accurate and efficient parsing of complex documents. Instead of relying solely on static fine-tuning or single-stage optimization, it employs a strategy of first Supervised Fine-Tuning (SFT) then Reinforcement Learning (RL), effectively fine-tuning the model on critical aspects such as formula syntax correctness, symbol integrity, and structural rationality—balancing parsing precision and efficiency across diverse document types.

<!-- --- 
This repository implements **S1-Parser**, as described in our paper:

> *Learning : Shaping *  
--- -->

![framework1](./assets/1.jpg)

---

## 📰 News

- ***[2025/10/28]*** We release the [Code](https://github.com/ScienceOne-AI/S1-Parser) for *S1-Parser*.  


## 🚀 Features

- 🧩 **Supervised Fine-Tuning** with task-oriented ([Parse Target: Scientific Equations]) to sharpen domain adaptation.
- 🎯 **Multi-stage RL** to refine, stabilize, and accelerate the learning process in strategic of behaviors.
- 📊 Benchmarked on Scientific Literature Dataset: SCI_LLM

---

## ⚙️ Environment Setup (Recommended)

We recommend using **Python 3.10** and **PyTorch ≥ 2.7**.  
Our experimental setup follows the configuration of the [DeepScaleR](https://github.com/agentica-project/rllm/tree/deepscaler) environment.

Install the environment:
```bash
# Recommend Python 3.10.18
git clone https://github.com/ScienceOne-AI/S1-Parser.git
cd S1-Parser
pip install -r requirements.txt
```

---

Make sure to configure your model paths and data in `parse/code/run_ocr_*.sh`.



<!-- --- 


---

## 📈 Evaluation

After training, evaluate the model using:

```bash
bash scripts/eval/eval_model.sh
```

## 📊 Results


![results](./assets/31.png)
![results](./assets/52.png)
![modes](./assets/41.png)
--- -->

---

## 🔍 Acknowledgements

We build and reference on the following open source trunks, and thank the following sources for their contributions to the open source community:
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [olmOCR](https://github.com/allenai/olmocr)
- [MonkeyOCR](https://github.com/Yuliang-Liu/MonkeyOCR)
- [MistralOCR](https://mistral.ai/news/mistral-ocr)
- [Dolphin](https://github.com/ByteDance/Dolphin.git)