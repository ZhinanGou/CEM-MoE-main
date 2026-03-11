# CEM-MoE: Empathetic Dialogue Generation via Dual-Stream Sparse MoE and Semantic Density-Aware Fusion

This repository contains the official PyTorch implementation of the paper: **Empathetic Dialogue Generation via Dual-Stream Sparse MoE and Semantic Density-Aware Fusion**.

<img width="850" height="480" alt="CEM-MoE" src="https://github.com/user-attachments/assets/fa89af18-7f92-430f-bd1d-43ab40803795" />



## 📝 Abstract
In multi-turn empathetic dialogue generation, achieving precise emotion perception under limited computational resources remains a significant challenge. To address this, we propose **CEM-MoE**, a resource-efficient model featuring a **Dual-Stream Sparse Mixture-of-Experts (MoE)** Emotion Decoder and a **Semantic Density-based Adaptive Soft Emotion Fusion** mechanism. The decoder enhances mixed emotion decoupling via expert specialization, while the fusion mechanism resolves emotional inconsistency by dynamically weighting historical and current cues. Experiments on the *Empathetic Dialogues* dataset demonstrate that CEM-MoE significantly outperforms existing baselines, achieving high performance using only consumer-grade GPUs.

## 🚀 Main Contributions
- **Dual-Stream Sparse MoE Decoder**: Breaks through the emotional representation bottleneck of lightweight models.
- **Adaptive Emotion Soft Fusion**: A learnable dual-channel strategy based on semantic density to precisely fit emotional flow.
- **High Efficiency**: Achieves State-of-the-Art performance (PPL: 36.59, Acc: 34.23%) on a single consumer-grade GPU (RTX 3090).

## ⚙️ Requirements
- Python >= 3.8
- PyTorch >= 1.10.0
- Transformers
- Numpy

Install the required packages:
```bash
pip install -r requirements.txt
```
## 📂 Dataset & Pre-trained Word Vectors
1. Dataset: We use the public EmpatheticDialogues dataset. Please download the dataset and place it in the ./data/ directory.

2. GloVe Vectors: Our model uses GloVe 300-dimensional word embeddings. Please download the pre-trained vectors from the official Stanford NLP website:

Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/

Extract the glove.6B.300d.txt file.

Place it in the ./vectors/ directory.

## 🏃‍♂️ Usage
1. Training
To train the CEM-MoE model from scratch, run:
```bash
python main.py --batch_size 16 --learning_rate 1e-4 --num_experts 4 --top_k 2
```
## 📎 Citation
If you find this code or our paper useful for your research, please cite:

```bibtex
@article{CEM-MoE-2026,
  title={Empathetic dialogue generation via dual-stream sparse MoE and semantic density-aware fusion},
  author={Gou, Zhinan and Liu, He and Wang, Yufan},
  journal={Int. J. [Journal Name]},
  volume={X},
  number={Y},
  pages={xxx--xxx},
  year={2026}
}
