# Semantic discovery vs. transactional precision: LLMs for Fashion Recommendation

This research project evaluates three distinct paradigms in recommendation systems using the **H&M Personalized Fashion Recommendations** dataset. We compare a semantic "style explorer" (LoRA-LLM) against a "behavioral specialist" (Hybrid VAE) and a traditional collaborative filtering baseline (BPR).

## 📌 Project overview
The core of our research explores the trade-off between **Precision** (accurately predicting past behaviors) and **Recall** (discovering new, stylistically relevant items).

### The models
* **`recommender_lora` (The explorer):** A Qwen-2.5-0.5B model fine-tuned via Low-Rank Adaptation (LoRA). It leverages rich metadata to map semantic style intent, achieving high recall and cross-category discovery.
* **`recommender_vae` (The specialist):** A generative Hybrid Variational Autoencoder that isolates latent consumer clusters. It excels at identifying high probability repeat purchases and transactional patterns.
* **`recommender_bpr` (The baseline):** A classic Bayesian Personalized Ranking matrix factorization approach to provide a transactional performance benchmark.

## ⚖️ Licensing & data use
* **Code:** This project's source code is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.
* **Data:** This project uses the H&M Kaggle dataset, which is restricted to **Non-Commercial, Academic Research** purposes. This repository **does not** redistribute the dataset. 
* **Compliance:** Users must download the data directly from [Kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules) and adhere to H&M's specific competition rules.

## 🛠️ Repository structure
The repository is modularly organized to keep model-specific checkpoints and pretrained weights separated:

```text
.
├── data/                       # H&M dataset (excluded from git)
├── recommender_vae/            # Hybrid VAE
│   ├── checkpoints/            # Local training saves
│   ├── pretrained/             # Final model weights
│   └── main_vae.py           # Model execution script
├── recommender_lora/           # LoRA-LLM style explorer
│   ├── checkpoints/
│   ├── pretrained/
│   └── main_lora.py
├── recommender_bpr/            # BPR baseline
│   ├── checkpoints/
│   └── main_bpr.py
├── download_pretrained.py      # Download pretrained weights
├── requirements.txt            # Python dependencies
├── LICENCE                     # Project licence
└── NOTICE                      # Legal & Data attributions

```

## 🚀 Installation & setup

1. **Clone the repository:** 
```bash
git clone https://github.com/Sami-Krim/h-m-recommendation.git
cd h-m-recommendation

```


2. **Environment setup:** 
We recommend using Conda:
```bash
conda create -n fashion-research python=3.13
conda activate fashion-research
pip install -r requirements.txt

```


3. **Data & weights preparation**

* Ensure your data/ folder contains transactions_train.csv and articles.csv.
* Run the following to fetch the pre-trained weights:
```bash
python download_pretrained.py

```


## 📖 Usage instructions

Navigate to the root directory and execute the specific script for the model you wish to test:

### 1. Hybrid VAE (behavioral specialist)

```bash
# To train
python recommender_vae/main_vae.py train --epochs 10 --lr 0.0005

# To validate
python recommender_vae/main_vae.py valid --model_name best_hybrid_vae.pt

```

### 2. LoRA-LLM (style explorer)

```bash
# To train (fine-tune with LoRA)
python recommender_lora/main_lora.py --mode finetuned --phase train

# To evaluate (uses pre-trained lora_epoch_4)
python recommender_lora/main_lora.py --mode finetuned --phase eval

# To compare with Baseline (untrained model)
python recommender_lora/main_lora.py --mode default --phase eval

```

### 3. BPR baseline

```bash
python recommender_bpr/main_bpr.py

```

## 📊 Performance comparison

| Model | Role | MAP@12 | Recall@12 | Lift (vs. BPR) |
| :--- | :--- | :--- | :--- | :--- |
| **BPR Baseline** | Baseline | 0.00560 | 0.01120 | -- |
| **Hybrid VAE** | Behavioral Specialist | 0.01203 | 0.01194 | +114.8% |
| **LoRA-LLM (Epoch 4)** | Style Explorer | **0.02008** | **0.03825** | **+258.6%** |

> **Note:** The results highlight a significant breakthrough where the fine-tuned LoRA-LLM (style explorer) outperforms purely behavioral models in both precision (MAP) and discovery (Recall), demonstrating the power of semantic understanding in fashion recommendation.

## 👥 Contributors

* **Hamame BENCHEIKH LEHOCINE**
* **Melissa MERABET**
* **Mohamed BOUTRIK**
* **Ouarda BOUMANSOUR**
* **Sami KRIM**

*Developed at Paris Cité University, Dept. of Basic and Biomedical Sciences (2025-2026).*