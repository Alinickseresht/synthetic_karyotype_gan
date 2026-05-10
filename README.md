# 🧬 Synthetic Karyotype GAN

Generating realistic synthetic karyotype images (normal and anomalous) using Generative Adversarial Networks.  
Ideal for augmenting rare chromosomal disorders datasets.

## :sparkles: Features
- Train StyleGAN2 on single chromosome images
- Generate trisomy, monosomy, translocation, ring chromosomes
- Conditionally generate anomalies (Conditional GAN)
- Mix real + synthetic data for improved classifier training

## 🗂 Dataset (example)
We use [BioImLAB Kaggle dataset](https://www.kaggle.com/datasets/...).  
Download and place in data/raw/ (or use the automatic script).

## :rocket: Quick Start
`bash
git clone https://github.com/Alinickseresht/synthetic-karyotype-gan
cd synthetic-karyotype-gan
pip install -r requirements.txt
python scripts/download_data.sh   # optional
python src/train.py --config config.yaml
python src/generate.py --num 1000 --output data/synthetic/
