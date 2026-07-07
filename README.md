# BERT-based Cyber-Threat Detection

> Exploring the performance of a Large Language Model (BERT) in cyber-threat detection tasks, with Explainable AI support via Integrated Gradients.

This repository contains the code developed for my Bachelor's thesis in Computer Science at the **Università degli Studi di Bari Aldo Moro**, titled *"Exploring the performance of a LLM in cyber-threat detection tasks"*.

The project explores whether a Large Language Model — specifically **BERT-Medium** — can be adapted to detect cyber-threats by converting tabular, feature-based cybersecurity data into natural language sentences ("storytelling"), and fine-tuning BERT to classify them. Model decisions are then explained using **Integrated Gradients (IG)**, an Explainable AI (XAI) technique.

## Table of Contents

- [Overview](#overview)
- [Approach](#approach)
- [Datasets](#datasets)
- [Repository Structure](#repository-structure)
- [Results](#results)
- [Explainability (XAI)](#explainability-xai)
- [Getting Started](#getting-started)
- [Tech Stack](#tech-stack)
- [Future Work](#future-work)
- [Authors & Supervisors](#authors--supervisors)
- [License](#license)

## Overview

Traditional cybersecurity systems typically rely on detecting known patterns of malicious behavior, which makes them ineffective against novel or evolving attacks. This project investigates an alternative approach: since BERT excels at understanding contextual relationships in text, cybersecurity feature vectors (network traffic records, Android app static/dynamic features) are **translated into descriptive sentences**, allowing BERT to learn the "language" of cyber-threats.

Two classification tasks are addressed:

- **Binary classification** — Normal vs. Abnormal traffic / Benign vs. Malware apps
- **Multiclass classification** — Identification of the specific malware family

## Approach

The pipeline consists of two phases:

**1. Training Phase**
`Feature vector → Storytelling (feature-to-sentence conversion) → BERT tokenization → Fine-tuning of BERT-Medium with a custom classification head`

**2. Test Phase**
`Single feature vector → Storytelling → BERT-based prediction → Explanation via Integrated Gradients`

### Storytelling of feature vectors

Each record is converted from a structured feature vector into a natural language sentence using the pattern:

```
[Feature Name] is [Value], [Feature Name] is [Value], ...
```

For example:

```
Input:  {duration: 0, protocol_type: tcp, service: private, flag: REJ, src_bytes: 0}

Output: "length of the connection is 0, type of the protocol is tcp,
         network service is private, status of connection is REJ,
         source bytes to destination is 0"
```

### Model

- **Backbone:** [BERT-Medium](https://huggingface.co/prajjwal1/bert-medium) (L=8, H=512, A=8, ~41M parameters) — chosen for its efficiency compared to BERT-Base/Large.
- **Classification head:** `[CLS]` token → Dense layer (128 neurons) → ReLU → output layer
  - Binary task → 1 neuron, Sigmoid activation
  - Multiclass task → N neurons (one per class)
- **Loss functions:** Binary Cross-Entropy (binary task), Cross-Entropy (multiclass task)
- **Optimizer:** AdamW, learning rate `1e-5`, batch size `16`
- **Training:** max 15 epochs, early stopping (patience = 3) on validation F1-Score

## Datasets

| Dataset | Task | Description |
|---|---|---|
| [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) | Network Intrusion Detection | Improved version of KDD'99, normal traffic vs. intrusions |
| [CIC-MalDroid 2020](https://www.unb.ca/cic/datasets/maldroid-2020.html) | Android Malware Detection | Benign apps vs. 4 malware families (Adware, Banking, SMS Malware, Riskware) |

| Dataset | Train | Test | Classes |
|---|---|---|---|
| NSL-KDD | 125,973 | 22,544 | Normal / Abnormal |
| CIC-MalDroid (Binary) | 8,118 | 3,480 | Benign / Malware |
| CIC-MalDroid (Multiclass) | 8,118 | 3,480 | Benign, Adware, Banking, SMS Malware, Riskware |

## Repository Structure

```
.
├── neural_network/                          # BERT model definitions & classification heads
├── preprocessing/                           # Data loading & storytelling scripts
├── columnmapping_KDD.json                   # Feature-name → natural language mapping (NSL-KDD)
├── columnmapping_CICMaldroid.json           # Feature-name → natural language mapping (CIC-MalDroid)
├── main.py                                  # Training/testing entry point — NSL-KDD (binary)
├── main_CICMaldroid.py                      # Training/testing entry point — CIC-MalDroid (binary)
├── main_CICMaldroidMulticlass.py            # Training/testing entry point — CIC-MalDroid (multiclass)
├── calculate_integrated_gradients_binary.py     # IG attribution scores — binary models
├── calculate_integrated_gradients_multiclass.py # IG attribution scores — multiclass model
├── xai_local_globalBINARY.py                # Local & global XAI score aggregation — binary
└── xai_local_globalMULTICLASS.py            # Local & global XAI score aggregation — multiclass
```

## Results

### NSL-KDD (Binary)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Intrusions | 0.94 | 0.65 | 0.77 |

### CIC-MalDroid (Binary)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Malware | 0.95 | 0.99 | 0.97 |

### CIC-MalDroid (Multiclass)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Benign | 0.90 | 0.73 | 0.80 |
| Adware | 0.64 | 0.77 | 0.70 |
| Banking | 0.82 | 0.83 | 0.83 |
| SMS Malware | 0.94 | 0.96 | 0.95 |
| Riskware | 0.86 | 0.85 | 0.86 |
| **Macro Average** | **0.83** | **0.83** | **0.83** |

### Comparison with State of the Art (CIC-MalDroid, Macro F1-Score)

| Model | Macro F1-Score |
|---|---|
| VINCENT (ViT + Knowledge Distillation) | 0.841 |
| **This work (BERT-Medium + storytelling)** | **0.827** |
| Renoir (Autoencoder + Triplet Network) | 0.736 |
| Roulette (CNN + Attention) | 0.536 |

The proposed approach achieves results competitive with (and in some cases superior to) more complex, purpose-built architectures for cyber-threat detection, while leveraging a general-purpose language model.

## Explainability (XAI)

Model predictions are explained using **Integrated Gradients**, which attributes an importance score to each input token by comparing the actual input against a neutral baseline. This allows identification of which features drive the model's decisions — both at the **local** level (single prediction) and **global** level (average behavior per class).

Key findings:
- **NSL-KDD:** `src_bytes`, `flag`, and `dst_host_serror_rate` are the most influential features for detecting intrusions, consistent with known DoS attack mechanics.
- **CIC-MalDroid:** malicious apps are strongly characterized by system calls such as `newselect`, `mprotect`, and `getInstallerPackageName`, often associated with detection-evasion behavior.

## Getting Started

### Requirements

- Python 3.13.3+
- PyTorch
- Hugging Face Transformers
- Captum (for Integrated Gradients)
- scikit-learn, pandas, numpy

```bash
pip install torch transformers captum scikit-learn pandas numpy
```

### Usage

```bash
# NSL-KDD — binary intrusion detection
python main.py

# CIC-MalDroid — binary malware detection
python main_CICMaldroid.py

# CIC-MalDroid — multiclass malware family classification
python main_CICMaldroidMulticlass.py

# Explainability (Integrated Gradients)
python calculate_integrated_gradients_binary.py
python calculate_integrated_gradients_multiclass.py
python xai_local_globalBINARY.py
python xai_local_globalMULTICLASS.py
```

> Dataset files are not included in this repository. Download NSL-KDD and CIC-MalDroid 2020 from the links above and update the paths in the scripts accordingly.

## Tech Stack

- **Model:** BERT-Medium ([prajjwal1/bert-medium](https://huggingface.co/prajjwal1/bert-medium))
- **Framework:** PyTorch / Hugging Face Transformers
- **XAI:** Integrated Gradients ([Captum](https://captum.ai/))
- **Language:** Python

## Future Work

- Improve Recall on network intrusion detection (e.g., data augmentation, class-imbalance-aware loss functions)
- Improve detection performance on the Adware category
- Evaluate scalability with larger language models (BERT-Base/Large or other LLMs)
- Real-world evaluation on unseen Android applications and live network traffic

## Authors & Supervisors

**Author:** Lorenzo Levanto — Computer Science, Università degli Studi di Bari Aldo Moro

**Supervisors:**
- Prof.ssa Annalisa Appice
- Dr. Giuseppina Andresini
- Dr. Vincenzo Pasquadibisceglie
