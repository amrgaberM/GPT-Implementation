# GPT Implementation: Deep Dive & Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

**A custom GPT architecture implementation built from first principles, featuring memory-efficient training and deep interpretability analysis.**

> Read the full technical breakdown on Medium: [Building a 163M-Parameter GPT Model from Scratch](https://medium.com/@amrgabeerr20/building-a-163m-parameter-gpt-model-from-scratch-a-deep-dive-into-transformer-architecture-1e0976ccda10)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features & Optimizations](#key-features--optimizations)
- [Training Journey](#training-journey)
- [Interpretability & Analysis](#interpretability--analysis)
- [Getting Started](#getting-started)
- [Future Roadmap](#future-roadmap)

---

## Project Overview

This repository contains a manual implementation of the GPT-2 Transformer decoder, designed to explore the internal mechanics of Large Language Models without relying on high-level abstractions like Hugging Face's `Trainer`.

The project is optimized for a **single NVIDIA T4 GPU**, using advanced resource management techniques. It also emphasizes **mechanistic interpretability**, with diagnostic tools that visualize semantic learning and highlight failure points.

---

## Key Features & Optimizations

| Feature | Description |
| :--- | :--- |
| **Custom Causal Attention** | Manual Query-Key-Value attention with masked self-attention. |
| **Mixed Precision (AMP)** | `torch.cuda.amp` reduces VRAM usage by ~40% while preserving convergence. |
| **Gradient Accumulation** | Decouples batch size from GPU memory; enables effective batch sizes of 64+ on a single GPU. |
| **Deep Inspection Hooks** | Forward-pass hooks to extract and visualize attention scores without training overhead. |

---

## Training Journey

### Phase 1: Proof of Concept (Shakespeare)
- **Dataset:** Tiny Shakespeare  
- **Objective:** Architectural validation  
- **Result:** Loss 1.1 (Overfitting)  
- **Insight:** Model captures Old English grammar and vocabulary; attention masking and positional encodings validated.  

### Phase 2: Scale-Up (Wikipedia)
- **Dataset:** Enwik8 (Wikipedia English)  
- **Compute:** Single NVIDIA T4 (Google Colab)  
- **Constraint:** 1.5 hours training time  
- **Result:** Loss 3.4  
- **Challenge:** Model plateaued; developed **Analysis Suite** to diagnose internal states instead of blind tuning.  

---

## Interpretability & Analysis

A **Depth Scan** tool visualizes attention head entropy across layers:

### Layer 0: Healthy Syntax
- Diagonal and local attention patterns dominate.  
- **Observation:** Model learns local dependencies, e.g., adjectives modifying nearby nouns.  

### Layer 11: Semantic Collapse
- Final layers show vertical attention stripes ("Attention Sinks").  
- **Observation:** Model over-focuses on high-frequency or dominant tokens, reducing semantic specificity.  

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/gpt-from-scratch.git
cd gpt-from-scratch

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train.py
```

*(Default configuration uses Mixed Precision & Gradient Accumulation)*

### 4. Run Analysis

```bash
python analyze.py --checkpoint checkpoints/model_final.pt
```

Generates attention maps and performs a depth scan.

---

## Future Roadmap

* [ ] **RoPE Implementation:** Replace absolute positional encodings with Rotary Embeddings.
* [ ] **DDP Support:** Scale training across multiple GPUs using `DistributedDataParallel`.
* [ ] **KV Caching:** Optimize inference for longer sequence generation.

---

*Built from first principles using PyTorch.*




