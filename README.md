# High-Fidelity Face Swapping with Identity
Preservation and Background Consistency via
Conditional Diffusion

This repository contains the official implementation of the paper **"High-Fidelity Face Swapping with Identity
Preservation and Background Consistency via
Conditional Diffusion"**, which is currently submitted to *The Visual Computer*.

## Overview

We propose a high-fidelity face swapping framework based on conditional denoising diffusion probabilistic models. Our method disentangles source identity and target attributes through a cross-attention mechanism, introduces a ground-truth noise-based midpoint estimation strategy for more precise identity supervision, and employs an iterative face-background fusion module to improve visual coherence.

## Prerequisites

The code has been verified in the following environment:

- Python 3.7

## Installation

Clone the repository:

```bash
git clone https://github.com/Zty031220/guided_diffusion.git
cd guided_diffusion

Install dependencies:
pip install -r requirements.txt

## Training
To train the model, ensure your dataset is prepared, then execute:
bash train_image.sh

## Inference
To perform face swapping using the trained model, execute the sampling script:
bash sample_image_ddpm.sh
