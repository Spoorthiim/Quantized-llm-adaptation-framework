# Quantized-llm-adaptation-framework

An advanced parameter-efficient fine-tuning system for Llama 3.2 3B using quantized LoRA adapters and accelerated training via Unsloth framework.

## Features
- **Quantized Training Pipeline**: 4-bit precision optimization for memory efficiency
- **Low-Rank Adaptation (LoRA)**: Parameter-efficient fine-tuning with rank-16 adapters
- **Accelerated Training**: 2x faster convergence using Unsloth optimization framework
- **Multi-Head Attention Targeting**: Selective adaptation of query, key, value, and MLP projections
- **Dynamic Memory Management**: Gradient accumulation with optimized batch processing
- **Template-based Conversation Formatting**: Automated chat template integration
- **Mixed Precision Training**: FP16/BF16 support for enhanced computational efficiency

## Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Minimum 16GB GPU memory

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llama-finetuning.git
cd llama-finetuning

