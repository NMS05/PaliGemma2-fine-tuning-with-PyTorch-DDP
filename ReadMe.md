# PaliGemma fine-tuning with PyTorch DDP

## Motivation

Hugging Face's tools for fine-tuning Vision-Language Models (VLMs), like `Trainer`, are easy to use but can be hard to understand in detail.

This repository, inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), aims to make fine-tuning VLMs clearer. It uses PyTorch Distributed Data Parallel (DDP) to show how the fine-tuning process works.

The main goal is to help people learn how VLM fine-tuning works behind the scenes.

## Repository Structure

* **`Fine_tune_PaliGemma.ipynb`**: A notebook for fine-tuning the PaliGemma model using huggingface `Trainer()`.
* **`PaliGemma_torch_ddp_finetune.py`**: A Python script that implements the fine-tuning logic from the notebook using PyTorch DDP. This allows for multi-GPU training.
* **`Inference.ipynb`**: A Jupyter notebook to perform inference using a fine-tuned model.

### Fine-tuning with DDP

*  **Configuration:** Key training parameters are defined within the script in the `TrainingConfig` dataclass. This includes batch size, learning rate, LoRA settings, and DDP configurations. Adjust these as needed.
*  **Running the Script:** Use `torchrun` to launch the DDP training. Adjust `--nproc_per_node` based on the number of GPUs you want to use.

    ```bash
    --nproc_per_node=4 PaliGemma_torch_ddp_finetune.py
    ```
*  **Output:** LoRA adapter weights will be saved periodically to the directory specified in `TrainingConfig` (default: `paligemma_vqav2_custom/`).

### Inference

Refer to the `Inference.ipynb` notebook for loading the fine-tuned adapter weights and running the model for inference.