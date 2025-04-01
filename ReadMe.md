# PaliGemma2 fine-tuning with PyTorch DDP

Hugging Face's tools for fine-tuning Vision-Language Models (VLMs), like `Trainer`, are easy to use but can be hard to understand in detail. This repository, inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), aims to make fine-tuning VLMs clearer. The main goal of this repo is to understand how VLM fine-tuning works behind the scenes.

## Repository Structure

* **`Fine_tune_PaliGemma.ipynb`**: A notebook for fine-tuning the PaliGemma model using huggingface `Trainer()`. Refer to [smol-vision](https://github.com/merveenoyan/smol-vision/blob/main/Fine_tune_PaliGemma.ipynb)
* **`PaliGemma_torch_ddp_finetune.py`**: A Python script that implements the fine-tuning logic from the above notebook using PyTorch Distributed Data Parallel (DDP).
* **`Inference.ipynb`**: A Jupyter notebook to perform inference using the fine-tuned model.

### Fine-tuning with DDP

*  **Configuration:** Key training parameters are defined within the script in the `TrainingConfig` dataclass. This includes batch size, learning rate, LoRA settings, and DDP configurations. Adjust these as needed.
*  **Running the Script:** Use `torchrun` to launch the DDP training. Adjust `--nproc_per_node` based on the number of GPUs you want to use.
    ```bash
    torchrun --standalone --nproc_per_node=4 PaliGemma_torch_ddp_finetune.py
    ```
*  **Output:** LoRA adapter weights will be saved periodically to the directory specified in `TrainingConfig` (default: `paligemma_vqav2_custom/`).

### Inference

Refer to the `Inference.ipynb` notebook for loading the fine-tuned adapter weights and running the model for inference.
