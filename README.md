# AIMO Math Finetuning using DeepSeek-R1-Distill-Qwen-7B

## Description

This project focuses on fine-tuning the `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` model for mathematical problem-solving, specifically targeting problems similar to those found in the [AI Mathematical Olympiad (AIMO) competition on Kaggle](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2).

The fine-tuning process utilizes the `Floppanacci/QWQ-LongCOT-AIMO` dataset. The resulting fine-tuned model is available as [`Floppanacci/DeepSeek-R1-Distill-Qwen-7B-Floppanacci`](https://huggingface.co/Floppanacci/DeepSeek-R1-Distill-Qwen-7B-Floppanacci), and a 4-bit AWQ quantized version is available at [`Floppanacci/DeepSeek-R1-Distill-Qwen-7B-Floppanacci-AWQ`](https://huggingface.co/Floppanacci/DeepSeek-R1-Distill-Qwen-7B-Floppanacci-AWQ).

## Installation

Both notebooks were executed on Google Colab, using L4 gpu for the fine-tuning and A100 for the quantization (VRAM requirements were too high for the L4). Both notebooks would require modifications to run on a different hardware. Notably the dtype *torch.bfloat16* is only available on high-end Nvidia GPUs (e.g. L4, A100, H100) but not on other hardware such as T4 GPUs, which would require *torch.float16*.

## Data

The fine-tuning process uses the [`Floppanacci/QWQ-LongCOT-AIMO`](https://huggingface.co/datasets/Floppanacci/QWQ-LongCOT-AIMO) dataset available on Hugging Face.
*   **Size:** 36.8k samples (29.5k train, 3.68k validation, 3.68k test).
*   **Format:** CSV/Parquet, containing mathematical questions and detailed step-by-step answers (Chain-of-Thought style).
*   The dataset can be loaded using the `datasets` library.

## Fine-tuning

The `supervised-fine-tuning.ipynb` notebook contains the code for Supervised Fine-tuning (SFT).

1.  **Configure Training:** Modify the `TrainingArguments` within the notebook. Key parameters to adjust include:
    *   `output_dir`: Directory to save checkpoints and the final model.
    *   `num_train_epochs`: Number of training epochs (e.g., 3).
    *   `per_device_train_batch_size`: Batch size per GPU (e.g., 4).
    *   `gradient_accumulation_steps`: Steps for gradient accumulation (e.g., 8).
    *   `learning_rate`: Learning rate (e.g., 3e-5).
    *   Ensure you configure `report_to="wandb"` (or other tracker) and set an appropriate project name for monitoring.
2.  **Run Fine-tuning:** Execute the cells in the notebook. The script loads the base model (`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`), prepares the dataset, and runs the training using the `SFTTrainer` from the TRL library.

## Models

*   **Base Model:** [`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
*   **Fine-tuned Model:** [`Floppanacci/DeepSeek-R1-Distill-Qwen-7B-Floppanacci`](https://huggingface.co/Floppanacci/DeepSeek-R1-Distill-Qwen-7B-Floppanacci)
*   **Quantized Fine-tuned Model (4-bit AWQ):** [`Floppanacci/DeepSeek-R1-Distill-Qwen-7B-Floppanacci-AWQ`](https://huggingface.co/Floppanacci/DeepSeek-R1-Distill-Qwen-7B-Floppanacci-AWQ)


