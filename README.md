
# Project Overview

This project is a modified version based on the [RWKV-LM-LoRA](https://github.com/Blealtan/RWKV-LM-LoRA/) project and is used for training on ChatGal data. Since this project incorporates some of the latest community solutions, it can better achieve the long-context capability of training RWKV, making it suitable for other projects as well.

Main Features:

- Automatic saving of the latest model parameters when exiting with Ctrl+C.
- Avoiding VRAM bottlenecks caused by CUDA kernels, supporting full training of RWKV 7B models on 24GB VRAM GPUs, and LoRA training of RWKV 7B models.
- Experimental training with infinite context.

The main code of this project is the same as [RWKV-LM-LoRA](https://github.com/Blealtan/RWKV-LM-LoRA/), and you can refer to its training configuration. **A reminder: To train LoRA models, you need to merge them with the main model first (using merge_lora.py or merge_lora_plus.py).**

Due to the accumulation of a large amount of code in this project, the code may be organized and refactored in the future. Here, I will briefly introduce the solution for handling long contexts.

During the training process of the RWKV model, the training text length `ctx_len` has a significant impact on VRAM usage. It not only increases the storage occupied by activation values but also increases the temporary storage overhead during the WKV kernel computation, making it difficult to train texts of length 4096 or above for a 7B model on a 24GB VRAM GPU (even when layer-wise gradient checkpointing `--grad_cp 1` is enabled).

## Solution 1: Reducing VRAM Overhead of WKV

This project uses a modified WKV kernel that supports state input. One solution is to divide the input sequence into smaller sequence lengths during WKV computation, perform WKV computations sequentially, and maintain the normal propagation of states.

You can adjust the `--ctx_parts` parameter, starting from 1 and gradually increasing it. For example, when `--ctx_len 4096 --ctx_parts 4`, the model will actually use a WKV kernel with a length of 1024, avoiding OOM errors caused by WKV computations.

This solution can slightly reduce VRAM overhead and extend the maximum trainable length, but it cannot solve the VRAM overhead caused by activation values and the computation of logits in the final layer. Therefore, the improvement is limited. However, the advantage of this solution is that it is compatible with DeepSpeed Stage 2 and offload, and it can reduce VRAM without side effects compared to the original RWKV-LM.

## Solution 2: Training with Infinite Context (State Gradient Checkpointing)

[infctx](https://github.com/Blealtan/RWKV-LM-LoRA/tree/dev-infctx) is an RWKV training program implemented by Blealtan and others within the RWKV community. It allows training with significantly longer sequences. The principle is to utilize the RNN-like characteristics of RWKV and perform gradient checkpointing in the time dimension, trading off VRAM for reduced memory consumption and supporting training with long sequences.

To enable this feature, use the `--state_cp 1` parameter. To gradually adapt the model to long contexts, it is recommended to enable `--initial_ctx_len 4096 --ctx_warmup_steps 200`, gradually increasing the training length from 4096 to `ctx_len` over 200 steps.

Using LoRA training with a single 24GB GPU, it is possible to train text lengths of 128K or above (with significant VRAM remaining).

Due to some issues with the DeepSpeed framework, this solution is only applicable to DeepSpeed Stage 1. For a single 24GB VRAM GPU, this solution can only fine-tune the 7B model using LoRA. To fully fine-tune the 7B model, at least 4x80GB GPUs are required.

# Miscellaneous

## Steps for Full Fine-tuning

- Preprocessing: Place the preprocessed neox-style JSONL files in the `datasource` folder. Refer to the command in `preprocess_data.sh`. A preprocessed version (without R18 content) has already been provided in the `data` folder.
- Fine-tuning: Refer to the command in `train.sh`. The training should be conducted within the `RWKV-v4neo-LoRA` folder.
- Inference: Use ChatRWKV or the text-generation-webui to load the trained model.

## Random Notes

- Put the model in the `pretrained_models` folder or modify it yourself according to `train.sh`.
- `megatron` and `preprocess_data.py` are from https://github.com/EleutherAI/gpt-neox.
- The command to run `preprocess_data.sh` is from https://github.com/BlinkDL/RWKV-LM#training--fine-tuning with some modifications.
- `RWKV-v4neo` is from https://github.com/BlinkDL/RWKV-LM.
- The training command reference is from https://www.bilibili.com/read/cv22445881 with some modifications.
- Inference currently uses `temp = 0.7, top_p = 1`.
- Explanation of parameters in `train.sh`:
  - load_model: Specify the path of the pre-trained model to be used as the initial model for training.
  - wandb: Set the project name for Weights & Biases, a platform for tracking deep learning experiments.
  - data_file: Specify the path of the training data file.
  - data_type: Set the type of training data.
  - vocab_size: Set the size of the vocabulary.
  - ctx_len: Set the length of the context.
  - accumulate_grad_batches: Set the number of gradient accumulation batches for gradient accumulation optimization.
  - epoch_steps: Set the number of training steps per epoch.
  - epoch_count: Set the total number of training epochs.
  - epoch_save: Set how often to save the model, in terms of epochs.
  - micro_bsz: Set the size of the micro-batch.
  - n_layer: Set the number of layers in the model.
  - n_embd: Set the embedding dimension of the model.
  - pre_ffn: Set the preprocessing of the feed-forward neural network (FFN) (0 means not using).
  - head_qk: Set the query and key for attention heads (0 means not using).
  - lr_init: Set the initial learning rate.
  - lr_final: Set the final learning rate.
  - warmup_steps: Set the number of warm-up steps for learning rate.
  - beta1: Set the beta1 parameter of the Adam optimizer.
  - beta2: Set the beta2 parameter of the Adam optimizer.
  - adam_eps: Set the epsilon parameter of the Adam optimizer.
  - accelerator: Set the accelerator type (e.g., GPU).
  - devices: Set the number of devices to use.
  - precision: Set the computation precision (e.g., bfloat16).
  - strategy: Set the training strategy (e.g., deepspeed_stage_2_offload).
  - grad_cp: Layer gradient checkpoint flag.
  - state_cp: State gradient checkpoint flag. When enabled, it significantly reduces the VRAM usage for long contexts.
  - initial_ctx_len: Initial training context length. Used in conjunction with ctx_warmup_steps.
  - ctx_warmup_steps: Context warm-up steps. The context length gradually increases from initial_ctx_len to ctx_len during training.
  - ctx_parts: Number of context slices. WKV operators of length ctx_len/ctx_parts will be used during training. For example, if ctx_len=8192 and ctx_parts=8, WKV operators of length 1024 will be used. Shorter WKV operators consume less VRAM but may require more computation time.

 # More trainning details: https://wandb.ai/one-/projects
