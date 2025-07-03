# ZonUI-3B Training Instruction (reproduce)
## 🔧Install Environment

```
conda create -n zonui python=3.10
conda activate zonui
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 --user
pip install -r requirements.txt --user
```

## 📦Setup Datasets
### Grounding datasets
- Download grounding training dataset -- [UGround-V1-8k](https://huggingface.co/datasets/zonghanHZH/UGround-V1-8k), [ShowUI-Web-8k](https://huggingface.co/datasets/zonghanHZH/ShowUI-web-8k), [ShowUI-Desktop](https://huggingface.co/datasets/showlab/ShowUI-desktop) and [AMEX-8k](https://huggingface.co/datasets/zonghanHZH/AMEX-8k)
- Download grounding evaluation dataset -- [ScreenSpot](https://huggingface.co/datasets/KevinQHLin/ScreenSpot) and [ScreenSpotv2](https://huggingface.co/datasets/zonghanHZH/ScreenSpot-v2)

You can use git clone to download these datasets easily:
```bash
# Navigate to datasets directory and download
export DATA_DIR="./datasets"          # Dataset directory
cd $DATA_DIR

git clone https://huggingface.co/datasets/showlab/ShowUI-desktop
git clone https://huggingface.co/datasets/zonghanHZH/UGround-V1-8k
git clone https://huggingface.co/datasets/zonghanHZH/ShowUI-web-8k
git clone https://huggingface.co/datasets/zonghanHZH/AMEX-8k
huggingface-cli download KevinQHLin/ScreenSpot --repo-type dataset --local-dir ScreenSpot
git clone https://huggingface.co/datasets/zonghanHZH/ScreenSpot-v2

# Return to project directory
cd ..
```

### Coordinate Transformation (Required for ShowUI-Desktop)
After downloading ShowUI-Desktop, you need to transform the 0-1 coordinates to original coordinates:
```bash
prepare/trans_coord_2_ori.ipynb
```

### Generate Sample Training Data
To create a small sample dataset for observing training data during training:
```bash
# Execute the sample data generation notebook
datasets/Training-data/sample.ipynb
```

This will generate `image` & `metadata` directories containing sample examples from the training datasets. This sample data is useful for:
- Monitoring training data quality
- Quick testing and debugging

## 🚀 Start Grounding Training

ZonUI-3B training follows a **two-stage training approach** to achieve optimal performance:

### 🎯 Training Overview
- **Stage 1**: Initial grounding training with cross-platform data
- **Stage 2**: Fine-tuned training with multi-resolution data and previous stage weights
- **Model Saving**: Checkpoint conversion and merging after each stage

Our codebase uses [Wandb](https://wandb.ai/) to monitor the training process. Please provide your own Wandb API key in the training scripts.

### 📋 Method

#### 1. Stage 1 Training
```bash
./stage1.sh
```

After Stage 1 completes, save the model checkpoints (see [Save Model Checkpoints](#save-model-checkpoints)).

#### 2. Stage 2 Training
Update the `--local_weight_dir` in `stage2.sh` to point to your Stage 1 merged model directory:
```bash
# Edit stage2.sh and update the path
--local_weight_dir=./logs/debug/2025-07-02_20-54-16/ckpt_model/merged_model
```

Then run Stage 2:
```bash
./stage2.sh
```

### 📊 Training Configuration

#### Stage 1 Key Parameters:
- **Epochs**: 12
- **Steps per epoch**: 122
- **Batch size**: 1
- **Grad accumulation**: 48
- **Learning rate**: 0.0001
- **LoRA rank**: 8

#### Stage 2 Key Parameters:
- **Epochs**: 12
- **Steps per epoch**: 122
- **Batch size**: 1
- **Grad accumulation**: 48
- **Learning rate**: 0.00005 (reduced)
- **LoRA rank**: 8

### 📈 Monitoring and Evaluation

- **Training Monitoring**: Monitor training progress through your Wandb dashboard
- **Evaluation Scripts**: We provide evaluation scripts for ScreenSpot benchmark:
  - `main/eval_screenspot.py` - Main evaluation script
  - `eval_ss.sh` - Evaluation script for ScreenSpot dataset (evaluation-only mode)
  - `eval_ss2.sh` - Evaluation script for ScreenSpot-v2 dataset (evaluation-only mode)
  - `ScreenSpot-Pro` - check out this repo: [screenspot-pro](https://github.com/Han1018/ScreenSpot-Pro-GUI-Grounding)
- **Custom Evaluation**: For custom evaluation datasets, create your evaluation function in `main/eval_X.py`

#### Running Evaluation Only
To run evaluation without training, use the provided evaluation scripts:
```bash
# Make sure to set LoRA rank to 0 for evaluation
chmod +x eval_ss.sh
./eval_ss.sh
```

**Important Notes:**
- Keep batch size as 1; increase `grad_accumulation_steps` for larger effective batch sizes
- **For evaluation only**: Use `--eval_only` flag and set `--lora_r=0` to disable LoRA modifications
- The evaluation scripts (`eval_ss.sh`, `eval_ss2.sh`) are pre-configured for evaluation-only mode
- Make sure to save checkpoints after Stage 1 before proceeding to Stage 2

## ⬇️Save Model Checkpoints

After each training stage completes, you need to convert and merge the model checkpoints for the next stage or final use.

### Automatic Checkpoint Conversion

The training script automatically saves checkpoints in DeepSpeed format. To convert and merge these checkpoints:

```bash
# Set your experiment directory (replace with your actual path)
exp_dir="./logs/debug/2025-07-02_20-54-16/"  # or your actual experiment director
zonui_dir=$(pwd)
ckpt_dir="${exp_dir}/ckpt_model/"
merge_dir="${ckpt_dir}/merged_model"

# Navigate to checkpoint directory
cd "$ckpt_dir" || { echo "Failed to cd to $ckpt_dir"; exit 1; }

# Convert DeepSpeed checkpoint to standard PyTorch format
python zero_to_fp32.py . pytorch_model.bin

# Create merged model directory
mkdir -p merged_model

# Return to main directory for further processing if needed
cd "$zonui_dir"
python3 merge_weight.py --exp_dir="$exp_dir"

echo "Merged model saved at: $merge_dir"
```

### Using the Converted Model

After Stage 1, update your `stage2.sh` script to point to the merged model:
```bash
--local_weight_dir=./logs/debug/2025-07-02_20-54-16/ckpt_model/merged_model
```

### Important Notes

- The `zero_to_fp32.py` script is automatically generated by DeepSpeed during training
- Make sure to use the correct experiment directory path with timestamp
- The merged model will be ready for Stage 2 training or final inference
- Keep the original checkpoints as backup until you confirm the merged model works correctly

