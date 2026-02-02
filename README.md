Our study focuses on domain shift for different geographical location from opposing training data for popular segmentation models: U-Net, Prithvi and U-Prithvi. The experiment suggests that the choice of loss function largely determine how well the model performs on Out-Of Distribution (OOD) inference for real world deployment in flood segmentation, which is a modern concern regarding climate change globally.

Primarily, real world deployment on this context spans around different satellite sensors and varying resolution, this codebase provides the hands-on methodology to reproduce the findings and help geoinformatics researchers/engineers for their application.

## Setup

```
# Setup python 3.13 environment
chmod +x scripts/setup_py.sh
./scripts/setup_py.sh

# Initialize environment
source .venv/bin/activate
uv pip install -r requirements.txt
```

```
# Download sen1floods11 and Timor Leste test set
chmod +x datasets/installds.sh
./datasets/installds.sh
```

```
# Clone Prithvi from HF
git clone https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M ./prithvi
```

## Training

The training script (`train.py`) implements a comprehensive comparison framework for three segmentation models: U-Net, Prithvi, and Prithvi-UNet (U-Prithvi). It includes support for multiple loss functions, two-phase training for Prithvi-based models, and automatic evaluation on both test and Bolivia datasets.

### Basic Training Command

```bash
python ./exp/train.py \
    --data_path ./datasets/sen1floods11_v1.1 \
    --epochs 100 \
    --batch_size 12 \
    --learning_rate 5e-4 \
    --loss_func focal
```

### Training Arguments

**Data and Experiment Configuration:**
- `--data_path`: Path to the sen1floods11 dataset directory (default: `./datasets/sen1floods11_v1.1`)
- `--version`: Experiment version identifier (default: `comparison`)
- `--batch_size`: Batch size for training (default: `12`)
- `--epochs`: Number of training epochs (default: `5`)

**Optimization:**
- `--learning_rate`: Initial learning rate for AdamW optimizer (default: `5e-4`)
- `--loss_func`: Loss function selection (default: `bce`)
  - `bce`: Binary Cross-Entropy with class weights [0.7, 0.3]
  - `diceloss`: Dice Loss implementation
  - `dl2`: Dice Loss variant 2
  - `focal`: Focal Loss (α=0.25, γ=2)
  - `lovasz`: Lovasz-Softmax Loss
  - `tversky`: Tversky Loss (α=0.3, β=0.7, γ=1.33)

**Model Architecture:**
- `--prithvi_out_channels`: Output channels from Prithvi encoders (default: `768`)
- `--unet_out_channels`: Output channels from UNet encoders (default: `768`)
- `--combine_func`: Feature combination method for U-Prithvi (default: `concat`)
  - `concat`: Concatenate features
  - `mul`: Element-wise multiplication
  - `add`: Element-wise addition
- `--random_dropout_prob`: Dropout probability for U-Prithvi (default: `0.667`)

**Prithvi-Specific Training:**
- `--prithvi_finetune_ratio`: Fine-tuning epochs ratio (default: `1`)
  - Set to `1` for equal initial and fine-tuning epochs
  - Set to `0` to skip fine-tuning phase
  - Initial phase: Prithvi encoder frozen, only segmentation head trains
  - Fine-tuning phase: All weights trainable with 10x reduced learning rate

**Checkpointing:**
- `--save_model_interval`: Save model checkpoint every N epochs (default: `5`)
- `--test_interval`: Evaluate on validation set every N epochs (default: `1`)

### Training Examples

**Quick experiment (5 epochs):**
```bash
python train.py --epochs 5 --loss_func bce
```

**Full training with focal loss (100 epochs):**
```bash
python ./exp/train.py \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --loss_func focal \
    --version focal_100e
```

**Training with Lovasz loss and custom architecture:**
```bash
python ./exp/train.py \
    --epochs 100 \
    --loss_func lovasz \
    --prithvi_out_channels 512 \
    --unet_out_channels 512 \
    --combine_func add \
    --version lovasz_512ch
```

**Training with extended fine-tuning:**
```bash
python ./exp/train.py \
    --epochs 50 \
    --prithvi_finetune_ratio 2 \
    --loss_func focal \
    --version extended_finetune
```
This runs 50 initial epochs + 100 fine-tuning epochs (50 × 2).

### Training Output

The training script automatically:
- Creates log directories: `./logs/bolivia_{EPOCHS}E_{LOSS_FUNC}/`
- Saves TensorBoard logs for each model (train/validation metrics)
- Saves model checkpoints at specified intervals
- Saves final model weights as `model_final.pt`
- Generates `comparison_results.json` with comprehensive metrics

**Directory Structure:**
```
logs/bolivia_100E_FOCAL/
├── unet/
│   ├── models/
│   │   ├── model_epoch_5.pt
│   │   ├── model_epoch_10.pt
│   │   └── model_final.pt
│   └── events.out.tfevents.*  # TensorBoard logs
├── prithvi_sen1floods/
│   └── ...
├── prithvi_unet/
│   └── ...
└── comparison_results.json
```

### Monitoring Training

View training progress with TensorBoard:
```bash
tensorboard --logdir ./logs/bolivia_100E_FOCAL
```

### Training Phases for Prithvi Models

1. **Phase 1 - Frozen Encoder** (epochs 0 to `--epochs`)
   - Prithvi encoder weights frozen
   - Only segmentation head and decoder train
   - Full learning rate

2. **Phase 2 - Fine-tuning** (epochs `--epochs` to `--epochs + --epochs × --prithvi_finetune_ratio`)
   - All Prithvi weights unfrozen
   - End-to-end training
   - Learning rate reduced by 10x
   - PolynomialLR scheduler applied

## Inference

The repository provides two inference scripts for different test scenarios:

### 1. Bolivia Dataset Inference (`infer_bolivia.py`)

Performs inference on the sen1floods11 Bolivia test set and generates comparative visualizations between all three models.

**Basic Usage:**
```bash
python ./exp/infer/infer_bolivia.py \
    --data_path ./datasets/sen1floods11_v1.1 \
    --split bolivia \
    --output_path ./outputs/bolivia_results \
    --model_type prithvi_unet
```

**Arguments:**
- `--data_path`: Path to sen1floods11 dataset (default: `./datasets/sen1floods11_v1.1`)
- `--split`: Dataset split to use (choices: `train`, `valid`, `test`, `bolivia`; default: `test`)
- `--output_path`: Directory to save visualization outputs
- `--bands`: Sentinel-2 bands to use (default: `[1,2,3,8,11,12]` = B,G,R,NIR,SWIR1,SWIR2)
- `--model_type`: Model for single inference (choices: `unet`, `prithvi`, `prithvi_unet`)
- `--weights_path`: Path to Prithvi foundation weights (default: `./prithvi/Prithvi_100M.pt`)

**Model Paths Configuration:**
Edit the `models_paths` dictionary in `infer_bolivia.py`:
```python
models_paths = {
    'unet': 'logs/bolivia_100E_FOCAL/unet/models/model_final.pt',
    'prithvi': 'logs/bolivia_100E_FOCAL/prithvi_sen1floods/models/model_final.pt',
    'prithvi_unet': 'logs/bolivia_100E_FOCAL/prithvi_unet/models/model_final.pt',
}
```

**Inference Process:**
- Uses sliding window approach (224×224 window, 128×128 stride)
- Processes all three models simultaneously
- Generates comparative visualizations showing:
  - RGB input image with band information
  - Prithvi-UNet vs UNet comparison
  - Prithvi-UNet vs Prithvi comparison
  - Prithvi vs UNet comparison
- Color-coded predictions:
  - **White**: Both models correct
  - **Gray**: No data available
  - **Blue**: Both models incorrect
  - **Red**: First model correct only
  - **Green**: Second model correct only

**Outputs:**
- Visualization images: `{filename}.png` for each test sample
- Console output: Cumulative metrics (mIoU, accuracy) for all models

**Example Command:**
```bash
# Inference on Bolivia test set
python ./exp/infer/infer_bolivia.py \
    --split bolivia \
    --output_path ./outputs/bolivia_focal_100e

# Inference on regular test set
python ./exp/infer/infer_bolivia.py \
    --split test \
    --output_path ./outputs/test_results
```

### 2. Timor-Leste Dataset Inference (`infer_timor.py`)

Performs cross-resolution domain shift inference on the Timor-Leste ML4Floods dataset (Sentinel-1 + Sentinel-2 at 10m resolution).

**Basic Usage:**
```bash
python ./exp/infer/infer_timor.py \
    --data_path ./datasets/Timor_ML4Floods \
    --output_path ./outputs/timor_results \
    --aoi all
```

**Arguments:**
- `--data_path`: Path to Timor-Leste dataset (default: `./datasets/Timor_ML4Floods`)
- `--output_path`: Directory to save results (default: `./outputs/timor_results`)
- `--bands`: Sentinel-2 bands to use (default: `[1,2,3,8,11,12]`)
- `--model_type`: Model for inference (choices: `unet`, `prithvi`, `prithvi_unet`)
- `--weights_path`: Path to Prithvi weights (default: `./prithvi/Prithvi_100M.pt`)
- `--aoi`: Area of Interest to process (choices: `all`, `01`, `02`, `03`, `05`, `07`; default: `all`)

**Model Paths Configuration:**
Edit the `models_paths` dictionary in `infer_timor.py`:
```python
models_paths = {
    'unet': 'logs/bolivia_100E_DL2/unet/models/model_final.pt',
    'prithvi': 'logs/bolivia_100E_DL2/prithvi_sen1floods/models/model_final.pt',
    'prithvi_unet': 'logs/bolivia_100E_DL2/prithvi_unet/models/model_final.pt',
}
```

**Ground Truth Format:**
Timor-Leste ground truth uses 2-band format:
- **Band 0** (Cloud/Validity Mask):
  - `0`: Ignore (no-data)
  - `1`: Valid clear land (evaluated)
  - `2`: Cloud (visualized but not evaluated)
- **Band 1** (Flood Annotation):
  - `0`: Ignore (no-data)
  - `1`: Non-Flood
  - `2`: Flood

**Inference Process:**
- Processes all AOIs (Areas of Interest) or specific AOI
- Uses sliding window inference (224×224 window, 128×128 stride)
- Only evaluates on clear, valid pixels (Band0==1 & Band1!=0)
- Tracks cloud-flood predictions separately
- Generates comparative visualizations with legends

**Outputs:**
1. **Visualization Images**: `AOI{num}_{filename}_comparison.png` for each tile
2. **Metrics JSON**: `metrics_results.json` containing:
   - Per-model confusion matrices
   - mIoU and accuracy metrics
   - Pixel count statistics (flood, non-flood, cloud-flood)

**Example Commands:**
```bash
# Process all AOIs
python ./exp/infer/infer_timor.py --aoi all

# Process specific AOI
python ./exp/infer/infer_timor.py --aoi 03 --output_path ./outputs/timor_aoi03

# Process with custom model path
python ./exp/infer/infer_timor.py \
    --aoi all \
    --output_path ./outputs/timor_custom \
    --weights_path ./prithvi/Prithvi_EO_V1_100M.pt
```

### Understanding Inference Outputs

**Confusion Matrix Metrics:**
- **TP** (True Positives): Flood pixels correctly identified
- **FP** (False Positives): Non-flood pixels incorrectly predicted as flood
- **TN** (True Negatives): Non-flood pixels correctly identified
- **FN** (False Negatives): Flood pixels incorrectly predicted as non-flood

**Derived Metrics:**
- **IoU_floods**: TP / (TP + FN + FP)
- **IoU_non_floods**: TN / (TN + FP + FN)
- **Avg_IOU**: Mean of flood and non-flood IoU (primary metric)
- **ACC_floods**: TP / (TP + FN)
- **ACC_non_floods**: TN / (TN + FP)
- **Avg_ACC**: Mean of flood and non-flood accuracy

### Normalization Details

Both inference scripts use sen1floods11 normalization statistics (trained on 30m resolution):
```python
MEANS = [0.1425, 0.1392, 0.1243, 0.3142, 0.2074, 0.1205]
STDS = [0.0404, 0.0419, 0.0527, 0.0822, 0.0683, 0.0529]
```

For Timor-Leste (10m resolution), the script includes computed statistics as comments for reference, but uses sen1floods11 normalization to test domain adaptation.

# Acknowledgement
 
Large part of the source code is borrowed from [Vit Kostejn](https://github.com/kostejnv/prithvi_segmentation.git)

# Citations

```bibtex
Vit Kostejn, Yamil Essus, Jenna Abrahamson, and Ranga Raju Vatsavai. U-Prithvi: Integrating a Foundation Model and U-Net for Enhanced Flood Inundation Mapping. In 13th International Conference on Geographic Information Science (GIScience 2025). Leibniz International Proceedings in Informatics (LIPIcs), Volume 346, pp. 18:1-18:17, Schloss Dagstuhl – Leibniz-Zentrum für Informatik (2025) https://doi.org/10.4230/LIPIcs.GIScience.2025.18
```