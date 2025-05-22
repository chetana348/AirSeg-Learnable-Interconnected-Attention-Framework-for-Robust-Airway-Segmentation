# AirSeg: Learnable Interconnected Attention Framework for Robust Airway Segmentation

This is the **official code repository** for the paper:  
📄 **AirSeg: Learnable Interconnected Attention Framework for Robust Airway Segmentation**  
Published in *Journal of Digital Imaging, 2025*  
🔗 [Read the paper here](https://link.springer.com/article/10.1007/s10278-025-01545-z)

---

## 🧠 Overview

AirSeg proposes a novel attention-driven segmentation framework tailored for airway extraction from chest CT scans. It is designed to overcome challenges like class imbalance, anatomical complexity, and structural discontinuity.

The model introduces **interconnected attention modules** that dynamically fuse features across scales and stages, enhancing both global and local contextual understanding.

---

## ✨ Features

- ✅ **Interconnected Attention Blocks**: Fusion of spatial and channel-wise attention across encoder-decoder stages.
- 🔁 **Learnable Variance-Based Embedding**: Enhances robustness to noise and spatial inconsistencies.
- 🌲 **Improved Peripheral Branch Detection**: Captures thin airway structures with higher fidelity.
- 📉 **Statistically Significant Performance**: Demonstrated superior Dice scores across both *in vivo* and *in situ* datasets.
- 💡 **Modular & Extensible**: Easy to plug into existing 2D or 3D segmentation pipelines.

---

> **Note:** If you are looking for the lightweight encoder-decoder adaptation of this work, please see this repository:  
> 👉 [Lung-and-Airway-Segmentation-using-Modified-UNet](https://github.com/chetana348/Lung-and-Airway-Segmentation-using-Modified-UNet)

--- 
## 📄 Requirements

The codebase was developed and tested with the following dependencies:

- Python 3.9.21
- PyTorch 2.5.0 (with CUDA 11.8)
- NumPy
- TensorBoard
- scikit-learn
- torchvision
- tqdm
- matplotlib
- opencv-python
- albumentations
- SimpleITK
- nibabel

---

## 📁 Repository Structure

- `models/`  
  - `SCA.py` – Main AirSeg architecture with Spatial-Channel Attention  
  - `resunet.py` – ResNet-based encoder backbone  

- `scripts/`  
  - `dataloader.py` – Custom dataset and augmentation logic  
  - `losses.py` – Loss functions (Dice, Focal, etc.)  
  - `metrics.py` – Evaluation metrics
 
---

## 📊 Datasets

This study utilizes the following datasets:

- **UAB Airway Dataset** *(Proprietary)*  
  A private dataset of thoracic CT scans annotated for airway segmentation, sourced from the University of Alabama at Birmingham. Due to institutional restrictions, this dataset is **not publicly available**.

- **[ProstateX]([https://www.aapm.org/GrandChallenge/PROSTATEX/](https://www.cancerimagingarchive.net/collection/prostatex/))**  
  A public prostate MRI dataset originally designed for lesion classification, adapted here to evaluate generalization of AirSeg to non-thoracic anatomical structures.

> **Note:** Preprocessing scripts for both datasets are included under `scripts/dataloader.py`. Custom data loaders handle patch extraction, normalization, and mask alignment.

---

## 🏋️‍♀️ How to Train

AirSeg is compatible with any standard UNet-like training loop.

You can use your existing training scripts with the following setup:

- Use `models/SCA.py` as the main segmentation model.
- Use `models/resunet.py` as a backbone encoder.
- Use loss functions and metrics from `scripts/losses.py` and `scripts/metrics.py`.

### Example (PyTorch)

```python
from models.resunet import *
model = ResUNet(in_channels=1, out_channels=3)  # or adjust channels as needed

> **Note:** The architecture is modular and plug-and-play compatible with most 2D medical image segmentation pipelines.
> **Note** AirSeg can be used with any UNet like 2D encoders/decoders.
---

## 📦 Model Weights

Pretrained model weights can be provided upon request.  
Please contact us via email or open an issue in this repository.
---

## 📚 Citation

If you use this work in your research, **please cite us**:

```bibtex
@article{krishnan2025airseg,
  title     = {AirSeg: Learnable Interconnected Attention Framework for Robust Airway Segmentation},
  author    = {Krishnan, C. and Hussain, S. and Stanford, D. and others},
  journal   = {Journal of Digital Imaging},
  year      = {2025},
  publisher = {Springer},
  doi       = {10.1007/s10278-025-01545-z},
  note      = {Inform. med.}
}

---
## 🛡 License

This project is licensed under the **MIT License**.  

