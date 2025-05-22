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

## 🛠 Installation

```bash
git clone https://github.com/your-username/AirSeg.git
cd AirSeg
pip install -r requirements.txt
