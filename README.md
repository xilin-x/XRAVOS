# XRAVOS: Few-Shot Video Object Segmentation in X-Ray Angiography

This repository is the official implementation of the paper:
**"Few-Shot Video Object Segmentation in X-Ray Angiography Using Local Matching and Spatio-Temporal Consistency Loss"**

---

## 📢 News
- **[2024.xx.xx]**: Repository created. We are currently cleaning up the code for public release. Stay tuned! 🚀

---

## ✨ Introduction

Video Object Segmentation (VOS) in X-ray angiography is challenging due to low contrast, overlapping structures, and dynamic blood flow. Our method addresses these issues through:

1.  **Local Matching Module**: Enhancing feature correspondence in low-contrast medical imaging.
2.  **Spatio-Temporal Consistency Loss**: Ensuring smooth and robust mask propagation across video frames.
3.  **Few-Shot Learning**: Achieving high precision with minimal annotated frames.

> **Abstract:** High-quality, densely annotated data serve as a crucial foundation for developing robust X-ray angiography segmentation models. However, obtaining per-object pixel-level annotations in the medical domain is both expensive and time-consuming, often requiring close collaboration between clinical experts and developers. This paper aims to reduce the annotation costs of X-ray angiography videos by leveraging few-shot video object segmentation (FSVOS), which separates target objects from the background using only a single annotated frame during inference. We introduce a novel FSVOS model that employs a local matching strategy to restrict the search space to the most relevant neighboring pixels. Rather than relying on inefficient standard im2col-like implementations (*e.g.*, spatial convolutions, depthwise convolutions and feature-shifting mechanisms) or hardware-specific CUDA kernels (*e.g.*, deformable and neighborhood attention), which often suffer from limited portability across non-CUDA devices, we reorganize the local sampling process through a direction-based sampling perspective. Specifically, we implement a non-parametric sampling mechanism that enables dynamically varying sampling regions. This approach provides the flexibility to adapt to diverse spatial structures without the computational costs of parametric layers and the need for model retraining. To further enhance feature coherence across frames, we design a supervised spatio-temporal contrastive learning scheme that enforces consistency in feature representations. In addition, we introduce a publicly available benchmark dataset for multi-object segmentation in X-ray angiography videos (MOSXAV), featuring detailed, manually labeled segmentation ground truth. Extensive experiments on the CADICA, XACV, and MOSXAV datasets show that our proposed FSVOS method outperforms current state-of-the-art video segmentation methods in terms of segmentation accuracy and generalization capability (*i.e.*, seen and unseen categories). This work offers enhanced flexibility and potential for a wide range of clinical applications. Our code will be made publicly available.

---

## 🛠️ Coming Soon
The following contents will be released soon:
- [ ] Full training and inference code (PyTorch)
- [ ] Pre-trained weights for X-ray Angiography datasets
- [ ] Data preprocessing scripts
- [ ] Evaluation tools and metrics

---

## 🚀 Getting Started

*Detailed instructions for installation and environment setup will be provided upon the official code release.*

### Requirements (Anticipated)
- Python 3.8+
- PyTorch >= 1.10
- CUDA Support

---

## 📊 Results

*(Optional: You can place a GIF or a teaser image here to show your segmentation results)*
![Teaser Image](docs/teaser.png)

---

## ✒️ Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{yourname2024angiovos,
  title={Few-Shot Video Object Segmentation in X-Ray Angiography Using Local Matching and Spatio-Temporal Consistency Loss},
  author={Your Name and Co-authors},
  journal={Journal/Conference Name},
  year={2024}
}
```

## 📧 Contact

For any questions, please open an issue or contact xilin.chibchin@outlook.com
