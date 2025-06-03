# 🧩 Cluster-Aware SfM Pipeline for the Image Matching Challenge 2025

This repository provides a clean, modular, and configurable pipeline for solving the [Image Matching Challenge 2025](https://www.kaggle.com/competitions/image-matching-challenge-2025). The goal is to reconstruct a 3D scene from a set of possibly related images.

This is a core problem in computer vision, as Images sourced online or through crowdsourcing are often:
- Mixed with unrelated photos
- Confusing due to near-identical structures (e.g. two similar building facades)
- Lacking reliable metadata like GPS or video sequences

While current methods work well in controlled environments with professional equipment, they struggle with real-world image collections. [Image Matching Challenge 2025](https://www.kaggle.com/competitions/image-matching-challenge-2025) challenges participants to **identify which images belong together** and **which should be discarded** for accurate 3D scene reconstruction.

<div align="center">
  <img src="demo-img/demo-sample.png" width="1000">
</div>

---

## 🛠️ Dependencies

This project is built on top of the baseline implementation [Baseline: DINOv2+ALIKED+LightGLUE](https://www.kaggle.com/code/octaviograu/baseline-dinov2-aliked-lightglue) which uses

- **Feature detection** [ALIKED](https://github.com/rpautrat/ALIKED)
- **Feature matching** [LightGlue](https://github.com/cvg/LightGlue)
- **3D reconstruction** [COLMAP](https://colmap.github.io/)

---

## 🚀 Getting Started

### 1. Clone and Install

```bash
git clone https://github.com/ludw-bj/3d-image-matching.git
cd 3d-image-matching

# Install core dependencies
pip install -r requirements.txt

# Optional: install hloc and LightGlue if needed externally
pip install git+https://github.com/cvg/Hierarchical-Localization@main
pip install git+https://github.com/cvg/LightGlue.git

```
### 2. Prepare Dataset

 >  Download the [Image Matching Challenge 2025 Dataset](https://www.kaggle.com/competitions/image-matching-challenge-2025/data), or place your own customized dataset into the `data_dir` directory.
 >  Ensure that you also provide the accompanying `train_labels.csv` (for training mode) or `sample_submission.csv` (for test mode), formatted according to the official structure specified in the challenge [dataset page](https://www.kaggle.com/competitions/image-matching-challenge-2025/data).

### 3. Run the Pipeline
```bash
python main.py \
  --data_dir /path/to/dataset \
  --workdir ./results/ \
  --cluster_method hierarchical \
  --device cuda
```
---

## 📤 Output
> Final camera pose submission
```bash
/kaggle/working/submission.csv
```

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{3D-Image-Matching,
  author = {Lu, Danwei},
  title = {Cluster-Aware SfM Pipeline for the Image Matching Challenge 2025},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ludw-bj/3d-image-matching}},
}
