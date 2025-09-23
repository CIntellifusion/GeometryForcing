<div align="center">

<h1>Geometry Forcing: Marrying Video Diffusion and 3D Representation for Consistent World Modeling </h1>
<a href="https://www.arxiv.org/abs/2507.07982">
<img src='https://img.shields.io/badge/arxiv-geometryforcing-darkred' alt='Paper PDF'></a>
<a href="https://geometryforcing.github.io/">
<img src='https://img.shields.io/badge/Project-Website-orange' alt='Project Page'></a>


[Haoyu Wu](https://cintellifusion.github.io/)$^{1*}$, Diankun Wu $^{2*}$, Tianyu He $^{1†}$, Junliang Guo $^{1}$, Yang Ye $^{1}$, Yueqi Duan $^{2}$, Jiang Bian $^{1}$

$^1$ Microsoft Research $^2$ Tsinghua University

($^*$ Equal Contribution. † Project Lead)

</div>

# Reference 

```
@article{wu2025geometryforcing,
  title={Geometry Forcing: Marrying Video Diffusion and 3D Representation for Consistent World Modeling},
  author={Wu, Haoyu and Wu, Diankun and He, Tianyu and Guo, Junliang and Ye, Yang and Duan, Yueqi and Bian, Jiang},
  journal={arXiv preprint arXiv:2507.07982},
  year={2025}
}
```

<!-- include asset/main.png -->

# Overview 
![](main.png)
**Geometry Forcing (GF) Overview.**
(a) Our proposed GF paradigm enhances video diffusion models by aligning with geometric features from VGGT~\citep{wang2025vggt}. 
(b) Compared to DFoT~\citep{dfot}, our method generates more temporally and geometrically consistent videos. 
(c) While baseline features fail to reconstruct meaningful 3D geometry, GF-learned features enable accurate 3D reconstruction.

# 🚀News

- [2025/9/24] We release code and checkpoint.
- [2025/9/22] [Geometry Forcing](https://geometryforcing.github.io/) is accepted to [NeurIPS 2025 NextVid Workshop](https://what-makes-good-video.github.io/) as an Oral!
- [2025/7/10] We release the paper and the project. 

# 💪Get Started 

## Setup Environments 

```shell
conda create -n geometryforcing python=3.10 -y
conda activate geometryforcing
pip install -r requirements.txt
```

## Connect to Weights & Biases:

We use Weights & Biases for logging. [Sign up](https://wandb.ai/login?signup=true) if you don't have an account, and *modify `wandb.entity` in `config.yaml` to your user/organization name*.

## Download Checkpoints and Data
1. Download pretrained checkpiont using huggingface: 
```shell
bash scripts/hf_download_checkpoints.sh
```


2. Download pretrained checkpiont using modelscope: 

```shell
bash scripts/ms_download_checkpoints.sh
```

3. Download and process RealEstate10k dataset to  `data/real-estate-10k`

## Generating Videos with Pretrained Models

### 1. Single Image to Long Video (256 Frames):

```shell
bash scripts/eval_geometry_forcing.sh
```

### 2. Single Image to Rotation Video (16 Frames):

```shell
bash scripts/eval_geometry_forcing_rotation.sh
```

## Training Geometry Forcing

```shell
bash scripts/train_geometry_forcing.sh
```