# Gaussian World Model (GWM)

[Project Page](https://gaussian-world-model.github.io/) | [Paper](https://arxiv.org/abs/2508.17600)

| ⚠️ WIP. More codes are still being sorted out.

This repository contains the training code for the Gaussian World Model (GWM), which is a latent Diffusion Transformer (DiT) combined with a 3D variational autoencoder, enabling fine-grained scene-level future state reconstruction with Gaussian Splatting. 

## 📦 Installation

```bash
# clone this repo, then:
export GWM_PATH=$(pwd)
echo "export GWM_PATH=$(pwd)" >> ~/.bashrc

# Install uv (if not already installed):
pip install uv

# ensure your cuda toolkit is installed
nvcc -V

# Install dependencies
uv sync
source .venv/bin/activate

uv pip install git+https://github.com/dcharatan/diff-gaussian-rasterization-modified --no-build-isolation
uv pip install git+https://github.com/facebookresearch/pytorch3d.git --no-build-isolation

# (Optional) Compile the CUDA kernels for Splatt3r
cd third_party/splatt3r/src/mast3r_src/dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../../../../..

# Download splatt3r checkpoint
mkdir -p third_party/splatt3r/checkpoints/splatt3r_v1.0
cd third_party/splatt3r/checkpoints/splatt3r_v1.0
wget https://huggingface.co/brandonsmart/splatt3r_v1.0/resolve/main/epoch%3D19-step%3D1200.ckpt
cd ../../../../..
```

## 🏞️ Pretraining

See [docs/pretraining.md](docs/pretraining.md).

### Docker Compose

The container entrypoint installs the system-Python dependencies, builds the
CUDA extensions for `TORCH_CUDA_ARCH_LIST`, downloads the Splatt3r checkpoint,
and links the partial DROID dataset automatically on first startup:

```bash
docker compose up --build -d
docker compose logs -f gaussianwm
docker compose exec gaussianwm bash
```

Setup results are cached in the `gaussianwm-setup-state` volume. Re-running
`docker compose up` verifies the imports and skips completed compilation. To
run setup manually or after changing dependencies:

```bash
docker compose exec gaussianwm bash scripts/setup_container.sh
```

Set `GWM_DOWNLOAD_SPLATT3R=0` in `docker-compose.yml` if the 3.1 GB Splatt3r
checkpoint should not be downloaded automatically. Change
`TORCH_CUDA_ARCH_LIST` when using a GPU architecture other than the RTX 2060's
compute capability 7.5.

## 🏷️ License

This repository is released under the MIT license.

## 🙏 Acknowledgement

Our code is built upon [iVideoGPT](https://github.com/thuml/iVideoGPT) and [diamond](https://github.com/eloialonso/diamond), thanks to the authors for the great work!

## 🥰 Citation

If you find this repository helpful, please consider citing:

```
@article{lu2025gwm,
  title={GWM: Towards Scalable Gaussian World Models for Robotic Manipulation},
  author={Lu, Guanxing and Jia, Baoxiong and Li, Puhao and Chen, Yixin and Wang, Ziwei and Tang, Yansong and Huang, Siyuan},
  booktitle={ICCV},
  year={2025},
  organization={IEEE}
}
```

