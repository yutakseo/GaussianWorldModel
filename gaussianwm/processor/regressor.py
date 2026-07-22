#!/usr/bin/env python3
# The MASt3R Gradio demo, modified for predicting 3D Gaussian Splats

# --- Original License ---
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# Usage: python gwm/data/regressor.py
import os
import sys
import einops
import lightning as L
import lpips
import torch
import torch.nn as nn
import time
import numpy as np
from typing import TypedDict
from pathlib import Path
from termcolor import cprint
from collections import OrderedDict

# 获取项目根目录路径
ROOT_DIR = Path(__file__).parent.parent.parent

# 添加third_party目录到Python路径
sys.path.append(str(ROOT_DIR / 'third_party/splatt3r'))
sys.path.append(str(ROOT_DIR / 'third_party/splatt3r/src/pixelsplat_src'))
sys.path.append(str(ROOT_DIR / 'third_party/splatt3r/src/mast3r_src'))
sys.path.append(str(ROOT_DIR / 'third_party/splatt3r/src/mast3r_src/dust3r'))

from dust3r.utils.image import load_images
from src.mast3r_src.dust3r.dust3r.losses import L21
from src.mast3r_src.mast3r.losses import ConfLoss, Regr3D
import data.scannetpp.scannetpp as scannetpp
import src.mast3r_src.mast3r.model as mast3r_model
import src.pixelsplat_src.benchmarker as benchmarker
import src.pixelsplat_src.decoder_splatting_cuda as pixelsplat_decoder
import utils.compute_ssim as compute_ssim
import utils.export as export
import utils.geometry as geometry
import utils.loss_mask as loss_mask
import utils.sh_utils as sh_utils


class SplatFile(TypedDict):
    import numpy.typing as npt
    """Data loaded from an antimatter15-style splat file."""
    centers: npt.NDArray[np.floating]
    """(N, 3)."""
    rgbs: npt.NDArray[np.floating]
    """(N, 3). Range [0, 1]."""
    opacities: npt.NDArray[np.floating]
    """(N, 1). Range [0, 1]."""
    covariances: npt.NDArray[np.floating]
    """(N, 3, 3)."""

gaussian_feature_to_dim = OrderedDict({
    'means': 3,
    'means_in_other_view': 3,
    # 'covariances': 9,
    'scales': 3,
    'rotations': 4,
    'sh': 3,
    'opacities': 1,
    # 'desc': 24,
    # 'desc_conf': 1,
})

def get_gaussain_tensor(pred):
    B = pred['scales'].shape[0]
    return torch.cat([
        pred[key].reshape(B, -1, value) 
        for key, value in gaussian_feature_to_dim.items() if key in pred
        ], dim=2
    )

def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')

    # print(ckpt.keys())  # dict_keys(['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks', 'optimizer_states', 'lr_schedulers', 'hparams_name', 'hyper_parameters'])

    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)

# class MAST3RGaussians(L.LightningModule):
class MAST3RGaussians(nn.Module):

    def __init__(self, config):

        super().__init__()

        # Save the config
        self.config = config

        # The encoder which we use to predict the 3D points and Gaussians,
        # trained as a modified MAST3R model. The model's configuration is
        # primarily defined by the pretrained checkpoint that we load, see
        # MASt3R's README.md
        self.encoder = mast3r_model.AsymmetricMASt3R(
            pos_embed='RoPE100',
            patch_embed_cls='ManyAR_PatchEmbed',
            img_size=(512, 512),
            head_type='gaussian_head',
            output_mode='pts3d+gaussian+desc24',
            depth_mode=('exp', -mast3r_model.inf, mast3r_model.inf),
            conf_mode=('exp', 1, mast3r_model.inf),
            enc_embed_dim=1024,
            enc_depth=24,
            enc_num_heads=16,
            dec_embed_dim=768,
            dec_depth=12,
            dec_num_heads=12,
            two_confs=True,
            use_offsets=config.use_offsets,
            sh_degree=config.sh_degree if hasattr(config, 'sh_degree') else 1
        )
        self.encoder.requires_grad_(False)
        self.encoder.downstream_head1.gaussian_dpt.dpt.requires_grad_(True)
        self.encoder.downstream_head2.gaussian_dpt.dpt.requires_grad_(True)

        # The decoder which we use to render the predicted Gaussians into
        # images, lightly modified from PixelSplat
        self.decoder = pixelsplat_decoder.DecoderSplattingCUDA(
            background_color=[0.0, 0.0, 0.0]
        )

        self.benchmarker = benchmarker.Benchmarker()

        # Loss criteria
        if config.loss.average_over_mask:
            self.lpips_criterion = lpips.LPIPS('vgg', spatial=True)
        else:
            self.lpips_criterion = lpips.LPIPS('vgg')

        if config.loss.mast3r_loss_weight is not None:
            self.mast3r_criterion = ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2)
            self.encoder.downstream_head1.requires_grad_(True)
            self.encoder.downstream_head2.requires_grad_(True)

        # self.save_hyperparameters()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        # if device is None:
        #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
            
        if os.path.isfile(pretrained_model_name_or_path):
            # Load the checkpoint
            checkpoint = torch.load(pretrained_model_name_or_path, map_location=device)
            
            # Create a config object
            from types import SimpleNamespace
            config = SimpleNamespace()
            
            # Extract necessary config parameters
            config.use_offsets = True
            config.loss = SimpleNamespace()
            config.loss.average_over_mask = True
            config.loss.apply_mask = True
            config.loss.mse_loss_weight = 1.0
            config.loss.lpips_loss_weight = 0.1
            config.loss.mast3r_loss_weight = None
            config.sh_degree = 1
            
            # Create the model
            model = cls(config)
            
            # Convert Lightning state_dict to regular PyTorch format
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if it exists in the keys
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v  # Remove 'model.' prefix
                else:
                    new_state_dict[k] = v
            
            # Load the state dict
            model.load_state_dict(new_state_dict, strict=False)
            # model = model.to(device)
            model.eval()
            return model
        else:
            raise NotImplementedError(f"Not implemented for {pretrained_model_name_or_path}")

    def forward(self, view1, view2):

        # Freeze the encoder and decoder
        with torch.no_grad():
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.encoder._encode_symmetrized(view1, view2)
            dec1, dec2 = self.encoder._decoder(feat1, pos1, feat2, pos2)

        # Train the downstream heads
        pred1 = self.encoder._downstream_head(1, [tok.float() for tok in dec1], shape1)
        pred2 = self.encoder._downstream_head(2, [tok.float() for tok in dec2], shape2)

        pred1['covariances'] = geometry.build_covariance(pred1['scales'], pred1['rotations'])
        pred2['covariances'] = geometry.build_covariance(pred2['scales'], pred2['rotations'])

        learn_residual = True
        if learn_residual:
            new_sh1 = torch.zeros_like(pred1['sh'])
            new_sh2 = torch.zeros_like(pred2['sh'])
            new_sh1[..., 0] = sh_utils.RGB2SH(einops.rearrange(view1['original_img'], 'b c h w -> b h w c'))
            new_sh2[..., 0] = sh_utils.RGB2SH(einops.rearrange(view2['original_img'], 'b c h w -> b h w c'))
            pred1['sh'] = pred1['sh'] + new_sh1
            pred2['sh'] = pred2['sh'] + new_sh2

        # Update the keys to make clear that pts3d and means are in view1's frame
        pred2['pts3d_in_other_view'] = pred2.pop('pts3d')
        pred2['means_in_other_view'] = pred2.pop('means')

        return pred1, pred2

    def training_step(self, batch, batch_idx):

        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch['context']

        # Predict using the encoder/decoder and calculate the loss
        pred1, pred2 = self.forward(view1, view2)
        color, _ = self.decoder(batch, pred1, pred2, (h, w))

        # Calculate losses
        mask = loss_mask.calculate_loss_mask(batch)
        loss, mse, lpips = self.calculate_loss(
            batch, view1, view2, pred1, pred2, color, mask,
            apply_mask=self.config.loss.apply_mask,
            average_over_mask=self.config.loss.average_over_mask,
            calculate_ssim=False
        )

        # Log losses
        self.log_metrics('train', loss, mse, lpips)
        return loss

    def validation_step(self, batch, batch_idx):

        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch['context']

        # Predict using the encoder/decoder and calculate the loss
        pred1, pred2 = self.forward(view1, view2)
        color, _ = self.decoder(batch, pred1, pred2, (h, w))

        # Calculate losses
        mask = loss_mask.calculate_loss_mask(batch)
        loss, mse, lpips = self.calculate_loss(
            batch, view1, view2, pred1, pred2, color, mask,
            apply_mask=self.config.loss.apply_mask,
            average_over_mask=self.config.loss.average_over_mask,
            calculate_ssim=False
        )

        # Log losses
        self.log_metrics('val', loss, mse, lpips)
        return loss

    def test_step(self, batch, batch_idx):

        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch['context']
        num_targets = len(batch['target'])

        # Predict using the encoder/decoder and calculate the loss
        with self.benchmarker.time("encoder"):
            pred1, pred2 = self.forward(view1, view2)
        with self.benchmarker.time("decoder", num_calls=num_targets):
            color, _ = self.decoder(batch, pred1, pred2, (h, w))

        # Calculate losses
        mask = loss_mask.calculate_loss_mask(batch)
        loss, mse, lpips, ssim = self.calculate_loss(
            batch, view1, view2, pred1, pred2, color, mask,
            apply_mask=self.config.loss.apply_mask,
            average_over_mask=self.config.loss.average_over_mask,
            calculate_ssim=True
        )

        # Log losses
        self.log_metrics('test', loss, mse, lpips, ssim=ssim)
        return loss

    def on_test_end(self):
        benchmark_file_path = os.path.join(self.config.save_dir, "benchmark.json")
        self.benchmarker.dump(os.path.join(benchmark_file_path))

    def calculate_loss(self, batch, view1, view2, pred1, pred2, color, mask, apply_mask=True, average_over_mask=True, calculate_ssim=False):

        target_color = torch.stack([target_view['original_img'] for target_view in batch['target']], dim=1)
        predicted_color = color

        if apply_mask:
            assert mask.sum() > 0, "There are no valid pixels in the mask!"
            target_color = target_color * mask[..., None, :, :]
            predicted_color = predicted_color * mask[..., None, :, :]

        flattened_color = einops.rearrange(predicted_color, 'b v c h w -> (b v) c h w')
        flattened_target_color = einops.rearrange(target_color, 'b v c h w -> (b v) c h w')
        flattened_mask = einops.rearrange(mask, 'b v h w -> (b v) h w')

        # MSE loss
        rgb_l2_loss = (predicted_color - target_color) ** 2
        if average_over_mask:
            mse_loss = (rgb_l2_loss * mask[:, None, ...]).sum() / mask.sum()
        else:
            mse_loss = rgb_l2_loss.mean()

        # LPIPS loss
        lpips_loss = self.lpips_criterion(flattened_target_color, flattened_color, normalize=True)
        if average_over_mask:
            lpips_loss = (lpips_loss * flattened_mask[:, None, ...]).sum() / flattened_mask.sum()
        else:
            lpips_loss = lpips_loss.mean()

        # Calculate the total loss
        loss = 0
        loss += self.config.loss.mse_loss_weight * mse_loss
        loss += self.config.loss.lpips_loss_weight * lpips_loss

        # MAST3R Loss
        if self.config.loss.mast3r_loss_weight is not None:
            mast3r_loss = self.mast3r_criterion(view1, view2, pred1, pred2)[0]
            loss += self.config.loss.mast3r_loss_weight * mast3r_loss

        # Masked SSIM
        if calculate_ssim:
            if average_over_mask:
                ssim_val = compute_ssim.compute_ssim(flattened_target_color, flattened_color, full=True)
                ssim_val = (ssim_val * flattened_mask[:, None, ...]).sum() / flattened_mask.sum()
            else:
                ssim_val = compute_ssim.compute_ssim(flattened_target_color, flattened_color, full=False)
                ssim_val = ssim_val.mean()
            return loss, mse_loss, lpips_loss, ssim_val

        return loss, mse_loss, lpips_loss

    def log_metrics(self, prefix, loss, mse, lpips, ssim=None):
        values = {
            f'{prefix}/loss': loss,
            f'{prefix}/mse': mse,
            f'{prefix}/psnr': -10.0 * mse.log10(),
            f'{prefix}/lpips': lpips,
        }

        if ssim is not None:
            values[f'{prefix}/ssim'] = ssim

        prog_bar = prefix != 'val'
        sync_dist = prefix != 'train'
        self.log_dict(values, prog_bar=prog_bar, sync_dist=sync_dist, batch_size=self.config.data.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.config.opt.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [self.config.opt.epochs // 2], gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


def get_reconstructed_scene(outdir, model, device, silent, save_ply, image_size, ios_mode, filelist):

    time_start = time.time()
    assert len(filelist) == 1 or len(filelist) == 2, "Please provide one or two images"
    if ios_mode:
        filelist = [f[0] for f in filelist]
    if len(filelist) == 1:
        filelist = [filelist[0], filelist[0]]

    if isinstance(filelist[0], str):
        imgs = load_images(filelist, size=image_size, verbose=not silent)
    else:
        imgs = filelist

    for img in imgs:
        img['img'] = img['img'].to(device)
        img['original_img'] = img['original_img'].to(device)
        img['true_shape'] = torch.from_numpy(img['true_shape'])

    output = model(imgs[0], imgs[1])

    pred1, pred2 = output
    # pred1
    # dict_keys(['pts3d', 'conf', 'desc', 'desc_conf', 'scales', 'rotations', 'sh', 'opacities', 'means', 'covariances'])
    # sh: 3
    # desc: 24, but not used
    # pts3d: 3D points, shape: (1, h, w, 3)
    # means: 3D points, shape: (1, h, w, 3), the same as pts3d

    # pred2
    # dict_keys(['conf', 'desc', 'desc_conf', 'scales', 'rotations', 'sh', 'opacities', 'covariances', 'pts3d_in_other_view', 'means_in_other_view'])
    
    print(f"{pred1['pts3d'].shape=}")   # [B, H, W, 3]
    print(f"{pred1['conf'].shape=}")    # [B, H, W]
    print(f"{pred1['desc'].shape=}")    # [B, H, W, 24]
    print(f"{pred1['desc_conf'].shape=}")    # [B, H, W]
    print(f"{pred1['scales'].shape=}")    # [B, H, W, 3]
    print(f"{pred1['rotations'].shape=}")    # [B, H, W, 4]
    print(f"{pred1['sh'].shape=}")    # [B, H, W, 3, 1]
    print(f"{pred1['opacities'].shape=}")    # [B, H, W, 1]

    # print(f"{pred2['pts3d_in_other_view'].shape=}")   # [B, H, W, 3]

    if save_ply:
        plyfile = os.path.join(outdir, 'gaussians.ply')
        export.save_as_ply(pred1, pred2, plyfile)
    
    # Add visualization of the gaussians
    if not silent:
        visualize_gaussians([plyfile], downsample_factor=0.5)
    time_end = time.time()
    print(f"🔥🔥🔥 Time taken: {time_end - time_start} seconds")

    return output

def visualize_gaussians(splat_paths, downsample_factor: float = 1.0):
    """Visualize gaussian splatting using viser.
    
    Args:
        splat_paths: List of paths to splat files
        downsample_factor: Factor to downsample the gaussians (0.0-1.0). Default 1.0 means no downsampling.
    """
    import viser
    from viser import transforms as tf
    server = viser.ViserServer()
    server.gui.configure_theme(dark_mode=True)
    gui_reset_up = server.gui.add_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
            [0.0, -1.0, 0.0]
        )

    for i, splat_path in enumerate(splat_paths):
        if splat_path.endswith(".ply"):
            splat_data = load_ply_file(splat_path, center=True)
            
            # Downsample the gaussians
            if 0.0 < downsample_factor < 1.0:
                num_points = len(splat_data["centers"])
                num_samples = int(num_points * downsample_factor)
                indices = np.random.choice(num_points, num_samples, replace=False)
                
                splat_data = {
                    "centers": splat_data["centers"][indices],
                    "rgbs": splat_data["rgbs"][indices],
                    "opacities": splat_data["opacities"][indices],
                    "covariances": splat_data["covariances"][indices],
                }
                print(f"Downsampled to {num_samples} gaussians")
        else:
            raise SystemExit("Please provide a filepath to a .splat or .ply file.")

        server.scene.add_transform_controls(f"/{i}")
        gs_handle = server.scene.add_gaussian_splats(
            f"/{i}/gaussian_splats",
            centers=splat_data["centers"],
            rgbs=splat_data["rgbs"],
            opacities=splat_data["opacities"],
            covariances=splat_data["covariances"],
        )

        remove_button = server.gui.add_button(f"Remove splat object {i}")

        @remove_button.on_click
        def _(_, gs_handle=gs_handle, remove_button=remove_button) -> None:
            gs_handle.remove()
            remove_button.remove()

    while True:
        time.sleep(10.0)

def load_ply_file(ply_file_path: Path, center: bool = False) -> SplatFile:
    from plyfile import PlyData
    from viser import transforms as tf
    """Load Gaussians stored in a PLY file."""
    start_time = time.time()

    SH_C0 = 0.28209479177387814

    plydata = PlyData.read(ply_file_path)
    v = plydata["vertex"]
    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
    wxyzs = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    colors = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    opacities = 1.0 / (1.0 + np.exp(-v["opacity"][:, None]))

    Rs = tf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    if center:
        positions -= np.mean(positions, axis=0, keepdims=True)

    num_gaussians = len(v)
    print(
        f"PLY file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return {
        "centers": positions,
        "rgbs": colors,
        "opacities": opacities,
        "covariances": covariances,
    }

# class Splatt3rRegressor(L.LightningModule):
class Splatt3rRegressor(nn.Module):
    """Simplified interface for Gaussian reconstruction with direct tensor processing."""
    
    def __init__(self, model_name: str = "brandonsmart/splatt3r_v1.0"):
        super().__init__()
        self.model = self.from_pretrained(model_name)

    @classmethod
    def from_pretrained(cls, model_name: str = "brandonsmart/splatt3r_v1.0") -> nn.Module:
        """Load pretrained model from local checkpoint."""
        weights_path = ROOT_DIR / "third_party/splatt3r/checkpoints/splatt3r_v1.0/epoch=19-step=1200.ckpt"
        
        # Use the MAST3RGaussians.from_pretrained method
        model = MAST3RGaussians.from_pretrained(weights_path)
        cprint(f"🔥🔥🔥 Splatt3r Model loaded from {weights_path}", "green")
        return model

    def forward(self, *image_tensors: torch.Tensor) -> tuple[dict, dict]:
        """
        Process batched image tensors (B,C,H,W format)
        
        Args:
            image_tensors: 1 or 2 image tensors with batch dimension (B,C,H,W)
            
        Returns:
            tuple: (pred1, pred2) dictionaries containing Gaussian parameters
        """
        # Input validation
        assert 1 <= len(image_tensors) <= 2, "Accept 1 or 2 input tensors"
        assert all(t.ndim == 4 for t in image_tensors), "Inputs must be batched (B,C,H,W)"
        
        # Ensure tensors are on correct device
        device = image_tensors[0].device
        view1_tensor = image_tensors[0].to(device)
        view2_tensor = view1_tensor if len(image_tensors) == 1 else image_tensors[1].to(device)

        # Build input format expected by MAST3RGaussians
        batch_size = view1_tensor.shape[0]

        # print(f"{view1_tensor.shape=}")
        
        view1 = {
            'img': view1_tensor,
            'true_shape': torch.tensor([[view1_tensor.shape[2], view1_tensor.shape[3]]] * batch_size, 
                                       dtype=torch.int32, device=device),
            'idx': torch.arange(batch_size, device=device),
            'instance': [str(i) for i in range(batch_size)],
            'original_img': view1_tensor
        }
        view2 = {
            'img': view2_tensor,
            'true_shape': torch.tensor([[view2_tensor.shape[2], view2_tensor.shape[3]]] * batch_size, 
                                       dtype=torch.int32, device=device),
            'idx': torch.arange(batch_size, device=device),
            'instance': [str(i) for i in range(batch_size)],
            'original_img': view2_tensor
        }

        pred1, pred2 = self.model(view1, view2)
        
        return pred1, pred2

    def forward_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        pred1, pred2 = self.forward(image_tensor)
        return get_gaussain_tensor(pred1), get_gaussain_tensor(pred2)

if __name__ == "__main__":
    image_size = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = "brandonsmart/splatt3r_v1.0"
    filename = "epoch=19-step=1200.ckpt"
    weights_path = ROOT_DIR / "third_party/splatt3r/checkpoints/splatt3r_v1.0/epoch=19-step=1200.ckpt"
    
    # Use the from_pretrained method
    model = MAST3RGaussians.from_pretrained(weights_path).to(device)
    # Alternatively: model = Splatt3rRegressor.from_pretrained(model_name, device=device).model

    examples = [
        [
            "path/to/your/image.png",
        ],
    ]
    outdir = "gaussianwm/data/output"
    os.makedirs(outdir, exist_ok=True)

    # while True:
    save_ply = True
    silent = False
    get_reconstructed_scene(outdir=outdir, model=model, device=device, 
                            silent=silent, save_ply=save_ply, 
                            image_size=image_size, ios_mode=True, filelist=examples)
    print(f"Reconstructed scene saved to {outdir}")