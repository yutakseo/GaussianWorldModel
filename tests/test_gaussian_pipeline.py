import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from gaussianwm.gwm_predictor import state_dict_sha256
from gaussianwm.train_vae import LegacyGaussianBatchCache
from gaussianwm.diffusion.denoiser import (
    DenoiserConfig,
    GaussianLatentDenoiser,
    SigmaDistributionConfig,
)
from gaussianwm.diffusion.diffusion_sampler import (
    DiffusionSamplerConfig,
    GaussianDiffusionSampler,
)
from gaussianwm.diffusion.models import InnerModelConfig
from gaussianwm.encoder.models_ae import create_autoencoder


class GaussianVAEContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.points = torch.randn(2, 8, 14)
        self.points[..., 2] = self.points[..., 2].abs() + 1.0

    def test_public_query_decoder_outputs_one_gaussian_per_query(self):
        model = create_autoencoder(
            depth=2,
            dim=64,
            M=4,
            latent_dim=8,
            output_dim=14,
            N=8,
            deterministic=False,
            decoder_num_queries=8,
        )
        self.assertEqual(len(model.encoder_layers), 1)
        self.assertEqual(len(model.layers), 2)
        outputs = model(self.points, self.points)

        self.assertEqual(outputs["logits"].shape, (2, 8, 14))
        self.assertEqual(outputs["kl"].shape, (2,))
        self.assertTrue(torch.isfinite(outputs["logits"]).all())
        self.assertTrue((outputs["logits"][..., 3:6] > 0).all())
        self.assertTrue(
            torch.allclose(
                outputs["logits"][..., 6:10].norm(dim=-1),
                torch.ones(2, 8),
                atol=1.0e-5,
            )
        )
        self.assertTrue(
            (
                (outputs["logits"][..., 13] >= 0)
                & (outputs["logits"][..., 13] <= 1)
            ).all()
        )

        _, latents = model.encode(self.points)
        with self.assertRaisesRegex(ValueError, "query decoder"):
            model.decode(latents)

    def test_latent_only_decoder_remains_an_explicit_option(self):
        model = create_autoencoder(
            depth=1,
            dim=64,
            M=4,
            latent_dim=8,
            output_dim=14,
            N=8,
            deterministic=True,
            decoder_num_queries=None,
        )
        outputs = model(self.points)
        self.assertEqual(outputs["logits"].shape, (2, 4, 14))

    def test_vae_weight_identity_is_order_independent(self):
        first = {
            "weight": torch.arange(4, dtype=torch.float32),
            "bias": torch.tensor([1.0]),
        }
        second = {
            "bias": first["bias"].clone(),
            "weight": first["weight"].clone(),
        }
        self.assertEqual(
            state_dict_sha256(first), state_dict_sha256(second)
        )
        second["weight"][0] = -1.0
        self.assertNotEqual(
            state_dict_sha256(first), state_dict_sha256(second)
        )


class LegacyGaussianCacheContractTest(unittest.TestCase):
    def test_legacy_files_are_split_into_sequences(self):
        with TemporaryDirectory() as temporary_dir:
            split_dir = Path(temporary_dir) / "train"
            split_dir.mkdir()
            points = torch.zeros(4, 3, 14, dtype=torch.float16)
            points[:2] = 1
            points[2:] = 2
            intrinsics = torch.eye(3).repeat(4, 1, 1)
            torch.save(
                {
                    "version": 4,
                    "points": points,
                    "intrinsics": intrinsics,
                },
                split_dir / "batch_000000.pt",
            )
            torch.save(
                {
                    "version": 4,
                    "points": torch.full(
                        (2, 3, 14), 3, dtype=torch.float16
                    ),
                    "intrinsics": torch.eye(3).repeat(2, 1, 1),
                },
                split_dir / "batch_000001.pt",
            )

            cache = LegacyGaussianBatchCache(
                temporary_dir,
                split="train",
                segment_length=2,
                point_cloud_size=3,
            )
            self.assertEqual(cache.num_sequences, 3)
            loaded_points, loaded_intrinsics = cache.load_sequences(1, 2)
            self.assertEqual(loaded_points.shape, (4, 3, 14))
            self.assertEqual(loaded_intrinsics.shape, (4, 3, 3))
            self.assertTrue((loaded_points[:2] == 2).all())
            self.assertTrue((loaded_points[2:] == 3).all())


class GaussianEDMContractTest(unittest.TestCase):
    def test_training_and_sampling_keep_token_shapes(self):
        inner = InnerModelConfig(
            input_size=4,
            patch_size=1,
            in_channels=8,
            action_dim=3,
            hidden_size=32,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            class_dropout_prob=0.0,
            learn_sigma=False,
            context_length=2,
            token_based=True,
        )
        denoiser = GaussianLatentDenoiser(
            DenoiserConfig(
                inner_model=inner,
                sigma_data=0.5,
                sigma_offset_noise=0.0,
                noise_previous_obs=False,
                quantize_output=False,
            )
        )
        denoiser.setup_training(
            SigmaDistributionConfig(
                loc=-0.4,
                scale=1.2,
                sigma_min=0.002,
                sigma_max=20.0,
            )
        )

        latents = torch.randn(2, 4, 4, 8)
        actions = torch.randn(2, 4, 3)
        loss = denoiser(latents, actions)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        sampler = GaussianDiffusionSampler(
            denoiser,
            DiffusionSamplerConfig(
                num_steps_denoising=2,
                sigma_min=0.002,
                sigma_max=2.0,
                rho=7,
                order=1,
            ),
        )
        sampled, trajectory = sampler.sample(
            latents[:, :2], actions[:, 1]
        )
        self.assertEqual(sampled.shape, (2, 4, 8))
        self.assertEqual(len(trajectory), 3)
        self.assertTrue(torch.isfinite(sampled).all())


if __name__ == "__main__":
    unittest.main()
