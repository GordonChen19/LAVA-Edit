from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import MaskedLoRADataset
from .lora import LoRAManager, inject_lora_into_attention
from ..LUV.pipeline import ImageConditioning, WanSlimPipeline
from .prompting import random_seed


@dataclass
class TrainingConfig:
    dataset_root: Path
    output_dir: Path
    prompt_prefix: str = "p3rs0n,"
    manual_prompt: Optional[str] = None
    max_steps: int = 1000
    batch_size: int = 1
    gradient_accumulation: int = 1
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.0
    lora_rank: int = 16
    lora_alpha: float = 16.0
    num_inference_steps: int = 30
    seed: Optional[int] = None


class BackgroundLoRATrainer:
    """Fine-tunes attention-only LoRA modules on non-masked regions."""

    def __init__(self, pipeline: WanSlimPipeline, config: TrainingConfig):
        self.pipeline = pipeline
        self.pipeline.train()
        self.config = config
        self.device = pipeline.device
        self.dtype = pipeline.torch_dtype
        self.pipeline.load_models_to_device(["text_encoder", "dit", "vae"])

        self.dataset = MaskedLoRADataset(config.dataset_root)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=lambda batch: batch,
        )

        self.lora_manager: LoRAManager = inject_lora_into_attention(
            self.pipeline.dit,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
        )

        self.optimizer = AdamW(
            self.lora_manager.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.global_step = 0
        torch.manual_seed(random_seed(config.seed))

    def _object_mask(self, conditioning: ImageConditioning) -> torch.Tensor:
        """Binary mask for user-selected objects in latent space."""

        mask = conditioning.latent_mask.mean(dim=1, keepdim=True)  # [B, 1, F, H, W]
        mask = (mask > 0.5).to(self.device, self.dtype)
        return mask

    def _background_mask(self, conditioning: ImageConditioning) -> torch.Tensor:
        return 1.0 - self._object_mask(conditioning)

    def train(self):
        progress = tqdm(total=self.config.max_steps, desc="LoRA training", dynamic_ncols=True)
        data_iter = iter(self.dataloader)

        while self.global_step < self.config.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            loss = self._train_step(batch)
            self.global_step += 1
            progress.update(1)
            progress.set_postfix({"loss": f"{loss:.4f}"})

            if self.global_step % 100 == 0:
                self.save_checkpoint()

        self.save_checkpoint()
        progress.close()

    def _train_step(self, batch):
        grad_accum = self.config.gradient_accumulation
        total_loss = 0.0

        for sample in batch:
            conditioning = self.pipeline.prepare_conditioning(
                image=sample["edited_image"],
                pseudo_video_path=sample["pseudo_video"],
                mask_video_path=sample["mask_video"],
            )

            prompt_text = sample["prompt"] or self.config.manual_prompt or self.config.prompt_prefix
            prompt_ctx = self.pipeline.encode_prompt(prompt_text, positive=True)

            self.pipeline.scheduler.set_timesteps(self.config.num_inference_steps, shift=5.0)
            timestep_index = torch.randint(
                low=0,
                high=len(self.pipeline.scheduler.timesteps),
                size=(1,),
            ).item()
            timestep = self.pipeline.scheduler.timesteps[timestep_index].unsqueeze(0).to(self.device, self.dtype)

            latents = conditioning.y.clone().to(self.device, self.dtype)
            noise = torch.randn_like(latents)
            if hasattr(self.pipeline.scheduler, "add_noise"):
                noisy_latents = self.pipeline.scheduler.add_noise(latents, noise, timestep)
            else:
                noisy_latents = latents + noise

            noise_pred = self.pipeline.compute_noise_prediction(
                noisy_latents,
                conditioning,
                prompt_ctx,
                timestep,
                tea_cache=None,
            )

            background_mask = self._background_mask(conditioning)
            mse = ((noise_pred - noise) ** 2) * background_mask
            loss = mse.mean() / grad_accum
            loss.backward()
            total_loss += loss.item()

        self.optimizer.step()
        self.optimizer.zero_grad()
        return total_loss


