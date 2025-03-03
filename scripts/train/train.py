import datetime
import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import pyrallis
import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from vintix.data.torch_dataloaders import MultiTaskMapDataset
from vintix.training.utils.misc import set_seed
from vintix.training.utils.schedule import cosine_annealing_with_warmup
from vintix.training.utils.train_utils import (compute_stats,
                                               configure_optimizers,
                                               initialize_model, train_loop)


@dataclass
class TrainConfig:
    # dataloader config
    data_dir: str = "path/to/dataset/folder"
    context_len: int = 8192
    trajectory_sparsity: int = 257
    preload: bool = False
    last_frac: Optional[float] = None

    # model config
    action_emb_dim: int = 511
    observation_emb_dim: int = 511
    reward_emb_dim: int = 2
    hidden_dim: int = 1024  # a_dim+o_dim+r_dim
    transformer_depth: int = 20
    transformer_heads: int = 16
    attn_dropout: float = 0
    residual_dropout: float = 0
    normalize_qk: bool = True
    bias: bool = True
    parallel_residual: bool = False
    shared_attention_norm: bool = False
    norm_class: str = "LayerNorm"
    mlp_class: str = "GptNeoxMLP"
    intermediate_size: int = 4096
    inner_ep_pos_enc: bool = False
    norm_acs: bool = False
    norm_obs: bool = True

    # optimizer config
    optimizer: str = "Adam"
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 1e-1
    precision: str = "bf16"
    clip_grad: Optional[float] = None
    grad_accum_steps: int = 2
    warmup_ratio: float = 0.005

    # training config
    local_rank: int = 0
    epochs: int = 150
    batch_size: int = 8
    save_every: int = 2
    save_dir: str = "/path/to/savedir"
    stats_path: str = "vintix/stats.json"
    load_ckpt: Optional[str] = None
    start_epoch: int = 0
    seed: int = 5

    # Dataset config
    dataset_config_paths: List[str] = None

    # wandb config
    project: str = "Vintix"
    group: str = "default"
    name: str = "Vintix"

    def __post_init__(self):
        self.dataset_config = {}
        for dcp in self.dataset_config_paths:
            dc = OmegaConf.load(dcp)
            self.dataset_config = {**self.dataset_config, **dc}
        self.dataset_names = {
            v.path: v.group
                for k, v in self.dataset_config.items() if v.type == "default"
        }
        self.reward_scales = [
            v.reward_scale
                for v in self.dataset_config.values() if v.type == "default"
        ]
        self.episode_sparsity = [
            v.episode_sparsity
                for v in self.dataset_config.values() if v.type == "default"
        ]


HALF_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.seed)
    dist.init_process_group(backend='nccl',
                            timeout=datetime.timedelta(minutes=30))
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    config.local_rank = device_id
    num_gpus = int(os.getenv("WORLD_SIZE", "1"))

    print(f"NCCL Init on rank {rank}")

    if config.local_rank == 0:
        config.save_dir = os.path.join(config.save_dir, config.name)
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)

        _ = wandb.init(
            project=config.project,
            group=config.group,
            name=config.name,
            config=asdict(config),
            save_code=True,
        )
        print("COMPUTING STATS")
        stats = compute_stats(config.data_dir, config.dataset_names)
        with open(config.stats_path, 'w') as f:
            json.dump(stats, f)

    print("PREPARING DATASET")
    dataset = MultiTaskMapDataset(
        data_dir=config.data_dir,
        datasets_info=config.dataset_names,
        trajectory_len=config.context_len,
        trajectory_sparsity=config.trajectory_sparsity,
        ep_sparsity=config.episode_sparsity,
        preload=config.preload,
    )
    sampler = DistributedSampler(
        dataset,
        shuffle=True,
        drop_last=True,
        num_replicas=int(os.getenv("WORLD_SIZE", "1")),
        rank=int(config.local_rank),
        seed=0
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda x: x,
        sampler=sampler,
        drop_last=True,
        num_workers=12
    )

    print("INITIALIZING MODEL")
    scaler = torch.cuda.amp.GradScaler()
    torch_dtype = HALF_DTYPES[config.precision]
    dist.barrier()

    device = torch.device("cuda", config.local_rank)
    model, _ = initialize_model(config,
                                dataset.metadata,
                                device)
    model = DDP(model,
                device_ids=[config.local_rank],
                find_unused_parameters=True)
    optimizer = configure_optimizers(config, model)
    if config.load_ckpt:
        opt_dct = torch.load(os.path.join(config.load_ckpt, "state.pth"))
        optimizer.load_state_dict(opt_dct["optimizer_state"])

    if config.local_rank == 0:
        wandb.watch(model, log_freq=500)
    total_updates = len(dataloader) * config.epochs
    scheduler = cosine_annealing_with_warmup(
        optimizer=optimizer,
        warmup_steps=int(total_updates * config.warmup_ratio),
        total_steps=total_updates,
    )
    if config.load_ckpt:
        opt_dct = torch.load(os.path.join(config.load_ckpt, "state.pth"))
        scheduler.load_state_dict(opt_dct["scheduler_state"])

    print("MODEL INIT COMPLETE")
    print(f"PARAMETERS: {sum(p.numel() for p in model.parameters())}")
    print(f"Device {config.local_rank}")
    dist.barrier()

    if config.local_rank == 0:
        folder_name = "0".zfill(4) + "_epoch"
        model.module.save_model(
            os.path.join(config.save_dir, folder_name)
        )
        torch.save(
            {"optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict()},
            os.path.join(config.save_dir, folder_name, "state.pth")
        )

    for ep in range(config.start_epoch, config.epochs):
        sampler.set_epoch(ep)
        train_loop(config,
                   ep,
                   num_gpus,
                   model,
                   optimizer,
                   scheduler,
                   scaler,
                   dataloader,
                   device,
                   torch_dtype,
                   distributed=True)
        if ep % config.save_every == 0 and config.local_rank == 0:
            folder_name = f"{ep}".zfill(4) + "_epoch"
            model.module.save_model(
                os.path.join(config.save_dir, folder_name)
            )
            torch.save(
                {"optimizer_state": optimizer.state_dict(),
                 "scheduler_state": scheduler.state_dict()},
                os.path.join(config.save_dir, folder_name, "state.pth")
            )
        dist.barrier()


if __name__ == '__main__':
    train()
