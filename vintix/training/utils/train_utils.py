import os
import h5py
import time
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch import nn, optim
from tqdm import tqdm


from vintix.nn.individual_task_head import multitask_action_loss
from vintix.vintix import Vintix
from vintix.nn.nn import RMSNorm
from vintix.training.utils.misc import Timeit


# https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136
def configure_optimizers(config: dataclass,
                         model: nn.Module) -> torch.optim.Optimizer:
    # separate out all parameters to those that
    # will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (
        torch.nn.LayerNorm, RMSNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif (pn.endswith('weight')
                    and isinstance(m, whitelist_weight_modules)):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif (pn.endswith('weight')
                    and isinstance(m, blacklist_weight_modules)):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('gamma'):
                no_decay.add(fpn)

    # additional parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    if 'module.encoder.prev_rew_padding' in param_dict:
        no_decay.add('module.encoder.prev_rew_padding')
    if 'module.encoder.prev_acs_padding' in param_dict:
        no_decay.add('module.encoder.prev_acs_padding')
    if 'encoder.prev_rew_padding' in param_dict:
        no_decay.add('encoder.prev_rew_padding')
    if 'encoder.prev_acs_padding' in param_dict:
        no_decay.add('encoder.prev_acs_padding')

    # validate that we considered every parameter
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, \
        "parameters %s made it into both decay/no_decay sets!" \
        % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, \
        "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(
            list(decay))], "weight_decay": config.weight_decay},
        {"params": [param_dict[pn]
                    for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups, lr=config.lr, betas=config.betas)
    return optimizer


def initialize_model(config: dataclass,
                     ds_metadata: Dict[str, Any],
                     device: torch.device
                     ) -> Tuple[nn.Module, Optional[torch.optim.Optimizer]]:
    """Initialize training model

    Function for initializing model

    Args:
        config: main config
        ds_metadata: dataset metadata
        device: training device

    Returns:
        (nn.Module, torch.optim.Optimizer): initialized model,
        optimizer
    """
    with open(config.stats_path, 'r') as f:
        scales = json.load(f)
    model = Vintix()
    if config.load_ckpt is None:
        for tn in ds_metadata.keys():
            rew_scale = config.dataset_config[tn]["reward_scale"]
            ds_metadata[tn]["reward_scale"] = rew_scale
            for key in scales[tn].keys():
                ds_metadata[tn][key] = scales[tn][key]

        model.init_model(asdict(config), ds_metadata)
    else:
        model.load_model(config.load_ckpt)
    model = model.to(device)

    if config.optimizer == 'Adam':
        optimizer = configure_optimizers(
            config,
            model
        )
    else:
        optimizer = None
    return model, optimizer


def log_metrics(logs: Dict[str, Any],
                metadata: Dict[str, Any]) -> None:
    """Log metrics to wandb

    Args:
        logs: dict with logs to upload
        metadata: dict with additional data for batch
    """
    acs_cont_loss = []
    per_task_loss = defaultdict(list)
    for lg, mt in zip(
            logs['acs_loss_items'],
            metadata):
        per_task_loss[mt['task_name']].append(lg)
        if mt['action_type'] == 'continuous':
            acs_cont_loss.append(lg)
        else:
            raise ValueError

    # training metrics log
    data_to_log = {
        "train/action_loss": logs['action_loss'],
        "train/lr": logs["lr"],
        "train/sample_batch_time, s": logs["sample_batch_time"],
        "train/forward_time, s": logs["forward_time"],
        "train/calc_loss_time, s": logs["calc_loss_time"],
        "train/update_time, s": logs["update_time"],
        "train/tokens": logs["tokens"]
    }
    for tn, losses in per_task_loss.items():
        data_to_log[f"train/{tn}_loss"] = sum(losses) / len(losses)
    wandb.log(data_to_log)


def to_torch_traj(
    traj: Dict[str, Any],
    device: torch.device = torch.device('cpu'),
    dtype=np.float32
) -> Dict[str, Any]:
    """Convert to torch.Tensor

    Convert trajectory's np.ndarray to
    torch.Tensor where needed

    Args:
        traj: trajectory from dataset
        device: torch.device to transfer tensor to
        dtype: target dtype (casted from numpy to
        torch dtypes)
    Returns:
        Dict[str, Any]: trajectory with torch.Tensor's
        instead of np.ndarray's
    """
    torch_batch = {}
    for k, v in traj.items():
        if isinstance(v, np.ndarray):
            torch_batch[k] = torch.from_numpy(
                v.astype(dtype)
            ).to(device)
        elif isinstance(v, dict):
            torch_batch[k] = to_torch_traj(
                v, device=device
            )
        else:
            torch_batch[k] = v
    return torch_batch


def train_loop(config: dataclass,
               ep: int,
               num_gpus: int,
               model: nn.Module,
               optimizer: optim.Optimizer,
               scheduler: optim.lr_scheduler,
               scaler: torch.cuda.amp.GradScaler,
               dataloader: torch.utils.data.DataLoader,
               device: torch.device,
               torch_dtype: torch.dtype,
               distributed: bool = False) -> None:
    """Training loop

    Args:
        config: main config
        model: torch model
        optimizer: torch optimizer
        scheduler: lr scheduler
        scaler: loss scaler for fp16
        dataloader: dataloader for Vintix
        device: torch.device
        torch_dtype: half precision torch type
        distributed: flag if training is ddp
    """
    model.train()
    sample_batch_start = time.time()
    step_num = 1
    for batch in tqdm(dataloader):
        batch = [to_torch_traj(t, device) for t in batch]
        sample_batch_end = time.time()

        if step_num % config.grad_accum_steps != 0:
            with model.no_sync():
                with torch.amp.autocast("cuda", dtype=torch_dtype):
                    with Timeit() as forward_time:
                        pred_acs, metadata = model(batch)

                    with Timeit() as calc_loss_time:
                        loss, logs = multitask_action_loss(
                            acs=[t['action'] for t in batch],
                            pred_acs=pred_acs,
                            metadata=metadata,
                        )
                    loss /= config.grad_accum_steps

                with Timeit() as update_time:
                    if config.precision == "fp16":
                        scaler.scale(loss).backward()
                    elif config.precision == "bf16":
                        loss.backward()
                    else:
                        raise KeyError()
        else:
            with torch.amp.autocast("cuda", dtype=torch_dtype):
                with Timeit() as forward_time:
                    pred_acs, metadata = model(batch)

                with Timeit() as calc_loss_time:
                    loss, logs = multitask_action_loss(
                        acs=[t['action'] for t in batch],
                        pred_acs=pred_acs,
                        metadata=metadata,
                    )

            with Timeit() as update_time:
                if config.precision == "fp16":
                    scaler.scale(loss).backward()
                    if config.clip_grad is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       config.clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                elif config.precision == "bf16":
                    loss.backward()
                    if config.clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       config.clip_grad)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    raise KeyError()
            scheduler.step()
        step_num += 1
        seen_tokens = num_gpus * config.batch_size * \
            config.context_len * (ep * len(dataloader) + step_num)

        logs["sample_batch_time"] = sample_batch_end - sample_batch_start
        logs["forward_time"] = forward_time.elapsed_time_gpu
        logs["calc_loss_time"] = calc_loss_time.elapsed_time_gpu
        logs["update_time"] = update_time.elapsed_time_gpu
        logs["lr"] = scheduler.get_last_lr()[0]
        logs["tokens"] = seen_tokens

        if config.local_rank == 0:
            log_metrics(logs, metadata)
        if distributed:
            dist.barrier()
        sample_batch_start = time.time()


def compute_stats(data_dir: str,
                  ds_names: List[str]
                  ) -> Dict[str, Any]:
    stats = {}
    for ds in list(ds_names.keys()):
        ds_path = os.path.join(data_dir, ds)
        files = [i for i in os.listdir(ds_path) if i.endswith(".h5")]
        files.sort()
        metadata_path = os.path.join(
            ds_path,
            os.path.basename(ds_path)+'.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        task_name = metadata["task_name"]
        print(task_name)
        obses = []
        acses = []
        for file in tqdm(files):
            f_path = os.path.join(ds_path, file)
            h5 = h5py.File(f_path, 'r')
            keys = sorted(list(h5.keys()), key=lambda x: int(x.split('-')[0]))
            for key in keys:
                group = h5.get(key)
                obs = np.array(group.get("proprio_observation"))
                acs = np.array(group.get("action"))
                obses.append(obs)
                acses.append(acs)
        obses = np.vstack(obses)
        acses = np.vstack(acses)
        obs_mean = obses.mean(axis=0)
        obs_std = obses.std(axis=0, ddof=1)
        acs_mean = acses.mean(axis=0)
        acs_std = acses.std(axis=0, ddof=1)
        stats[task_name] = {}
        stats[task_name]["obs_mean"] = obs_mean.tolist()
        stats[task_name]["obs_std"] = obs_std.tolist()
        stats[task_name]["acs_mean"] = acs_mean.tolist()
        stats[task_name]["acs_std"] = acs_std.tolist()
    return stats
