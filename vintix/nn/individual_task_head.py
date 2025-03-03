from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class AcsDecoder(nn.Module):
    """Action decoder

    Decode action representations into
    continuous actions with MLP

    Args:
        task_metadata: task-specific metadata
        hidden_dim: input model dimension
        emb_activation: hidden activation
        out_activation: output_activation
    """

    def __init__(
            self,
            task_metadata: Dict[str, Any],
            hidden_dim: int,
            emb_activation: nn.Module = nn.LeakyReLU(),
            out_activation: nn.Module = nn.Identity(),
    ):
        super(AcsDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.acs_dim = task_metadata['action_dim']

        # Linear layer case
        self.layers = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            emb_activation,
            nn.Linear(self.hidden_dim, self.hidden_dim),
            emb_activation,
            nn.Linear(self.hidden_dim, self.hidden_dim),
            emb_activation,
            nn.Linear(self.hidden_dim, self.acs_dim),
            out_activation,
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Decode actions

        Action decoding forward pass

        Args:
            emb: hidden representation from model backbone
        Returns:
            torch.Tensor: decoded actions with shape
                (traj_len, acs_dim)
        """
        acs = self.layers(emb)
        return acs


class LinearObsDecoder(nn.Module):
    """Continuous observation decoder

    Decoder for proprioceptive observations

    Args:
        task_metadata: task-specific metadata
        hidden_dim: input embedding hidden dimension
        emb_activation: embedding activation
        out_activation: output activation
    """

    def __init__(
            self,
            task_metadata: Dict[str, Any],
            hidden_dim: int,
            emb_activation: nn.Module = nn.LeakyReLU(),
            out_activation: nn.Module = nn.Identity(),
    ):
        super(LinearObsDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.obs_shape = task_metadata['observation_shape']['proprio']

        # Linear layer case
        assert len(self.obs_shape) == 1
        self.layers = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            emb_activation,
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            emb_activation,
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.obs_shape[0]),
            out_activation,
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Obs decoder forward pass

        Proprioceptive observations decoder forward pass

        Args:
            emb: input backbone model embeddings
        Returns:
            torch.Tensor: decoded proprioceptive observations
                with shape (traj_len, obs_dim)
        """
        return self.layers(emb)


def get_obs_decoder(
        task_metadata: Dict[str, Any],
        hidden_dim: int,
        emb_activation: nn.Module = nn.LeakyReLU(),
        out_activation: nn.Module = nn.Identity(),
) -> nn.Module:
    """Get observation decoder

    Get observation decoder with respect to
    task's observation nature

    Args:
        task_metadata: task-specific metadata
        hidden_dim: hidden embedding dimension
        emb_activation: embedding activation
        out_activation: output activation
    Returns:
        nn.Module: module for observation decoding
    """
    obs_shape = task_metadata['observation_shape']
    if len(obs_shape) == 1 and 'proprio' in obs_shape.keys():
        # Continuous vector observation case
        decoder = LinearObsDecoder(
            task_metadata=task_metadata,
            hidden_dim=hidden_dim,
            emb_activation=emb_activation,
            out_activation=out_activation,
        )
    else:
        decoder = None
        raise NotImplementedError(
            f'No decoder for obs spec {obs_shape}, {obs_shape.keys()}'
        )
    return decoder


class IndividualTaskHeadNew(nn.Module):
    """Individual task model head

    Multi-head module for action prediction
    Decodes each groups's actions with a separate head

    Args:
        task2group: task_name-group_name mapping
        group_metadata: group-level metadata for observations
            and actions
        hidden_dim: hidden dimension of backbone model
        emb_activation: activation function for representations
        acs_activation: activation function for actions
    """

    def __init__(
            self,
            task2group: Dict[str, Any],
            task_metadata: Dict[str, Any],
            group_metadata: Dict[str, Any],
            hidden_dim: int,
            unnorm_acs: bool = False,
            emb_activation: nn.Module = nn.LeakyReLU(),
            acs_activation: nn.Module = nn.Identity(),
    ):
        super(IndividualTaskHeadNew, self).__init__()
        self.hidden_dim = hidden_dim
        self.task2group = task2group
        self.task_metadata = task_metadata
        self.group_metadata = group_metadata
        self.unnorm_acs = unnorm_acs

        self.acs_decoders = nn.ModuleDict(
            {gn: AcsDecoder(
                task_metadata=gm,
                hidden_dim=hidden_dim,
                emb_activation=emb_activation,
                out_activation=acs_activation,
            ) for gn, gm in self.group_metadata.items()}
        )

    def forward(
        self, emb: torch.Tensor,
        metadata: List[Dict[str, Any]]
    ) -> List[torch.Tensor]:
        """Multi-task head forward pass

        Args:
            emb: embeddings from transformer model
            metadata: batch-level metadata from multi-task
                encoder
        Returns:
            List[torch.Tensor]: predicted actions
                variable shaped tensors
        """
        acs = []
        for traj_emb, mt in zip(emb, metadata):
            task_name = mt['task_name']
            action = self.acs_decoders[self.task2group[task_name]].forward(
                traj_emb)
            if self.unnorm_acs:
                acs_mean = torch.tensor(
                    self.task_metadata[task_name]["acs_mean"])
                acs_mean = acs_mean.type(action.dtype)
                acs_mean = acs_mean.to(action.device)

                acs_std = torch.tensor(
                    self.task_metadata[task_name]["acs_std"])
                acs_std = acs_std.type(action.dtype)
                acs_std = acs_std.to(action.device)
                action = action * acs_std + acs_mean
            acs.append(action)
        return acs


def multitask_action_loss(
    acs: List[torch.Tensor],
    pred_acs: List[torch.Tensor],
    metadata: List[Dict[str, Any]],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Calculate multi-task loss

    Calculates multi-task loss for action

    Args:
        acs: list of action tensors from sampler
        pred_acs: list of predicted actions from model
        metadata: metadata of the trajectories
    Returns:
        Tuple[torch.Tensor, Dict[str, Any]]: tuple of
            1) Loss tensor with scalar value
            2) Logs dictionary for logging
    """
    logs = {}

    # Compute action loss
    acs_loss_items = []
    action_loss = 0
    for target_a, pred_a, mt in zip(
        acs, pred_acs, metadata
    ):
        a_type = mt['action_type']
        if a_type == 'continuous':
            acs_loss = F.mse_loss(pred_a, target_a, reduction='mean')
        else:
            acs_loss = 0
            raise NotImplementedError(
                f'Loss not implemented for action type {a_type}'
            )
        action_loss += acs_loss
        acs_loss_items.append(acs_loss.item())

    logs['acs_loss_items'] = acs_loss_items
    logs['action_loss'] = action_loss.item() / len(metadata)

    return action_loss, logs
