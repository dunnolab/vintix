from copy import copy
from typing import Any, Dict, List, Optional

import torch
from torch import nn


class LinearObsEncoder(nn.Module):
    """Continuous observation encoder

    Encoder for proprioceptive observations

    Args:
        group_metadata: task-specific metadata
        emb_dim: output embedding dimension
        emb_activation: output activation
    """

    def __init__(
            self,
            group_metadata: Dict[str, Any],
            emb_dim: int,
            emb_activation: nn.Module = nn.LeakyReLU()
    ):
        super(LinearObsEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.obs_shape = group_metadata['observation_shape']['proprio']

        # Linear layer case
        assert len(self.obs_shape) == 1
        self.layers = nn.Sequential(
            nn.Linear(self.obs_shape[0], self.emb_dim),
            emb_activation,
            nn.LayerNorm(self.emb_dim),
            nn.Linear(self.emb_dim, self.emb_dim),
            emb_activation,
            nn.LayerNorm(self.emb_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Obs encoder forward pass"""
        return self.layers(obs)


def get_obs_encoder(
        group_metadata: Dict[str, Any],
        emb_dim: int,
        emb_activation: nn.Module = nn.LeakyReLU(),
        image_encoder: nn.Module = None
) -> nn.Module:
    """Get observation encoder

    Get observation encoder with respect to
    task's observation nature

    Args:
        group_metadata: task-specific metadata
        emb_dim: output embedding dimension
        emb_activation: output activation
        image_encoder: module for encoding images
    Returns:
        nn.Module: module for observation processing
    """
    obs_shape = group_metadata['observation_shape']
    if len(obs_shape) == 1 and 'proprio' in obs_shape.keys():
        # Countinuous vector observation case
        encoder = LinearObsEncoder(
            group_metadata=group_metadata,
            emb_dim=emb_dim,
            emb_activation=emb_activation
        )
    elif len(obs_shape) == 1 and 'image' in obs_shape.keys():
        # Image case
        encoder = image_encoder
    else:
        encoder = None
        raise NotImplementedError(
            f'No encoder for obs spec {obs_shape}, {obs_shape.keys()}'
        )
    return encoder


class ContinuousAcsEncoder(nn.Module):
    """Continuous action encoder

    Encode continuous actions with MLP

    Args:
        group_metadata: group-specific metadata
        emb_dim: output embedding dimension
        emb_activation: output activation
    """

    def __init__(
            self,
            group_metadata: Dict[str, Any],
            emb_dim: int,
            emb_activation: nn.Module = nn.LeakyReLU()
    ):
        super(ContinuousAcsEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.acs_dim = group_metadata['action_dim']

        # Linear layer case
        assert group_metadata['action_type'] == 'continuous'
        self.layers = nn.Sequential(
            nn.Linear(self.acs_dim, self.emb_dim),
            emb_activation,
            nn.LayerNorm(self.emb_dim),
            nn.Linear(self.emb_dim, self.emb_dim),
            emb_activation,
            nn.LayerNorm(self.emb_dim),
        )

    def forward(
            self,
            acs: torch.Tensor,
            shift_padding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode actions

        Action encoding forward pass

        Args:
            acs: continuous actions trajectory with shape
                (traj_len, acs_dim) and dtype float
            shift_padding: if provided uses its value for
            1 step forward padding for correct SAR triplets
            alignment
        Returns:
            torch.Tensor: encoded continuous actions
        """
        acs_emb = self.layers(acs)
        if shift_padding is not None:
            acs_emb = torch.cat(
                [shift_padding, acs_emb[:-1]], dim=0
            )
        return acs_emb


class DiscreteAcsEncoder(nn.Module):
    """Discrete action encoder

    Encode discrete actions with embedding
    tables

    Args:
        group_metadata: group-specific metadata
        emb_dim: output embedding dimension
        emb_activation: output activation
    """

    def __init__(
            self,
            group_metadata: Dict[str, Any],
            emb_dim: int,
            emb_activation: nn.Module = nn.LeakyReLU()
    ):
        super(DiscreteAcsEncoder, self).__init__()
        assert group_metadata['action_type'] == 'discrete'
        action_dim = group_metadata['action_dim']
        self.emb_dim = emb_dim
        self.layers = nn.Sequential(
            nn.Embedding(action_dim, emb_dim),
            nn.Linear(emb_dim, emb_dim),
            emb_activation
        )

    def forward(
            self,
            acs: torch.Tensor,
            shift_padding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode actions

        Action encoding forward pass

        Args:
            acs: discrete actions trajectory with shape
                (traj_len, 1) and dtype int
            shift_padding: if provided uses its value for
            1 step forward padding for correct SAR triplets
            alignment
        Returns:
            torch.Tensor: encoded discrete actions
        """
        acs_emb = self.layers(acs.long())
        if shift_padding is not None:
            acs_emb = torch.cat(
                [shift_padding, acs_emb[:-1]], dim=0
            )
        return acs_emb


def get_acs_encoder(
        group_metadata: Dict[str, Any],
        emb_dim: int,
        emb_activation: nn.Module = nn.LeakyReLU()
) -> nn.Module:
    """Get action encoder

    Get action encoder with respect to
    task's action space nature

    Args:
        group_metadata: task-specific metadata
        emb_dim: output embedding dimension
        emb_activation: output activation
    Returns:
        nn.Module: module for action processing
    """
    action_type = group_metadata['action_type']
    if action_type == 'continuous':
        encoder = ContinuousAcsEncoder(
            group_metadata=group_metadata,
            emb_dim=emb_dim,
            emb_activation=emb_activation
        )
    elif action_type == 'discrete':
        encoder = DiscreteAcsEncoder(
            group_metadata=group_metadata,
            emb_dim=emb_dim,
            emb_activation=emb_activation
        )
    else:
        encoder = None
        raise NotImplementedError(
            f'No encoder for acs type {action_type}'
        )
    return encoder


class RewardEncoder(nn.Module):
    """Reward encoder

    Encoder for reward scalar value

    Args:
        emb_dim: output embedding dimension
        emb_activation: output activation
    """

    def __init__(
            self,
            emb_dim: int,
            emb_activation=nn.LeakyReLU()
    ):
        super(RewardEncoder, self).__init__()
        self.emb_dim = emb_dim

        # Linear layer case
        self.layers = nn.Sequential(
            nn.Linear(1, self.emb_dim),
            emb_activation
        )

    def forward(
            self,
            rews: torch.Tensor,
            shift_padding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode rewards

        Rewards encoding forward pass

        Args:
            rews: scalar rewards trajectory with shape
                (traj_len, 1) and dtype float
            shift_padding: if provided uses its value for
                1 step forward padding for correct SAR triplets
                alignment
        Returns:
            torch.Tensor: encoded reward scalars
        """
        rews_emb = self.layers(rews.unsqueeze(1))
        if shift_padding is not None:
            rews_emb = torch.cat(
                [shift_padding, rews_emb[:-1]], dim=0
            )
        return rews_emb


class IndividualTaskEncoderNew(nn.Module):
    """Individual task encoder

    Encode each task observation, action and reward with
    individual group encoder and concatenate representations
    into one token sequence. Maps each SAR triplet into
    one token and uses the following alignment:
    X: (prev_rew, prev_acs, obs) -> y: (acs)

    Args:
        task2group: task_name-group_name mapping
        task_metadata: task-level metadata for observations,
            actions and rewards
        group_metadata: group-level metadata for observations
            and actions
        action_emb_dim: action embedding dimension
        observation_emb_dim: observation embedding dimension
        reward_emb_dim: reward embedding dimension
        out_seq_len: output sequence length to left pad to
        max_emb_episode_length: maximum number of positional embedding
            tokens, if position is greater than `max_emb_episode_length`,
            then `max_emb_episode_length` is used
        image_encoder: image encoding module to share across all
            image based tasks
        emb_activation: activation function for representations
    """

    def __init__(
            self,
            task2group: Dict[str, Any],
            task_metadata: Dict[str, Any],
            group_metadata: Dict[str, Any],
            action_emb_dim: int,
            observation_emb_dim: int,
            reward_emb_dim: int,
            out_seq_len: int,
            max_emb_episode_length: int = 10000,
            inner_ep_pos_enc: bool = False,
            norm_acs: bool = False,
            norm_obs: bool = False,
            image_encoder: nn.Module = None,
            emb_activation: nn.Module = nn.LeakyReLU()
    ):
        super(IndividualTaskEncoderNew, self).__init__()
        self.task2group = task2group
        self.task_metadata = task_metadata
        self.group_metadata = group_metadata
        self.action_emb_dim = action_emb_dim
        self.observation_emb_dim = observation_emb_dim
        self.reward_emb_dim = reward_emb_dim
        self.total_dim = action_emb_dim + observation_emb_dim + reward_emb_dim
        self.out_seq_len = out_seq_len
        self.max_emb_episode_length = max_emb_episode_length
        self.inner_ep_pos_enc = inner_ep_pos_enc
        self.norm_acs = norm_acs
        self.norm_obs = norm_obs

        self.obs_encoders = nn.ModuleDict(
            {gn: get_obs_encoder(
                group_metadata=gm,
                emb_dim=observation_emb_dim,
                emb_activation=emb_activation,
                image_encoder=image_encoder
            ) for gn, gm in self.group_metadata.items()}
        )

        self.acs_encoders = nn.ModuleDict(
            {gn: get_acs_encoder(
                group_metadata=gm,
                emb_dim=action_emb_dim,
                emb_activation=emb_activation
            ) for gn, gm in self.group_metadata.items()}
        )

        self.rews_encoders = nn.ModuleDict(
            {gn: RewardEncoder(
                emb_dim=reward_emb_dim,
                emb_activation=emb_activation
            ) for gn, gm in self.group_metadata.items()}
        )
        if self.inner_ep_pos_enc:
            self.trans_ep_pos_emb = nn.Embedding(
                self.max_emb_episode_length,
                self.total_dim
            )

        # Padding parameters
        # Acs
        self.prev_acs_padding = nn.Parameter(
            torch.randn(size=(1, action_emb_dim)), requires_grad=True
        )
        # Rews
        self.prev_rew_padding = nn.Parameter(
            torch.randn(size=(1, reward_emb_dim)), requires_grad=True
        )

    def process_step_num(self, step_num: torch.Tensor) -> torch.LongTensor:
        """Process `step_num`

        Clip `step_num` according to
        `max_emb_episode_length`

        Args:
            step_num: torch.Tensor with transition positions
                within an episode
        Returns:
            torch.LongTensor: clipped step numbers
        """
        step_num = torch.clamp(
            step_num, min=None,
            max=self.max_emb_episode_length
        )
        return step_num.long()

    def within_episode_pos_emb(
            self, batch: List[Dict[str, Any]],
    ) -> List[torch.Tensor]:
        """Calculate positional embedding

        Calculate positional embedding

        Args:
            batch: a batch of trajectories
        Returns:
            List[torch.Tensor]: list of positional
            embedding for each trajectory
        """
        pos_emb = [self.trans_ep_pos_emb(
            self.process_step_num(
                traj['step_num']
            )) for traj in batch]
        return pos_emb

    def encode_traj(
            self, traj: Dict[str, Any]
    ) -> torch.Tensor:
        """Encode trajectory

        Encode individual trajectory

        Args:
            traj: sampled trajectory with torch.Tensor`s
        Returns:
            torch.Tensor: encoded trajectory tokens with
            shape (traj_len, acs_emb_dim + obs_emb_dim + rew_emb_dim)
        """
        # Get task name and observation field
        task_name = traj['task_name']
        group_name = self.task2group[task_name]
        reward_scale = self.task_metadata[task_name]["reward_scale"]
        obs = traj['observation']
        acs = traj['prev_action']
        if self.norm_acs:
            acs_mean = torch.tensor(self.task_metadata[task_name]["acs_mean"])
            acs_mean = acs_mean.type(acs.dtype)
            acs_mean = acs_mean.to(acs.device)

            acs_std = torch.tensor(self.task_metadata[task_name]["acs_std"])
            acs_std = acs_std.type(acs.dtype)
            acs_std = acs_std.to(acs.device)
            acs = (acs - acs_mean) / (acs_std + 1e-6)
        if self.norm_obs:
            obs_mean = torch.tensor(self.task_metadata[task_name]["obs_mean"])
            obs_mean = obs_mean.type(obs.dtype)
            obs_mean = obs_mean.to(obs.device)

            obs_std = torch.tensor(self.task_metadata[task_name]["obs_std"])
            obs_std = obs_std.type(obs.dtype)
            obs_std = obs_std.to(obs.device)
            obs = (obs - obs_mean) / (obs_std + 1e-6)

        # Create embeddings
        obs_emb = self.obs_encoders[group_name].forward(
            obs)
        acs_emb = self.acs_encoders[group_name].forward(
            acs
        )
        rews_emb = self.rews_encoders[group_name].forward(
            reward_scale * traj['prev_reward'].squeeze(1))

        # add padding to start of episodes
        mask = (traj['step_num'] == 0)
        acs_emb[mask] = self.prev_acs_padding.repeat(
            mask.sum(), 1).type(acs_emb.dtype)
        rews_emb[mask] = self.prev_rew_padding.repeat(
            mask.sum(), 1).type(rews_emb.dtype)

        sar_emb = torch.cat(
            [obs_emb, acs_emb, rews_emb], axis=1)
        out_emb = sar_emb

        # Add within episode positional embedding
        if self.inner_ep_pos_enc:
            pos_emb = self.trans_ep_pos_emb.forward(
                self.process_step_num(traj['step_num']))
            out_emb += pos_emb

        return out_emb

    def forward(
            self, batch: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Encoder forward pass

        Encode batch of different tasks' trajectories

        Args:
            batch: A batch of trajectories
        Returns:
            Dict[str, Union[Dict[str, Any], torch.Tensor]]:
            output dictionary with the following elements:
            {
                'metadata': List[Dict[str, Any]] - list of
                    trajectory-level metadata,
                'input_tokens': encoded trajectories with the shape:
                    (batch_size, max_traj_len, emb_dim),
                'padding_mask': padding mask for attention masking
                    with the shape: (batch_size, max_traj_len)
            }
        """
        sar_embs = [self.encode_traj(traj).unsqueeze(0)
                    for traj in batch]
        batch_metadata = []
        for traj in batch:
            task_name = traj['task_name']
            task_metadata = copy(
                self.group_metadata[self.task2group[task_name]])
            task_metadata['task_name'] = task_name
            batch_metadata.append(task_metadata)
        sar_embs = torch.cat(sar_embs, axis=0)

        return {
            'metadata': batch_metadata,
            'input_tokens': sar_embs
        }
