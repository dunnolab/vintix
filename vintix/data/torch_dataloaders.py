import json
import os
from copy import copy
from typing import Any, Dict, Generator, List, Optional, Tuple

import h5py
import numpy as np
from torch.utils.data import Dataset, IterableDataset


def left(name: str):
    """Get the start transaction number of a data group"""
    return int(name.split('-')[0])


def right(name: str):
    """Get the end transaction number of a data group"""
    return int(name.split('-')[1])


class FoundationMapDataset(Dataset):
    """Map dataset for single task

    Torch map dataset for single task

    Args:
        ds_path: the path to dataset
        context_len: the length of trajectory to sample
        traj_sparsity: if greater than 1, only trajectory_sparsity-th
            iteration will be used as a starting point for trajectory
            during the epoch
        ep_sparsity: if greater than 1, then only every ep_sparsity-th
            episode will be used for constructing trajectory while
            preserving episode order
        last_frac: if not None selects only the last fraction of
            collected trajectory, used for expert distillation
        preload: load dataset into RAM
        dtype: the base type of data
    """

    def __init__(self,
                 ds_path: str,
                 context_len: int,
                 traj_sparsity: int,
                 ep_sparsity: int = 1,
                 last_frac: Optional[float] = None,
                 preload: bool = False,
                 dtype: np.dtype = np.float32):
        self.ds_path = ds_path
        self.context_len = context_len + 1
        self.traj_sparsity = traj_sparsity
        self.ep_sparsity = ep_sparsity
        self.last_frac = last_frac
        self.preload = preload
        self.dtype = dtype

        files = os.listdir(ds_path)
        file_paths = [
            file for file in files if file.endswith('.h5')
        ]
        file_paths = [os.path.join(ds_path, file) for file in file_paths]
        file_paths.sort()
        self.h5_files = [
            h5py.File(path, 'r') for path in file_paths
        ]
        if preload:
            self.__preload()
        self.metadata_path = os.path.join(
            self.ds_path,
            os.path.basename(self.ds_path) + '.json'
        )
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.keys = [
            sorted(
                list(ds.keys()),
                key=lambda x: left(x)
            ) for ds in self.h5_files
        ]
        self.num_trans = [right(keys[-1]) for keys in self.keys]
        self.__precompute_splits()

    def __preload(self) -> None:
        """Preload hdf5 files"""
        self.preloaded = []
        for ds in self.h5_files:
            loaded_ds = {}
            for key in ds.keys():
                gr = ds.get(key)
                loaded_ds[key] = {}
                loaded_ds[key]['proprio_observation'] = np.array(
                    gr.get('proprio_observation'))
                loaded_ds[key]['action'] = np.array(gr.get('action'))
                loaded_ds[key]['reward'] = np.array(gr.get('reward'))
                loaded_ds[key]['step_num'] = np.array(gr.get('step_num'))
            self.preloaded.append(loaded_ds)

    def __precompute_splits(self) -> None:
        """Precompute splits for dataset"""
        self.splits = []
        self.transes = []
        # prepare transactions
        for ds_num in range(len(self.h5_files)):
            step_nums = []
            transes = []
            ds = self.h5_files[ds_num]
            for key in self.keys[ds_num]:
                step_num = np.array(ds.get(key).get('step_num'))
                step_nums.append(step_num)
            step_nums = np.hstack(step_nums)
            for i in range(step_nums.shape[0]):
                if step_nums[i] == 0:
                    transes.append([])
                transes[-1].append(i)
            transes = transes[::self.ep_sparsity]
            self.transes.append(np.hstack(transes))
        # prepare splits
        for ds_num in range(len(self.h5_files)):
            ds_key_num = 0
            max_trans_num = self.transes[ds_num].shape[0] - self.context_len
            for i in range(0, max_trans_num, self.traj_sparsity):
                trans_num = self.transes[ds_num][i]
                while trans_num > right(self.keys[ds_num][ds_key_num]):
                    ds_key_num += 1
                start_trans = i
                end_trans = i + self.context_len
                start_key = ds_key_num
                end_key = ds_key_num
                while (right(self.keys[ds_num][end_key]) <
                       self.transes[ds_num][end_trans - 1]):
                    end_key += 1
                if self.last_frac is None:
                    self.splits.append(
                        (ds_num, start_trans, start_key, end_trans, end_key))
                else:
                    if (self.transes[ds_num][start_trans] >=
                            self.num_trans[ds_num] * (1 - self.last_frac)):
                        self.splits.append(
                            (ds_num,
                             start_trans,
                             start_key,
                             end_trans,
                             end_key)
                        )

    def __get_data(self,
                   ds_num: int,
                   key_num: int) -> Tuple[np.ndarray,
                                          np.ndarray,
                                          np.ndarray,
                                          np.ndarray]:
        """Get data from group

        Get data from ds_num-th dataset key_num-th group

        Args:
            ds_num: the number of dataset
            key_num: the number of key in self.keys

        Returns:
            (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
                observations sequence, action sequence, reward
                sequence, step number within epoch sequence
        """
        if self.preload:
            ds = self.preloaded[ds_num]
            gr = ds[self.keys[ds_num][key_num]]
            obs = gr['proprio_observation']
            acs = gr['action']
            rew = gr['reward']
            stp_num = gr['step_num']
        else:
            ds = self.h5_files[ds_num]
            gr = ds.get(self.keys[ds_num][key_num])
            obs = np.array(gr.get('proprio_observation'))
            acs = np.array(gr.get('action'))
            rew = np.array(gr.get('reward'))
            stp_num = np.array(gr.get('step_num'))
        return obs, acs, rew, stp_num

    def __prepare_sample(self,
                         idx: int) -> Tuple[np.ndarray,
                                            np.ndarray,
                                            np.ndarray,
                                            np.ndarray]:
        """Prepare sample to return

        Prepare sample to return according to split
        parameters

        Args:
            idx: the number of split

        Returns:
            (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
                observations sequence, action sequence, reward
                sequence, step number within epoch sequence
        """
        obs = []
        acs = []
        rew = []
        stp_num = []

        ds_num, st_t, st_key, en_t, en_key = self.splits[idx]
        shift = left(self.keys[ds_num][st_key])
        for key_num in range(st_key, en_key + 1):
            cur_obs, cur_acs, cur_rew, cur_stp_num = self.__get_data(
                ds_num, key_num)
            obs.append(
                cur_obs
            )
            acs.append(
                cur_acs
            )
            rew.append(
                cur_rew
            )
            stp_num.append(
                cur_stp_num
            )
        obs = np.vstack(obs)
        acs = np.vstack(acs)
        rew = np.hstack(rew)
        rew = np.expand_dims(rew, -1)
        stp_num = np.hstack(stp_num)

        subsample = self.transes[ds_num][st_t:en_t] - shift
        obs = obs[subsample].astype(self.dtype)
        acs = acs[subsample].astype(self.dtype)
        rew = rew[subsample].astype(self.dtype)
        stp_num = stp_num[subsample].astype(int)
        return obs, acs, rew, stp_num

    def __getitem__(self,
                    idx: int) -> Dict[str, np.ndarray]:
        obs, acs, rew, stp_num = self.__prepare_sample(idx)
        sample = {}
        sample['observation'] = obs[1:]
        sample['prev_action'] = acs[0:-1]
        sample['prev_reward'] = rew[:-1]
        sample['action'] = acs[1:]
        sample['step_num'] = stp_num[1:]
        return sample

    def __len__(self) -> int:
        return len(self.splits)


class FoundationRandomMapDataset(Dataset):
    """Random map dataset for single task

    Random map torch dataset for single task.
    Episode sparsity and trajectory sparsity are
    randomized.

    Args:
        ds_path: path to dataset
        context_len: length of trajectory to sample
        traj_sparsity: If greater than 1, only trajectory_sparsity-th
            iteration will be used as a starting point for trajectory
            during the epoch
        ep_sparsity: if greater than 1, then only every ep_sparsity-th
            episode will be used for constructing trajectory while
            preserving episode order
        last_frac: of not None selects only the last fraction of
            collected trajectory, used for expert distillation
        preload: load dataset into RAM
        dtype: base type of data
    """

    def __init__(self,
                 ds_path: str,
                 context_len: int,
                 traj_sparsity: int,
                 ep_sparsity: int = 1,
                 last_frac: Optional[float] = None,
                 preload: bool = False,
                 dtype: np.dtype = np.float32):
        self.ds_path = ds_path
        self.context_len = context_len + 1
        self.traj_sparsity = traj_sparsity
        self.ep_sparsity = ep_sparsity
        self.last_frac = last_frac
        self.preload = preload
        self.dtype = dtype

        files = os.listdir(ds_path)
        file_paths = [
            file for file in files if file.endswith('.h5')
        ]
        file_paths = [os.path.join(ds_path, file) for file in file_paths]
        file_paths.sort()
        self.h5_files = [
            h5py.File(path, 'r') for path in file_paths
        ]
        if preload:
            self.__preload()
        self.metadata_path = os.path.join(
            self.ds_path,
            os.path.basename(self.ds_path) + '.json'
        )
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.keys = [
            sorted(
                list(ds.keys()),
                key=lambda x: left(x)
            ) for ds in self.h5_files
        ]
        self.num_trans = [right(keys[-1]) for keys in self.keys]
        self.__precompute_splits()

    def __preload(self):
        """Preload hdf5 files"""
        self.preloaded = []
        for ds in self.h5_files:
            loaded_ds = {}
            for key in ds.keys():
                gr = ds.get(key)
                loaded_ds[key] = {}
                loaded_ds[key]['proprio_observation'] = np.array(
                    gr.get('proprio_observation'))
                loaded_ds[key]['action'] = np.array(gr.get('action'))
                loaded_ds[key]['reward'] = np.array(gr.get('reward'))
                loaded_ds[key]['step_num'] = np.array(gr.get('step_num'))
            self.preloaded.append(loaded_ds)

    def __precompute_splits(self):
        """Precompute splits for dataset"""
        self.splits = []
        self.transes = []
        self.ep_lens = []
        # prepare transactions
        for ds_num in range(len(self.h5_files)):
            step_nums = []
            transes = []
            ep_lens = []
            ds = self.h5_files[ds_num]
            for key in self.keys[ds_num]:
                step_num = np.array(ds.get(key).get('step_num'))
                step_nums.append(step_num)
            step_nums = np.hstack(step_nums)
            for i in range(step_nums.shape[0]):
                if step_nums[i] == 0:
                    transes.append([])
                    ep_lens.append(0)
                transes[-1].append(i)
                ep_lens[-1] += 1
            transes = transes
            self.transes.append(np.hstack(transes))
            self.ep_lens.append(np.cumsum(ep_lens))
        # prepare splits
        for ds_num in range(len(self.h5_files)):
            ds_key_num = 0
            ep_num = 0
            max_trans_num = self.transes[ds_num].shape[0] - \
                self.context_len * self.ep_sparsity - self.traj_sparsity
            for i in range(0, max_trans_num, self.traj_sparsity):
                trans_num = self.transes[ds_num][i]
                while trans_num > right(self.keys[ds_num][ds_key_num]):
                    ds_key_num += 1
                while trans_num >= self.ep_lens[ds_num][ep_num]:
                    ep_num += 1
                start_trans = i
                start_key = ds_key_num
                start_ep_num = ep_num
                if self.last_frac is None:
                    self.splits.append(
                        # , end_trans, end_key))
                        (ds_num, start_trans, start_key, start_ep_num))
                else:
                    if (self.transes[ds_num][start_trans] >=
                            self.num_trans[ds_num] * (1 - self.last_frac)):
                        self.splits.append(
                            # , end_trans, end_key))
                            (ds_num, start_trans, start_key, start_ep_num))

    def __get_data(self,
                   ds_num: int,
                   key_num: int) -> Tuple[np.ndarray,
                                          np.ndarray,
                                          np.ndarray,
                                          np.ndarray]:
        """Get data from group

        Get data from ds_num-th dataset key_num-th group

        Args:
            ds_num: the number of dataset
            key_num: the number of key in self.keys

        (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
                observations sequence, action sequence, reward
                sequence, step number within epoch sequence
        """
        if self.preload:
            ds = self.preloaded[ds_num]
            gr = ds[self.keys[ds_num][key_num]]
            obs = gr['proprio_observation']
            acs = gr['action']
            rew = gr['reward']
            stp_num = gr['step_num']
        else:
            ds = self.h5_files[ds_num]
            gr = ds.get(self.keys[ds_num][key_num])
            obs = np.array(gr.get('proprio_observation'))
            acs = np.array(gr.get('action'))
            rew = np.array(gr.get('reward'))
            stp_num = np.array(gr.get('step_num'))
        return obs, acs, rew, stp_num

    def __prepare_sample(self,
                         idx: int) -> Tuple[np.ndarray,
                                            np.ndarray,
                                            np.ndarray,
                                            np.ndarray]:
        """Prepare sample to return

        Prepare sample to return according to split
        parameters

        Args:
            idx: the number of split

        Returns:
            (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
                observations sequence, action sequence, reward
                sequence, epoch step number sequence
        """
        obs = []
        acs = []
        rew = []
        stp_num = []

        ds_num, st_t, st_key, ep_num = self.splits[idx]
        st_t += np.random.randint(0, self.traj_sparsity)
        while st_t > right(self.keys[ds_num][st_key]):
            st_key += 1
        while st_t >= self.ep_lens[ds_num][ep_num]:
            ep_num += 1

        shift = left(self.keys[ds_num][st_key])
        cur_ctx_len = self.ep_lens[ds_num][ep_num] - st_t
        trans_to_choose = [True] * cur_ctx_len
        cur_ep_num = ep_num
        cur_key = st_key
        ep_starts = [0]
        while (cur_ctx_len < self.context_len and
                cur_ep_num < len(self.ep_lens[ds_num]) - 1):
            ep_starts.append(len(trans_to_choose))
            cur_ep_num += 1
            cur_ep_len = self.ep_lens[ds_num][cur_ep_num] - \
                self.ep_lens[ds_num][cur_ep_num - 1]

            if np.random.rand() < 1 / self.ep_sparsity:  #
                trans_to_choose += cur_ep_len * [True]
                cur_ctx_len += cur_ep_len
            else:
                trans_to_choose += cur_ep_len * [False]

            # compute if we need new file
            while (st_t + len(trans_to_choose) - 1 >
                    right(self.keys[ds_num][cur_key])):
                cur_key += 1
        ep_num = 0
        perm = np.random.permutation(
            len(ep_starts) - 1)  # make rand eps included
        while cur_ctx_len < self.context_len and ep_num < len(ep_starts) - 1:
            if trans_to_choose[ep_starts[perm[ep_num]]] is False:
                for i in range(ep_starts[perm[ep_num]],
                               ep_starts[perm[ep_num] + 1]):
                    trans_to_choose[i] = True
                    cur_ctx_len += 1
            ep_num += 1

        for key_num in range(st_key, cur_key + 1):
            cur_obs, cur_acs, cur_rew, cur_stp_num = self.__get_data(
                ds_num, key_num)
            obs.append(
                cur_obs
            )
            acs.append(
                cur_acs
            )
            rew.append(
                cur_rew
            )
            stp_num.append(
                cur_stp_num
            )
        obs = np.vstack(obs)
        acs = np.vstack(acs)
        rew = np.hstack(rew)
        rew = np.expand_dims(rew, -1)
        stp_num = np.hstack(stp_num)

        subsample = self.transes[ds_num][st_t:st_t +
                                         len(trans_to_choose)] - shift
        subsample = subsample[trans_to_choose]
        subsample = subsample[:self.context_len]
        obs = obs[subsample].astype(self.dtype)
        acs = acs[subsample].astype(self.dtype)
        rew = rew[subsample].astype(self.dtype)
        stp_num = stp_num[subsample].astype(int)
        return obs, acs, rew, stp_num

    def __getitem__(self,
                    idx: int) -> Dict[str, np.ndarray]:
        obs, acs, rew, stp_num = self.__prepare_sample(idx)
        sample = {}
        sample['observation'] = obs[1:]
        sample['prev_action'] = acs[0:-1]
        sample['prev_reward'] = rew[:-1]
        sample['action'] = acs[1:]
        sample['step_num'] = stp_num[1:]
        return sample

    def __len__(self) -> int:
        return len(self.splits)


class MultiTaskMapDataset(Dataset):
    """Map dataset for multi-task

    Torch map dataset for multi-task

    Args:
        data_dir: the directory containing HDF5 datasets
        datasets_info: the dict where key is path to folder with
            hdf5 files and value is group name for this task
        trajectory_len: the length of trajectory to sample
        trajectory_sparsity: if greater than 1, only trajectory_sparsity-th
            iteration will be used as a starting point for trajectory
            during the epoch
        ep_sparsity: if greater than 1, then only every ep_sparsity-th
            episode will be used for constructing trajectory while
            preserving episode order
        last_frac: if not None selects only the last fraction of
            collected trajectory, used for expert distillation
        preload: load dataset into RAM
        randomized: use randomized episode sparsity (WIP)
    """

    def __init__(self,
                 data_dir: str,
                 datasets_info: Dict[str, str],
                 trajectory_len: int,
                 trajectory_sparsity: int,
                 ep_sparsity: Tuple[List[int], int] = 1,
                 last_frac: Optional[float] = None,
                 preload: Tuple[List[bool], bool] = False,
                 randomized: bool = False):
        self.data_dir = data_dir
        self.datasets_info = datasets_info
        self.dataset_paths = list(self.datasets_info.keys())
        self.trajectory_len = trajectory_len
        self.trajectory_sparsity = trajectory_sparsity
        self.last_frac = last_frac
        self.randomized = randomized
        if isinstance(ep_sparsity, int):
            self.ep_sparsity = [ep_sparsity] * len(self.dataset_paths)
        else:
            assert len(ep_sparsity) == len(self.dataset_paths)
            self.ep_sparsity = ep_sparsity
        if isinstance(preload, bool):
            self.preload = [preload] * len(self.dataset_paths)
        else:
            assert len(preload) == len(self.dataset_paths)
            self.preload = preload

        self.datasets = []
        dataset_lens = [0]
        self.task2group = {}
        for i, ds_path in enumerate(self.dataset_paths):
            dataset_full_path = os.path.join(data_dir, ds_path)
            if self.randomized:
                single_ds_class = FoundationRandomMapDataset
            else:
                single_ds_class = FoundationMapDataset
            new_dataset = single_ds_class(
                ds_path=dataset_full_path,
                context_len=trajectory_len,
                traj_sparsity=trajectory_sparsity,
                ep_sparsity=self.ep_sparsity[i],
                last_frac=last_frac,
                preload=self.preload[i])
            self.datasets.append(new_dataset)
            dataset_lens.append(len(new_dataset))
            task_name = new_dataset.metadata['task_name']
            self.task2group[task_name] = self.datasets_info[ds_path]
        self.cumsum_lens = np.cumsum(dataset_lens)
        self.__create_metadata()

    def __create_metadata(self) -> None:
        """Create dataset metadata"""
        self.metadata = {}
        for ds in self.datasets:
            task_name = ds.metadata['task_name']
            self.metadata[task_name] = copy(ds.metadata)
            self.metadata[task_name]["group_name"] = self.task2group[task_name]
            self.metadata[task_name].pop('task_name')

    def __find_dataset_num(self, idx: int) -> int:
        """Find dataset number for index"""
        left = 0
        right = self.cumsum_lens.shape[0]
        while right - left > 1:
            piv = (right + left) // 2
            if idx < self.cumsum_lens[piv]:
                right = piv
            else:
                left = piv
        return left

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get idx-th sample"""
        ds_num = self.__find_dataset_num(idx)
        ds_idx = idx - self.cumsum_lens[ds_num]
        sample = self.datasets[ds_num][ds_idx]
        sample['task_name'] = self.datasets[ds_num].metadata['task_name']
        return sample

    def __len__(self) -> int:
        """Get length of dataset"""
        return self.cumsum_lens[-1]
