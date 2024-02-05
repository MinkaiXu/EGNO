import os
import random
import numpy as np
from scipy.sparse import coo_matrix
import torch
from pytorch3d import transforms
from torch.utils.data import Dataset

from MDAnalysisData import datasets
import MDAnalysis as mda
from MDAnalysis import transformations
from MDAnalysis.analysis import distances


class MDAnalysisDataset(Dataset):
    """
    NBodyDataset

    """
    def __init__(self, dataset_name, partition='train', tmp_dir=None, delta_frame=1, train_valid_test_ratio=None,
                 test_rot=False, test_trans=False, load_cached=False, cut_off=6, backbone=False):
        super().__init__()
        self.delta_frame = delta_frame
        self.dataset = dataset_name
        self.partition = partition
        self.load_cached = load_cached
        self.test_rot = test_rot
        self.test_trans = test_trans
        self.cut_off = cut_off
        self.backbone = backbone
        if load_cached:
            print(f'Loading {dataset_name} from cached data for {partition}...')
            if backbone:
                tmp_dir = os.path.join(tmp_dir, 'adk_backbone_processed')
            else:
                tmp_dir = os.path.join(tmp_dir, 'adk_processed')
        self.tmp_dir = tmp_dir
        if train_valid_test_ratio is None:
            train_valid_test_ratio = [0.6, 0.2, 0.2]
        assert sum(train_valid_test_ratio) <= 1

        if load_cached:
            edges, self.edge_attr, self.charges, self.n_frames = torch.load(os.path.join(tmp_dir,
                                                                                         f'{dataset_name}.pkl'))
            self.edges = torch.stack(edges, dim=0)
            self.train_valid_test = [int(train_valid_test_ratio[0] * (self.n_frames - delta_frame)),
                                     int(sum(train_valid_test_ratio[:2]) * (self.n_frames - delta_frame))]
            return

        assert not self.backbone, NotImplementedError("Use load_cached for backbone case.")
        if dataset_name.lower() == 'adk':
            adk = datasets.fetch_adk_equilibrium(data_home=tmp_dir)
            self.data = mda.Universe(adk.topology, adk.trajectory)
        else:
            raise NotImplementedError(f'{dataset_name} is not available in MDAnalysisData.')

        # Local Graph information
        try:
            self.charges = torch.tensor(self.data.atoms.charges)
        except OSError:
            print(f'Charge error')
        try:
            self.edges = torch.stack([torch.tensor(self.data.bonds.indices[:, 0], dtype=torch.long),
                                      torch.tensor(self.data.bonds.indices[:, 1], dtype=torch.long)], dim=0)
        except OSError:
            print(f'edges error')
        try:
            self.edge_attr = torch.tensor([bond.length() for bond in self.data.bonds])
        except OSError:
            print(f'edge_attr error')

        self.train_valid_test = [int(train_valid_test_ratio[0] * (len(self.data.trajectory) - delta_frame)),
                                 int(sum(train_valid_test_ratio[:2]) * (len(self.data.trajectory) - delta_frame))]

    def __getitem__(self, i):

        charges, edges, edge_attr = self.charges, self.edges, self.edge_attr
        if len(charges.size()) == 1:
            charges = charges.unsqueeze(-1)
        if len(edge_attr.size()) == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        if self.partition == "valid":
            i = i + self.train_valid_test[0]
        elif self.partition == "test":
            i = i + self.train_valid_test[1]

        # Frames
        frame_0, frame_t = i, i + self.delta_frame

        if self.load_cached:
            loc_0, vel_0, edge_global, edge_global_attr = torch.load(os.path.join(self.tmp_dir,
                                                                                  f'{self.dataset}_{frame_0}.pkl'))
            edge_global = torch.stack(edge_global, dim=0)
            if len(edge_global_attr.size()) == 1:
                edge_global_attr = edge_global_attr.unsqueeze(-1)

            loc_t, vel_t, _, _ = torch.load(os.path.join(self.tmp_dir,
                                                         f'{self.dataset}_{frame_t}.pkl'))
            if self.test_rot and self.partition == 'test':
                rot = transforms.random_rotation()
                loc_0 = torch.tensor(np.matmul(loc_0.detach().numpy(), rot.detach().numpy()))
                vel_0 = torch.tensor(np.matmul(vel_0.detach().numpy(), rot.detach().numpy()))
                loc_t = torch.tensor(np.matmul(loc_t.detach().numpy(), rot.detach().numpy()))
                vel_t = torch.tensor(np.matmul(vel_t.detach().numpy(), rot.detach().numpy()))
            if self.test_trans and self.partition == 'test':
                dimension = loc_t.max(dim=0)[0] - loc_t.min(dim=0)[0]
                trans = torch.randn(3) * dimension / 2
                loc_0 += trans
                loc_t += trans
            return loc_0, vel_0, edge_global, edge_global_attr, edges, edge_attr, charges, loc_t, vel_t

        assert not self.backbone, NotImplementedError("Use load_cached for backbone case.")

        ts_0, ts_t, d, angle, trans = None, None, [0, 0, 1], 0, [0, 0, 0]
        # Initial frame
        retry_0 = 0
        while retry_0 < 10:
            try:
                ts_0 = self.data.trajectory[frame_0].copy()
                if not ts_0.has_velocities:
                    ts_0.velocities = self.data.trajectory[frame_0 + 1].positions - \
                                      self.data.trajectory[frame_0].positions
                retry_0 = 11
            except OSError:
                print(f'Reading error at {frame_0}')
                retry_0 += 1
        assert retry_0 != 10, OSError(f'Falied to read positions by 10 times')

        # Rotations and Translations
        if self.test_rot and self.partition == "test":
            d = np.random.randn(3)
            d = d / np.linalg.norm(d)
            angle = random.randint(0, 360)
            ts_0 = transformations.rotate.rotateby(angle, direction=d, ag=self.data.atoms)(ts_0)
        if self.test_trans and self.partition == 'test':
            trans = np.random.randn(3) * ts_0.dimensions[:3] / 2
            ts_0 = transformations.translate(trans)(ts_0)
        loc_0 = torch.tensor(ts_0.positions)
        vel_0 = torch.tensor(ts_0.velocities)

        # Global edges
        edge_global = coo_matrix(distances.contact_matrix(loc_0.detach().numpy(),
                                                          cutoff=self.cut_off, returntype="sparse"))
        edge_global.setdiag(False)
        edge_global.eliminate_zeros()
        edge_global = torch.stack([torch.tensor(edge_global.row, dtype=torch.long),
                                   torch.tensor(edge_global.col, dtype=torch.long)], dim=0)
        edge_global_attr = torch.norm(loc_0[edge_global[0], :] - loc_0[edge_global[1], :], p=2, dim=1).unsqueeze(-1)

        # Final frames
        retry_t = 0
        while retry_t < 10:
            try:
                ts_t = self.data.trajectory[frame_t].copy()
                if not ts_t.has_velocities:
                    ts_t.velocities = self.data.trajectory[frame_t + 1].positions - \
                                      self.data.trajectory[frame_t].positions
                retry_t = 11
            except OSError:
                print(f'Reading error at {frame_t} t')
                retry_t += 1
        assert retry_t!= 10, OSError(f'Falied to read velocity by 10 times')

        # Rotations and Translations
        if self.test_rot and self.partition == "test":
            ts_t = transformations.rotate.rotateby(angle, direction=d, ag=self.data.atoms)(ts_t)
        if self.test_trans and self.partition == 'test':
            ts_t = transformations.translate(trans)(ts_t)
        loc_t = torch.tensor(ts_t.positions)
        vel_t = torch.tensor(ts_t.velocities)

        return loc_0, vel_0, edge_global, edge_global_attr, edges, edge_attr, charges, loc_t, vel_t

    def __len__(self):
        if self.load_cached:
            total_len = max(0, self.n_frames - self.delta_frame)
        else:
            total_len = max(0, len(self.data.trajectory) - 1 - self.delta_frame)
        if self.partition == 'train':
            return min(total_len, self.train_valid_test[0])
        if self.partition == 'valid':
            return max(0, min(total_len, self.train_valid_test[1]) - self.train_valid_test[0])
        if self.partition == 'test':
            return max(0, total_len - self.train_valid_test[1])

    @staticmethod
    def get_cfg(batch_size, n_nodes, cfg):
        offset = torch.arange(batch_size) * n_nodes
        for type in cfg:
            index = cfg[type]  # [B, n_type, node_per_type]
            cfg[type] = (index + offset.unsqueeze(-1).unsqueeze(-1).expand_as(index)).reshape(-1, index.shape[-1])
            if type == 'Isolated':
                cfg[type] = cfg[type].squeeze(-1)
        return cfg


def collate_mda(data):
    loc_0, vel_0, edge_global, edge_global_attr, edges, edge_attr, charges, loc_t, vel_t = zip(*data)

    # edges
    offset = torch.cumsum(torch.tensor([0] + [_.size(0) for _ in loc_0], dtype=torch.long), dim=0)
    edge_global = torch.cat(list(map(lambda _: _[0] + _[1], zip(edge_global, offset))), dim=-1)
    edges = torch.cat(list(map(lambda _: _[0] + _[1], zip(edges, offset))), dim=-1)
    edge_global_attr = torch.cat(edge_global_attr, dim=0).type(torch.float)
    edge_attr = torch.cat(edge_attr, dim=0).type(torch.float)

    loc_0 = torch.stack(loc_0, dim=0).type(torch.float)
    vel_0 = torch.stack(vel_0, dim=0).view(-1, vel_0[0].size(-1)).type(torch.float)
    loc_t = torch.stack(loc_t, dim=0).view(-1, loc_t[0].size(-1)).type(torch.float)
    vel_t = torch.stack(vel_t, dim=0).view(-1, vel_t[0].size(-1)).type(torch.float)
    charges = torch.stack(charges, dim=0).view(-1, charges[0].size(-1)).type(torch.float)

    return loc_0, vel_0, edge_global, edge_global_attr, edges, edge_attr, charges, loc_t, vel_t


def collate_mdd(data):
    '''
    collate function for MDDynamicsDataset
    '''
    loc_0, vel_0, edge_global, edge_global_attr, edges, edge_attr, charges, loc_t, vel_t = zip(*data)

    # edges
    offset = torch.cumsum(torch.tensor([0] + [_.size(0) for _ in loc_0], dtype=torch.long), dim=0)
    edge_global = torch.cat(list(map(lambda _: _[0] + _[1], zip(edge_global, offset))), dim=-1)
    edges = torch.cat(list(map(lambda _: _[0] + _[1], zip(edges, offset))), dim=-1)
    edge_global_attr = torch.cat(edge_global_attr, dim=0).type(torch.float)
    edge_attr = torch.cat(edge_attr, dim=0).type(torch.float)

    loc_0 = torch.stack(loc_0, dim=0).type(torch.float)
    vel_0 = torch.stack(vel_0, dim=0).view(-1, vel_0[0].size(-1)).type(torch.float)
    loc_t = torch.stack(loc_t, dim=0).type(torch.float)
    vel_t = torch.stack(vel_t, dim=0).type(torch.float)

    B, N, T, _ = loc_t.size()
    loc_t = loc_t.view(B*N, T, 3).transpose(0, 1).contiguous().view(T*B*N, 3)
    vel_t = vel_t.view(B*N, T, 3).transpose(0, 1).contiguous().view(T*B*N, 3)

    charges = torch.stack(charges, dim=0).view(-1, charges[0].size(-1)).type(torch.float)

    return loc_0, vel_0, edge_global, edge_global_attr, edges, edge_attr, charges, loc_t, vel_t


class MDDynamicsDataset(MDAnalysisDataset):

    def __init__(self, dataset_name, partition='train', tmp_dir=None, delta_frame=1, train_valid_test_ratio=None,
                 test_rot=False, test_trans=False, load_cached=False, cut_off=6, backbone=False,
                 num_timesteps=4):
        super().__init__(dataset_name, partition, tmp_dir, delta_frame, train_valid_test_ratio,
                         test_rot, test_trans, load_cached, cut_off, backbone)
        self.num_timesteps = num_timesteps

    def __getitem__(self, i):

        charges, edges, edge_attr = self.charges, self.edges, self.edge_attr
        if len(charges.size()) == 1:
            charges = charges.unsqueeze(-1)
        if len(edge_attr.size()) == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        if self.partition == "valid":
            i = i + self.train_valid_test[0]
        elif self.partition == "test":
            i = i + self.train_valid_test[1]

        # Frames
        # frame_0, frame_t = i, i + self.delta_frame
        frame_0 = i
        assert (self.num_timesteps * (self.delta_frame+1)//self.num_timesteps - 1) == self.delta_frame
        frame_ts = [i + j * (self.delta_frame+1)//self.num_timesteps - 1 for j in range(1, self.num_timesteps + 1)]

        if self.load_cached:
            loc_0, vel_0, edge_global, edge_global_attr = torch.load(os.path.join(self.tmp_dir,
                                                                                  f'{self.dataset}_{frame_0}.pkl'))
            edge_global = torch.stack(edge_global, dim=0)
            if len(edge_global_attr.size()) == 1:
                edge_global_attr = edge_global_attr.unsqueeze(-1)

            # loc_t, vel_t, _, _ = torch.load(os.path.join(self.tmp_dir,
            #                                              f'{self.dataset}_{frame_t}.pkl'))
            loc_t, vel_t = [], []
            for frame_t in frame_ts:
                loc_t_, vel_t_, _, _ = torch.load(os.path.join(self.tmp_dir,
                                                         f'{self.dataset}_{frame_t}.pkl'))
                loc_t.append(loc_t_)
                vel_t.append(vel_t_)
            loc_t = torch.stack(loc_t, dim=1).type(torch.float)
            vel_t = torch.stack(vel_t, dim=1).type(torch.float)
            
            if self.test_rot and self.partition == 'test':
                rot = transforms.random_rotation()
                loc_0 = torch.tensor(np.matmul(loc_0.detach().numpy(), rot.detach().numpy()))
                vel_0 = torch.tensor(np.matmul(vel_0.detach().numpy(), rot.detach().numpy()))
                loc_t = torch.tensor(np.matmul(loc_t.detach().numpy(), rot.detach().numpy()))
                vel_t = torch.tensor(np.matmul(vel_t.detach().numpy(), rot.detach().numpy()))
            if self.test_trans and self.partition == 'test':
                dimension = loc_t.view(-1, 3).max(dim=0)[0] - loc_t.view(-1, 3).min(dim=0)[0]
                trans = torch.randn(3) * dimension / 2
                loc_0 += trans
                loc_t += trans
            return loc_0, vel_0, edge_global, edge_global_attr, edges, edge_attr, charges, loc_t, vel_t
        else:
            raise NotImplementedError("Use load_cached for backbone dynamics case.")
