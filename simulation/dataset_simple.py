import numpy as np
import torch
import os.path as osp


class NBodyDataset():
    """
    NBodyDataset

    """

    def __init__(self, partition='train', data_dir='.', max_samples=1e8, dataset_name="nbody_small"):
        self.partition = partition
        self.data_dir = data_dir
        if self.partition == 'val':
            self.suffix = 'valid'
        else:
            self.suffix = self.partition
        self.dataset_name = dataset_name
        if dataset_name == "nbody":
            self.suffix += "_charged5_initvel1"
        elif dataset_name == "nbody_small" or dataset_name == "nbody_small_out_dist":
            self.suffix += "_charged5_initvel1small"
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)

        self.max_samples = int(max_samples)
        self.dataset_name = dataset_name
        self.data, self.edges = self.load()

    def load(self):
        dir = self.data_dir
        loc = np.load(osp.join(dir, 'simple', 'loc_' + self.suffix + '.npy'))
        vel = np.load(osp.join(dir, 'simple', 'vel_' + self.suffix + '.npy'))
        edges = np.load(osp.join(dir, 'simple', 'edges_' + self.suffix + '.npy'))
        charges = np.load(osp.join(dir, 'simple', 'charges_' + self.suffix + '.npy'))

        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)
        return (loc, vel, edge_attr, charges), edges

    def preprocess(self, loc, vel, edges, charges):
        # cast to torch and swap n_nodes <--> n_features dimensions
        loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        n_nodes = loc.size(2)
        loc = loc[0:self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0:self.max_samples, :, :, :]  # speed when starting the trajectory
        charges = charges[0:self.max_samples]
        edge_attr = []

        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(
            2)  # swap n_nodes <--> batch_size and add nf dimension

        return torch.Tensor(loc), torch.Tensor(vel), torch.Tensor(edge_attr), edges, torch.Tensor(charges)

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()

    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges = loc[i], vel[i], edge_attr[i], charges[i]

        if self.dataset_name == "nbody":
            frame_0, frame_T = 6, 8
        elif self.dataset_name == "nbody_small":
            frame_0, frame_T = 30, 40
        elif self.dataset_name == "nbody_small_out_dist":
            frame_0, frame_T = 20, 30
        else:
            raise Exception("Wrong dataset partition %s" % self.dataset_name)

        return loc[frame_0], vel[frame_0], edge_attr, charges, loc[frame_T]

    def __len__(self):
        return len(self.data[0])

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges


class NBodyDynamicsDataset(NBodyDataset):
    def __init__(self, partition='train', data_dir='.', max_samples=1e8, dataset_name="nbody_small", num_timesteps=1):
        self.num_timesteps = num_timesteps
        super(NBodyDynamicsDataset, self).__init__(partition, data_dir, max_samples, dataset_name)

    def __getitem__(self, i):
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges = loc[i], vel[i], edge_attr[i], charges[i]

        if self.dataset_name == "nbody":
            frame_0, frame_T = 6, 8
        elif self.dataset_name == "nbody_small":
            frame_0, frame_T = 30, 40
        elif self.dataset_name == "nbody_small_out_dist":
            frame_0, frame_T = 20, 30
        else:
            raise Exception("Wrong dataset partition %s" % self.dataset_name)

        delta_frame = frame_T - frame_0
        last = False
        if last:
            locs = [loc[frame_0 + delta_frame + ii - self.num_timesteps] for ii in range(1, self.num_timesteps + 1)]
        else:
            locs = [loc[frame_0 + delta_frame * ii // self.num_timesteps] for ii in range(1, self.num_timesteps + 1)]
        locs = np.stack(locs, axis=1)
        if last:
            vels = [vel[frame_0 + delta_frame + ii - self.num_timesteps] for ii in range(1, self.num_timesteps + 1)]
        else:
            vels = [vel[frame_0 + delta_frame * ii // self.num_timesteps] for ii in range(1, self.num_timesteps + 1)]
        vels = np.stack(vels, axis=1)

        return loc[frame_0], vel[frame_0], edge_attr, charges, locs


if __name__ == "__main__":
    dataset = NBodyDynamicsDataset('train', data_dir='./dataset', max_samples=3000, num_timesteps=6)
    for i in dataset[100]:
        print(i.shape)

