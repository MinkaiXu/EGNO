import numpy as np
import torch
import pickle as pkl
import os


class MotionDataset():
    """
    Motion Dataset

    """

    def __init__(self, partition, max_samples, delta_frame, data_dir, case='walk'):
        if case == 'walk':
            with open(os.path.join(data_dir, 'motion.pkl'), 'rb') as f:
                edges, X = pkl.load(f)
        elif case == 'run':
            with open(os.path.join(data_dir, 'motion_run.pkl'), 'rb') as f:
                edges, X = pkl.load(f)
        else:
            raise RuntimeError('Unknown case')

        V = []
        for i in range(len(X)):
            V.append(X[i][1:] - X[i][:-1])
            X[i] = X[i][:-1]


        N = X[0].shape[1]

        if case == 'walk':
            train_case_id = [20, 1, 17, 13, 14, 9, 4, 2, 7, 5, 16]
            val_case_id = [3, 8, 11, 12, 15, 18]
            test_case_id = [6, 19, 21, 0, 22, 10]
            split_dir = os.path.join(data_dir, 'split.pkl')
        elif case == 'run':
            train_case_id = [1, 2, 5, 6, 10]
            val_case_id = [0, 4, 9]
            test_case_id = [3, 7, 8]
            split_dir = os.path.join(data_dir, 'split_run.pkl')
        else:
            raise RuntimeError('Unknown case')

        self.partition = partition

        try:
            with open(split_dir, 'rb') as f:
                print('Got Split!')
                split = pkl.load(f)
        except:
            np.random.seed(100)

            # sample 100 for each case
            if case == 'walk':
                itv = 300
            elif case == 'run':
                itv = 90
            else:
                raise RuntimeError('Unknown case')
            train_mapping = {}
            for i in train_case_id:
                # cur_x = X[i][:itv]
                sampled = np.random.choice(np.arange(itv), size=80 if case == 'run' else 100, replace=False)
                train_mapping[i] = sampled
            val_mapping = {}
            for i in val_case_id:
                # cur_x = X[i][:itv]
                sampled = np.random.choice(np.arange(itv), size=80 if case == 'run' else 100, replace=False)
                val_mapping[i] = sampled
            test_mapping = {}
            for i in test_case_id:
                # cur_x = X[i][:itv]
                sampled = np.random.choice(np.arange(itv), size=80 if case == 'run' else 100, replace=False)
                test_mapping[i] = sampled

            with open(split_dir, 'wb') as f:
                pkl.dump((train_mapping, val_mapping, test_mapping), f)

            print('Generate and save split!')
            split = (train_mapping, val_mapping, test_mapping)

        if partition == 'train':
            mapping = split[0]
        elif partition == 'val':
            mapping = split[1]
        elif partition == 'test':
            mapping = split[2]
        else:
            raise NotImplementedError()

        each_len = max_samples // len(mapping)

        x_0, v_0, x_t, v_t = [], [], [], []
        for i in mapping:
            st = mapping[i][:each_len]
            cur_x_0 = X[i][st]
            cur_v_0 = V[i][st]
            cur_x_t = X[i][st + delta_frame]
            cur_v_t = V[i][st + delta_frame]
            x_0.append(cur_x_0)
            v_0.append(cur_v_0)
            x_t.append(cur_x_t)
            v_t.append(cur_v_t)
        x_0 = np.concatenate(x_0, axis=0)
        v_0 = np.concatenate(v_0, axis=0)
        x_t = np.concatenate(x_t, axis=0)
        v_t = np.concatenate(v_t, axis=0)

        print('Got {:d} samples!'.format(x_0.shape[0]))

        self.n_node = N

        atom_edges = torch.zeros(N, N).int()
        for edge in edges:
            atom_edges[edge[0], edge[1]] = 1
            atom_edges[edge[1], edge[0]] = 1

        atom_edges2 = atom_edges @ atom_edges
        self.atom_edge = atom_edges
        self.atom_edge2 = atom_edges2
        edge_attr = []
        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(N):
            for j in range(N):
                if i != j:
                    if self.atom_edge[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([1])
                    elif self.atom_edge2[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([2])
                    else:
                        pass  # TODO: Do we need to add the rest of edges here?

        edges = [rows, cols]  # edges for equivariant message passing
        edge_attr = torch.Tensor(np.array(edge_attr))  # [edge, 3]
        self.edge_attr = edge_attr  # [edge, 3]
        self.edges = torch.LongTensor(np.array(edges))  # [2, edge]

        self.x_0, self.v_0, self.x_t, self.v_t = torch.Tensor(x_0), torch.Tensor(v_0), torch.Tensor(x_t), torch.Tensor(
            v_t)
        mole_idx = np.ones(N)
        self.mole_idx = torch.Tensor(mole_idx)  # the node feature

    def __getitem__(self, i):
        edges = self.edges
        edge_attr = self.edge_attr
        local_edge_mask = edge_attr[..., -1] == 1
        local_edges = edges[..., local_edge_mask]
        local_edge_attr = edge_attr[local_edge_mask]

        # add z to node feature
        node_fea = self.x_0[i][..., 1].unsqueeze(-1) / 10

        return self.x_0[i], self.v_0[i], edges, edge_attr, local_edges, local_edge_attr, \
               node_fea, self.x_t[i], self.v_t[i]

    def __len__(self):
        return len(self.x_0)


class MotionDynamicsDataset(MotionDataset):
    def __init__(self, partition, max_samples, delta_frame, data_dir, case='walk', num_timesteps=6):
        if case == 'walk':
            with open(os.path.join(data_dir, 'motion.pkl'), 'rb') as f:
                edges, X = pkl.load(f)
        elif case == 'run':
            with open(os.path.join(data_dir, 'motion_run.pkl'), 'rb') as f:
                edges, X = pkl.load(f)
        else:
            raise RuntimeError('Unknown case')

        V = []
        for i in range(len(X)):
            V.append(X[i][1:] - X[i][:-1])
            X[i] = X[i][:-1]


        N = X[0].shape[1]

        if case == 'walk':
            train_case_id = [20, 1, 17, 13, 14, 9, 4, 2, 7, 5, 16]
            val_case_id = [3, 8, 11, 12, 15, 18]
            test_case_id = [6, 19, 21, 0, 22, 10]
            split_dir = os.path.join(data_dir, 'split.pkl')
        elif case == 'run':
            train_case_id = [1, 2, 5, 6, 10]
            val_case_id = [0, 4, 9]
            test_case_id = [3, 7, 8]
            split_dir = os.path.join(data_dir, 'split_run.pkl')
        else:
            raise RuntimeError('Unknown case')

        self.partition = partition

        try:
            with open(split_dir, 'rb') as f:
                print('Got Split!')
                split = pkl.load(f)
        except:
            np.random.seed(100)

            # sample 100 for each case
            if case == 'walk':
                itv = 300
            elif case == 'run':
                itv = 90
            else:
                raise RuntimeError('Unknown case')
            train_mapping = {}
            for i in train_case_id:
                # cur_x = X[i][:itv]
                sampled = np.random.choice(np.arange(itv), size=80 if case == 'run' else 100, replace=False)
                train_mapping[i] = sampled
            val_mapping = {}
            for i in val_case_id:
                # cur_x = X[i][:itv]
                sampled = np.random.choice(np.arange(itv), size=80 if case == 'run' else 100, replace=False)
                val_mapping[i] = sampled
            test_mapping = {}
            for i in test_case_id:
                # cur_x = X[i][:itv]
                sampled = np.random.choice(np.arange(itv), size=80 if case == 'run' else 100, replace=False)
                test_mapping[i] = sampled

            with open(split_dir, 'wb') as f:
                pkl.dump((train_mapping, val_mapping, test_mapping), f)

            print('Generate and save split!')
            split = (train_mapping, val_mapping, test_mapping)

        if partition == 'train':
            mapping = split[0]
        elif partition == 'val':
            mapping = split[1]
        elif partition == 'test':
            mapping = split[2]
        else:
            raise NotImplementedError()

        each_len = max_samples // len(mapping)

        x_0, v_0, x_t, v_t = [], [], [], []
        last = True
        for i in mapping:
            st = mapping[i][:each_len]
            cur_x_0 = X[i][st]
            cur_v_0 = V[i][st]
            if last:
                cur_x_t = [X[i][st + delta_frame + ii - num_timesteps] for ii in range(1, num_timesteps + 1)]
            else:
                cur_x_t = [X[i][st + delta_frame * ii // num_timesteps] for ii in range(1, num_timesteps + 1)]
            # cur_x_t = [X[i][st + delta_frame] for ii in range(1, num_timesteps + 1)]  # here for repeat target only
            cur_x_t = np.stack(cur_x_t, axis=2)
            # cur_x_t = X[i][st + delta_frame]
            if last:
                cur_v_t = [V[i][st + delta_frame + ii - num_timesteps] for ii in range(1, num_timesteps + 1)]
            else:
                cur_v_t = [V[i][st + delta_frame * ii // num_timesteps] for ii in range(1, num_timesteps + 1)]
            cur_v_t = np.stack(cur_v_t, axis=2)
            # cur_v_t = V[i][st + delta_frame]
            x_0.append(cur_x_0)
            v_0.append(cur_v_0)
            x_t.append(cur_x_t)
            v_t.append(cur_v_t)
        x_0 = np.concatenate(x_0, axis=0)
        v_0 = np.concatenate(v_0, axis=0)
        x_t = np.concatenate(x_t, axis=0)
        v_t = np.concatenate(v_t, axis=0)

        print('Got {:d} samples!'.format(x_0.shape[0]))

        self.n_node = N

        atom_edges = torch.zeros(N, N).int()
        for edge in edges:
            atom_edges[edge[0], edge[1]] = 1
            atom_edges[edge[1], edge[0]] = 1

        atom_edges2 = atom_edges @ atom_edges
        self.atom_edge = atom_edges
        self.atom_edge2 = atom_edges2
        edge_attr = []
        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(N):
            for j in range(N):
                if i != j:
                    if self.atom_edge[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([1])
                    elif self.atom_edge2[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([2])
                    else:
                        pass  # TODO: Do we need to add the rest of edges here?

        edges = [rows, cols]  # edges for equivariant message passing
        edge_attr = torch.Tensor(np.array(edge_attr))  # [edge, 3]
        self.edge_attr = edge_attr  # [edge, 3]
        self.edges = torch.LongTensor(np.array(edges))  # [2, edge]

        self.x_0, self.v_0, self.x_t, self.v_t = torch.Tensor(x_0), torch.Tensor(v_0), torch.Tensor(x_t), torch.Tensor(
            v_t)
        mole_idx = np.ones(N)
        self.mole_idx = torch.Tensor(mole_idx)  # the node feature


if __name__ == '__main__':
    data = MotionDynamicsDataset(partition='train', max_samples=200, delta_frame=30, data_dir='./dataset',
                                 num_timesteps=6)
    print(data[10][-1].shape)
