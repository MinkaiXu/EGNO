import torch
import numpy as np
from torch import nn
from motion.dataset import MotionDataset


def collector(batch):
    """
    Rebatch the input and padding zeros for loc, vel, loc_end, vel_end.
    Add the additional mask (B*N, 1) at the last.
    :param batch:
    :return: the re_batched list.
    """
    re_batch = [[] for _ in range(len(batch[0]))]
    for b in batch:
        [re_batch[i].append(d) for i, d in enumerate(b)]

    loc, vel, edge_attr, charges, loc_end, vel_end = re_batch[:6]
    res = []
    padding = [True, True, False, False, True, True, False, False, False]
    for b, p in zip(re_batch[:-1], padding[:len(re_batch) -1]):
        res.append(do_padding(b, padding=p))
    mask = generate_mask(loc)
    res.append(re_batch[-1])
    res.append(mask)
    return res


def collector_simulation(batch):
    """
    Rebatch the input and padding zeros for loc, vel, loc_end, vel_end.
    Add the additional mask (B*N, 1) at the last.
    :param batch:
    :return: the re_batched list.
    """
    re_batch = [[] for _ in range(len(batch[0]))]
    for b in batch:
        [re_batch[i].append(d) for i, d in enumerate(b)]

    assert len(re_batch) == 8
    loc, vel, edges, edge_attr, local_edge_mask, charges, loc_end, vel_end = re_batch
    max_size = max([x.size(0) for x in loc])
    node_nums = torch.tensor([x.size(0) for x in loc])
    mask = generate_mask(loc)
    loc = _padding(loc, max_size)
    vel = _padding(vel, max_size)
    edges = _pack_edges(edges, max_size)
    edge_attr = torch.cat(edge_attr, dim=0)
    local_edge_mask = torch.cat(local_edge_mask, dim=0)
    charges = _padding(charges, max_size)
    loc_end = _padding(loc_end, max_size)
    vel_end = _padding(vel_end, max_size)
    return loc, vel, edges, edge_attr, local_edge_mask, charges, loc_end, vel_end, mask, node_nums, max_size


def collector_simulation_no(batch):
    """
    Rebatch the input and padding zeros for loc, vel, loc_end, vel_end.
    Add the additional mask (B*N, 1) at the last.
    :param batch:
    :return: the re_batched list.
    """
    re_batch = [[] for _ in range(len(batch[0]))]
    for b in batch:
        [re_batch[i].append(d) for i, d in enumerate(b)]

    assert len(re_batch) == 8
    loc, vel, edges, edge_attr, local_edge_mask, charges, loc_end, vel_end = re_batch
    max_size = max([x.size(0) for x in loc])
    node_nums = torch.tensor([x.size(0) for x in loc])
    mask = generate_mask(loc)
    loc = _padding(loc, max_size)
    vel = _padding(vel, max_size)
    edges = _pack_edges(edges, max_size)
    edge_attr = torch.cat(edge_attr, dim=0)
    local_edge_mask = torch.cat(local_edge_mask, dim=0)
    charges = _padding(charges, max_size)
    loc_end = _padding_3(loc_end, max_size)
    vel_end = _padding_3(vel_end, max_size)
    return loc, vel, edges, edge_attr, local_edge_mask, charges, loc_end, vel_end, mask, node_nums, max_size


def _padding(tensor_list, max_size):
    res = [torch.cat([r, torch.zeros([max_size - r.size(0), r.size(1)])]) for r in tensor_list]
    res = torch.cat(res, dim=0)
    return res


def _padding_3(tensor_list, max_size):
    res = [torch.cat([r, torch.zeros([max_size - r.size(0), r.size(1), r.size(2)])]) for r in tensor_list]
    res = torch.cat(res, dim=0)
    return res


def _pack_edges(edge_list, n_node):
    for idx, edge in enumerate(edge_list):
        edge[0] += idx * n_node
        edge[1] += idx * n_node
    return torch.cat(edge_list, dim=1)  # [2, BM]


def do_padding(tensor_list, padding=True):
    """
    Pad the input tensor_list ad
    :param tensor_list: list(B, tensor[N, *])
    :return: padded tensor [B*max_N, *]
    """
    if padding:
        max_size = max([x.size(0) for x in tensor_list])
        res = [torch.cat([r, torch.zeros([max_size - r.size(0), r.size(1)])]) for r in tensor_list]
    else:
        res = tensor_list
    res = torch.cat(res, dim=0)
    return res


def generate_mask(tensor_list):
    max_size = max([x.size(0) for x in tensor_list])
    res = [torch.cat([torch.ones([r.size(0)]), torch.zeros([max_size - r.size(0)])]) for r in tensor_list]
    res = torch.cat(res, dim=0)
    return res


def test_do_padding():
    tensor_list = [torch.ones([2, 3]), torch.zeros([4, 3])]
    res = do_padding(tensor_list)

    # tensor([[1., 1., 1.],
    #         [1., 1., 1.],
    #         [0., 0., 0.],
    #         [0., 0., 0.],
    #         [0., 0., 0.],
    #         [0., 0., 0.],
    #         [0., 0., 0.],
    #         [0., 0., 0.]])


def test_generate_mask():
    tensor_list = [torch.rand([2, 3]), torch.rand([4, 3])]
    res = generate_mask(tensor_list)
    print(res)


def test_collector():
    data_train = MotionDataset(partition='train', max_samples=100, delta_frame=30, data_dir='motion/dataset')
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=2, shuffle=True, drop_last=True,
                                               num_workers=1, collate_fn=collector)
    for batch_idx, data in enumerate(loader_train):
        print(data)


class MaskMSELoss(nn.Module):
    def __init__(self):
        super(MaskMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction="none")

    def forward(self, pred, target, mask, grouped_size=None):
        """

        :param pred: [N, d]
        :param target: [N, d]
        :param mask: [N, 1]
        :param grouped_size: [B, K], B * K = N
        :return:
        """
        assert grouped_size is None or (mask.size(0) % grouped_size.size(0) == 0)
        loss = self.loss(pred, target)
        # Looks strange, do I miss something?
        loss = (loss.T * mask).T
        if grouped_size is not None:
            loss = loss.reshape([grouped_size.size(0), -1, pred.size(-1)])
            # average loss by grouped_size on dim=1
            loss = torch.sum(loss, dim=1) / grouped_size.unsqueeze(dim=1)
            loss = torch.mean(loss)
        else:
            loss = torch.sum(loss) / (torch.sum(mask) * loss.size(-1))
        return loss


def test_MaskMSELoss():
    input = torch.rand([6, 2])
    target = torch.rand([6, 2])
    mask = torch.tensor([1, 0, 1, 0, 1, 1])
    grouped_size = torch.tensor([1, 1, 2])
    loss = MaskMSELoss()
    print(loss(input, target, mask, grouped_size))
    print(loss(input, target, mask))


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model, master_worker=True):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, master_worker)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, master_worker)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, master_worker=True):
        '''Saves model when validation loss decrease.'''
        if not master_worker:
            return
        if self.verbose and master_worker:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss