import argparse
from argparse import Namespace
import torch
import torch.utils.data
from md17.dataset import MD17DynamicsDataset as MD17Dataset
from model.egno import EGNO
import os, sys
from torch import nn, optim
import json
from torch.optim.lr_scheduler import StepLR

import random
import numpy as np

parser = argparse.ArgumentParser(description='EGNO')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--num_timesteps', type=int, default=8, metavar='N',
                    help='number of time steps per sample')
parser.add_argument('--use_time_conv', type=eval, default=False)

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='log/md17', metavar='N',
                    help='folder to output the json log file')
parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=64, metavar='N',
                    help='hidden dim')
parser.add_argument('--model', type=str, default='egno', metavar='N',
                    help='available models: egno')
parser.add_argument('--attention', type=int, default=0, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--n_layers', type=int, default=5, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N',
                    help='weight decay')
parser.add_argument('--norm_diff', type=eval, default=False, metavar='N',
                    help='normalize_diff')
parser.add_argument('--tanh', type=eval, default=False, metavar='N',
                    help='use tanh')
parser.add_argument('--delta_frame', type=int, default=50,
                    help='Number of frames delta.')
parser.add_argument('--mol', type=str, default='aspirin',
                    help='Name of the molecule.')
parser.add_argument('--data_dir', type=str, default='',
                    help='Data directory.')
parser.add_argument('--learnable', type=eval, default=False, metavar='N',
                    help='Use learnable FK.')

parser.add_argument("--config_by_file", default=False, action="store_true", )
parser.add_argument("--config", default='config_md17_no.json', 
                    type=str, help='Path to the config file.')


args = parser.parse_args()
if args.config_by_file:
    job_param_path = 'configs/' + args.config
    with open(job_param_path, 'r') as f:
        hyper_params = json.load(f)
        # update keys existing in config
        args = vars(args)
        args.update((k, v) for k, v in hyper_params.items() if k in args)
        args = Namespace(**args)

assert torch.cuda.is_available(), "no cuda device available"
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")
loss_mse = nn.MSELoss(reduction='none')

try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs(args.outf + "/" + args.exp_name)
except OSError:
    pass


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass


sys.stdout = Logger(args.outf + "/" + args.exp_name + "/log.txt")

print(args)
# torch.autograd.set_detect_anomaly(True)


def get_velocity_attr(loc, vel, rows, cols):

    diff = loc[cols] - loc[rows]
    norm = torch.norm(diff, p=2, dim=1).unsqueeze(1)
    u = diff/norm
    va, vb = vel[rows] * u, vel[cols] * u
    va, vb = torch.sum(va, dim=1).unsqueeze(1), torch.sum(vb, dim=1).unsqueeze(1)
    return va


def main():
    # fix seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset_train = MD17Dataset(partition='train', max_samples=args.max_training_samples, data_dir=args.data_dir,
                                molecule_type=args.mol, delta_frame=args.delta_frame,
                                num_timesteps=args.num_timesteps)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                               num_workers=0)

    dataset_val = MD17Dataset(partition='val', max_samples=2000, data_dir=args.data_dir,
                                molecule_type=args.mol, delta_frame=args.delta_frame,
                                num_timesteps=args.num_timesteps)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                             num_workers=0)

    dataset_test = MD17Dataset(partition='test', max_samples=2000, data_dir=args.data_dir,
                                molecule_type=args.mol, delta_frame=args.delta_frame,
                                num_timesteps=args.num_timesteps)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                              num_workers=0)

    if args.model == 'egno':
        model = EGNO(in_node_nf=2, in_edge_nf=2 + 3, hidden_nf=args.nf, device=device, n_layers=args.n_layers,
                    with_v=True, flat=False, activation=nn.SiLU(),
                    use_time_conv=args.use_time_conv, num_modes=2, num_timesteps=args.num_timesteps)
    else:
        raise Exception("Wrong model specified")

    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=2500, gamma=0.5)

    results = {'epochs': [], 'loss': [], 'train loss': []}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best_train_loss = 1e8
    for epoch in range(args.epochs):
        train_loss = train(model, optimizer, epoch, loader_train)
        results['train loss'].append(train_loss)
        if epoch % args.test_interval == 0:
            val_loss = train(model, optimizer, epoch, loader_val, backprop=False)
            test_loss = train(model, optimizer, epoch, loader_test, backprop=False)
            results['epochs'].append(epoch)
            results['loss'].append(test_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_train_loss = train_loss
                best_epoch = epoch
                # torch.save(model.state_dict(), args.outf + '/' + 'saved_model.pth')
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best apoch %d"
                  % (best_val_loss, best_test_loss, best_epoch))
        scheduler.step()

        json_object = json.dumps(results, indent=4)
        with open(args.outf + "/" + args.exp_name + "/loss.json", "w") as outfile:
            outfile.write(json_object)
    return best_train_loss, best_val_loss, best_test_loss, best_epoch


def train(model, optimizer, epoch, loader, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'counter': 0}

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, _ = data[0].size()
        data, cfg = data[:-1], data[-1]
        data = [d.to(device) for d in data]
        data = [d.view(-1, d.size(-1)) for d in data]  # construct mini-batch graphs
        for i in [4, 5]:
            d = data[i].view(batch_size * n_nodes, args.num_timesteps, 3)
            data[i] = d.transpose(0, 1).contiguous().view(-1, 3)
        loc, vel, edge_attr, charges, loc_end, vel_end, Z = data

        edges = loader.dataset.get_edges(batch_size, n_nodes)
        edges = [edges[0].to(device), edges[1].to(device)]

        cfg = loader.dataset.get_cfg(batch_size, n_nodes, cfg)
        cfg = {_: cfg[_].to(device) for _ in cfg}

        optimizer.zero_grad()

        if args.model == 'egno':
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            nodes = torch.cat((nodes, Z / Z.max()), dim=-1)
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
            loc_mean = loc.view(batch_size, n_nodes, 3).mean(dim=1, keepdim=True).repeat(1, n_nodes, 1).view(-1, 3)  # [BN, 3]
            loc_pred, vel_pred, _ = model(loc.detach(), nodes, edges, edge_attr, vel, loc_mean=loc_mean)
        else:
            raise Exception("Wrong model")

        losses = loss_mse(loc_pred, loc_end).view(args.num_timesteps, batch_size * n_nodes, 3)
        losses = torch.mean(losses, dim=(1, 2))
        loss = torch.mean(losses)

        if backprop:
            loss.backward()
            optimizer.step()
        res['loss'] += losses[-1].item()*batch_size
        res['counter'] += batch_size

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f' % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter']))

    return res['loss'] / res['counter']


if __name__ == "__main__":
    best_train_loss, best_val_loss, best_test_loss, best_epoch = main()
    print("best_train = %.6f" % best_train_loss)
    print("best_val = %.6f" % best_val_loss)
    print("best_test = %.6f" % best_test_loss)
    print("best_epoch = %d" % best_epoch)
