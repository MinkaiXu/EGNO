import argparse
from argparse import Namespace
import torch
from tqdm import tqdm
import torch.utils.data
from mdanalysis.dataset import collate_mdd as collate_mda
from mdanalysis.dataset import MDDynamicsDataset as MDAnalysisDataset
from model.egno import EGNO
import os
from torch import nn, optim
import json
import  time
import random
import numpy as np

from utils import EarlyStopping

parser = argparse.ArgumentParser(description='EGNO')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='exp_results', metavar='N',
                    help='folder to output the json log file')
parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=64, metavar='N',
                    help='hidden dim')
parser.add_argument('--model', type=str, default='egno', metavar='N',
                    help='available models: egno')
parser.add_argument('--attention', type=int, default=0, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--dataset', type=str, default="nbody_small", metavar='N',
                    help='nbody_small, nbody')
parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N',
                    help='timing experiment')
parser.add_argument('--delta_frame', type=int, default=50,
                    help='Number of frames delta.')
parser.add_argument('--data_dir', type=str, default='mdanalysis/dataset/',
                    help='Data directory.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument("--backbone", action="store_true",
                    help="Load backbone data of protein")

parser.add_argument('--num_modes', type=int, default=2, help='The number of modes.')
parser.add_argument('--num_timesteps', type=int, default=4, help='The number of timesteps.')

parser.add_argument('--lambda_link', type=float, default=1, help='The weight of the linkage loss.')
parser.add_argument('--interaction_layer', type=int, default=3, help='The number of interaction layers per block.')
parser.add_argument('--pooling_layer', type=int, default=3, help='The number of pooling layers in EGPN.')
parser.add_argument('--decoder_layer', type=int, default=1, help='The number of decoder layers.')
parser.add_argument('--n_cluster', type=int, default=20, help='The number of clusters.')
parser.add_argument('--flat', action='store_true', default=False, help='flat MLP')
parser.add_argument("--config_by_file", default=None, nargs='?', const='')
parser.add_argument("--n_workers", '-n', type=int, default=8, help="Number of workers.")
parser.add_argument("--load_cached", action="store_true", help="Load cached dataset.")
parser.add_argument("--test_rot", action="store_true", help="Rotate the test")
parser.add_argument("--test_trans", action="store_true", help="Translate the test")

time_exp_dic = {'time': 0, 'counter': 0}


args = parser.parse_args()
if args.config_by_file is not None:
    if len(args.config_by_file) == 0:
        job_param_path = './configs/config_mdanalysis_no.json'
    else:
        job_param_path = args.config_by_file
    with open(job_param_path, 'r') as f:
        hyper_params = json.load(f)
        # Only update existing keys
        args = vars(args)
        args.update((k, v) for k, v in hyper_params.items() if k in args)
        args = Namespace(**args)

args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

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

# torch.autograd.set_detect_anomaly(True)

def main():
    # fix seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset_train = MDAnalysisDataset('adk', partition='train', tmp_dir=args.data_dir,
                                      delta_frame=args.delta_frame, load_cached=args.load_cached, 
                                      backbone=args.backbone, num_timesteps=args.num_timesteps)
    sampler = None
    shuffle = True

    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                               shuffle=shuffle, sampler=sampler, drop_last=True,
                                               num_workers=args.n_workers, collate_fn=collate_mda)

    dataset_val = MDAnalysisDataset('adk', partition='valid', tmp_dir=args.data_dir,
                                    delta_frame=args.delta_frame, load_cached=args.load_cached, 
                                    backbone=args.backbone, num_timesteps=args.num_timesteps)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                             drop_last=False, num_workers=args.n_workers, collate_fn=collate_mda)

    # Val and test do not need sampler.
    dataset_test = MDAnalysisDataset('adk', partition='test', tmp_dir=args.data_dir,
                                     delta_frame=args.delta_frame, load_cached=args.load_cached,
                                     test_rot=False, test_trans=False, backbone=args.backbone,
                                     num_timesteps=args.num_timesteps)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                              shuffle=False,  drop_last=False,
                                              num_workers=args.n_workers, collate_fn=collate_mda)

    dataset_test_hard = MDAnalysisDataset('adk', partition='test', tmp_dir=args.data_dir,
                                          delta_frame=args.delta_frame, load_cached=args.load_cached,
                                          test_rot=True, test_trans=True, backbone=args.backbone,
                                          num_timesteps=args.num_timesteps)
    loader_test_hard = torch.utils.data.DataLoader(dataset_test_hard, batch_size=args.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=args.n_workers, collate_fn=collate_mda)

    if args.load_cached:
        print("Data loading finished.")

    if args.model == 'egno':
        model = EGNO(n_layers=args.n_layers, in_node_nf=2, in_edge_nf=2, hidden_nf=args.nf, device=device, with_v=True,
                     flat=args.flat, activation=nn.SiLU(),
                     use_time_conv=True, num_modes=args.num_modes, num_timesteps=args.num_timesteps)
    else:
        raise NotImplementedError('Unknown model:', args.model)

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model_save_path = os.path.join(args.outf, args.exp_name, 'saved_model.pth')
    early_stopping = EarlyStopping(patience=50, verbose=True, path=model_save_path)

    results = {'eval epoch': [], 'val loss': [], 'test loss': [], 'train loss': [], 'test loss hard':[]}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_test_loss_hard = 1e8
    best_epoch = 0
    best_train_loss = 1e8
    best_lp_loss = 1e8
    for epoch in range(0, args.epochs):
        train_loss, lp_loss = train(model, optimizer, epoch, loader_train)
        results['train loss'].append(train_loss)
        if epoch % args.test_interval == 0:
            # every worker need evaluate this part!
            val_loss, _ = train(model, optimizer, epoch, loader_val, backprop=False)
            test_loss, _ = train(model, optimizer, epoch, loader_test, backprop=False)
            if args.backbone:
                test_loss_hard, _ = train(model, optimizer, epoch, loader_test_hard, backprop=False)
            else:
                test_loss_hard = 0

            results['eval epoch'].append(epoch)
            results['val loss'].append(val_loss)
            results['test loss'].append(test_loss)
            results['test loss hard'].append(test_loss_hard)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_train_loss = train_loss
                best_test_loss_hard = test_loss_hard
                best_epoch = epoch
                best_lp_loss = lp_loss
                # Save model is move to early stopping.
                print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best Hard Test Loss: %.5f \t Best epoch %d"
                    % (best_val_loss, best_test_loss, best_test_loss_hard, best_epoch))
            # only master worker will store the model.
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                # This state is consistent for all workers.
                print("Early Stopping.")
                break

        json_object = json.dumps(results, indent=4)
        with open(args.outf + "/" + args.exp_name + "/loss.json", "w") as outfile:
            outfile.write(json_object)
    return best_train_loss, best_lp_loss, best_val_loss, best_test_loss, best_test_loss_hard, best_epoch


def train(model, optimizer, epoch, loader, backprop=True):
    s = time.time()
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'counter': 0, 'lp_loss': 0}

    #tqdm_loader = tqdm(loader, desc=f'Epoch {epoch}')
    for batch_idx, data in enumerate(tqdm(loader, desc=f'Epoch {epoch}')):
        batch_size, n_nodes, _ = data[0].size()
        data = [d.to(device) for d in data]
        loc, vel, edges, edge_attr, local_edge_index, local_edge_fea, Z, loc_end, vel_end = data
        # convert into graph minibatch
        loc_mean = loc.mean(dim=1, keepdim=True).repeat(1, n_nodes, 1).view(-1, loc.size(2))  # [BN, 3]
        loc = loc.view(-1, loc.size(2))

        optimizer.zero_grad()

        if args.model == 'egno':
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            nodes = torch.cat((nodes, Z / Z.max()), dim=-1)
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
            loc_pred, vel_pred, _ = model(loc, nodes, edges, edge_attr, v=vel, loc_mean=loc_mean)
        else:
            raise Exception("Wrong model")

        losses = loss_mse(loc_pred, loc_end).view(args.num_timesteps, batch_size * n_nodes, -1)
        losses = torch.mean(losses, dim=(1, 2))
        loss = torch.mean(losses)

        if backprop:
            loss.backward()
            optimizer.step()
        res['loss'] += losses[-1].item()*batch_size
        res['counter'] += batch_size

    if not backprop:
        prefix = "==> "
        time_prefix = "val time"
    else:
        prefix = ""
        time_prefix = "traning time"
    print('%s epoch %d avg loss: %.5f avg lploss: %.5f, %s: %.5f'
          % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter'], res['lp_loss'] / res['counter'], time_prefix, time.time() - s))

    return res['loss'] / res['counter'],  res['lp_loss'] / res['counter']


if __name__ == "__main__":
    best_train_loss, best_lp_loss, best_val_loss, best_test_loss, best_test_loss_hard, best_epoch = main()
    print("best_train = %.6f" % best_train_loss)
    print("best_lp = %.6f" % best_lp_loss)
    print("best_val = %.6f" % best_val_loss)
    print("best_test = %.6f" % best_test_loss)
    print("best_test_hard = %.6f" % best_test_loss)
    print("best_epoch = %d" % best_epoch)
    print("best_train = %.6f, best_lp = %.6f, best_val = %.6f, best_test = %.6f, best_test_hard = %.6f, best_epoch = %d"
            % (best_train_loss, best_lp_loss, best_val_loss, best_test_loss, best_test_loss_hard, best_epoch))
    print("model saved at %s" % args.outf)

