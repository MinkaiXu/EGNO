import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix

from MDAnalysisData import datasets
import MDAnalysis
from MDAnalysis.analysis import distances


def compute_ele(ts, index, cutoff):
    edge = coo_matrix(distances.contact_matrix(ts.positions[index], cutoff=cutoff, returntype="sparse"))
    edge.setdiag(False)
    edge.eliminate_zeros()
    edge_global = [torch.tensor(edge.row, dtype=torch.long), torch.tensor(edge.col, dtype=torch.long)]
    global_edge_attr = torch.norm(torch.tensor(ts.positions[index[edge.row], :] - ts.positions[index[edge.col], :]),
                                  p=2, dim=1)
    return edge_global, global_edge_attr


# delta_frame = 50
backbone = True
tmp_dir = 'mdanalysis/dataset/'
cut_off = 8
# train_valid_test_ratio = [0.6, 0.2, 0.2]
is_save = True

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='mdanalysis/dataset/')
parser.add_argument('--top_file', type=str, default=None,
                    help="topology file name 'in' the directory")
parser.add_argument('--traj_file', type=str, default=None,
                    help="trajectory file name 'in' the directory")
args = parser.parse_args()

tmp_dir = args.dir

if args.top_file is not None and args.traj_file is not None:
    top_path = os.path.join(args.dir, args.top_file)
    traj_path = os.path.join(args.dir, args.traj_file)
    data = MDAnalysis.Universe(top_path, traj_path)
else:
    print("Warning: No topology or trajectory file given. Using default adk dataset.")
    adk = datasets.fetch_adk_equilibrium(data_home=tmp_dir)
    data = MDAnalysis.Universe(adk.topology, adk.trajectory)
if backbone:
    ag = data.select_atoms('backbone')
else:
    ag = data.atoms

# train_valid_test = [int(train_valid_test_ratio[0] * len(data.trajectory)),
#                     int(sum(train_valid_test_ratio[:2]) * len(data.trajectory))]


charges = torch.tensor(data.atoms[ag.ix].charges)
bonds = np.stack([bond.indices for bond in data.bonds if bond.indices[0] in ag.ix and bond.indices[1] in ag.ix])
map_dict = {v:k for k,v in enumerate(ag.ix)}
bonds = np.vectorize(map_dict.get)(bonds)
edges = [torch.tensor(bonds[:, 0], dtype=torch.long),
         torch.tensor(bonds[:, 1], dtype=torch.long)]

edge_attr = torch.tensor([bond.length() for bond in data.bonds
                          if bond.indices[0] in ag.ix and bond.indices[1] in ag.ix])

loc = []
vel = []

for i in tqdm(range(len(data.trajectory) - 1)):
    loc.append(torch.tensor(data.trajectory[i].positions[ag.ix]))
    vel.append(torch.tensor(data.trajectory[i + 1].positions[ag.ix] - data.trajectory[i].positions[ag.ix]))

if backbone:
    save_path = os.path.join(tmp_dir, 'adk_backbone_processed', 'adk.pkl')
    os.makedirs(os.path.join(tmp_dir, 'adk_backbone_processed'), exist_ok=True)
else:
    save_path = os.path.join(tmp_dir, 'adk_processed', 'adk.pkl')
    os.makedirs(os.path.join(tmp_dir, 'adk_processed'), exist_ok=True)
if is_save:
    torch.save((edges, edge_attr, charges, len(data.trajectory) - 1), save_path)

edges_global, edges_global_attr = zip(*Parallel(n_jobs=-1)(delayed(lambda a: compute_ele(a, ag.ix, cut_off))(_)
                                                           for _ in tqdm(data.trajectory)))
edges_global = edges_global[:-1]
edges_global_attr = edges_global_attr[:-1]
    #
    # edge = coo_matrix(distances.contact_matrix(data.trajectory[i].positions, cutoff=cut_off, returntype="sparse"))
    # edge.setdiag(False)
    # edge.eliminate_zeros()
    # edge_global.append([torch.tensor(edge.row, dtype=torch.long), torch.tensor(edge.col, dtype=torch.long)])
    # global_edge_attr.append(torch.norm(torch.tensor(data.trajectory[0].positions[edge.row, :] -
    #                                                 data.trajectory[0].positions[edge.col, :]), p=2, dim=1))


# loc = torch.stack(loc, dim=0)
# vel = torch.stack(vel, dim=0)

if backbone:
    save_path = os.path.join(tmp_dir, 'adk_backbone_processed')
else:
    save_path = os.path.join(tmp_dir, 'adk_processed')

if is_save:
    for i in tqdm(range(len(loc))):
        try:
            torch.save((loc[i], vel[i], edges_global[i], edges_global_attr[i]),
                       os.path.join(save_path, f'adk_{i}.pkl'))
        except RuntimeError:
            print(i)

# torch.save((loc[:train_valid_test[0] + delta_frame], vel[:train_valid_test[0] + delta_frame],
#             edges_global[:train_valid_test[0]], edges_global_attr[:train_valid_test[0]],
#             edges, edge_attr, charges), os.path.join(tmp_dir, 'adk_train.pkl'))
#
# torch.save((loc[train_valid_test[0]:train_valid_test[1] + delta_frame],
#             vel[train_valid_test[0]:train_valid_test[1] + delta_frame],
#             edges_global[train_valid_test[0]:train_valid_test[1]],
#             edges_global_attr[train_valid_test[0]:train_valid_test[1]],
#             edges, edge_attr, charges), os.path.join(tmp_dir, 'adk_valid.pkl'))
#
# torch.save((loc[train_valid_test[1]:], vel[train_valid_test[1]:], edges_global[train_valid_test[1]:],
#             edges_global_attr[train_valid_test[1]:], edges, edge_attr, charges), os.path.join(tmp_dir, 'adk_test.pkl'))