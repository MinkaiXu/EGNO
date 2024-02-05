from torch import nn
import torch
import torch.nn.functional as F


def aggregate(message, row_index, n_node, aggr='sum', mask=None):
    """
    The aggregation function (aggregate edge messages towards nodes)
    :param message: The edge message with shape [M, K]
    :param row_index: The row index of edges with shape [M]
    :param n_node: The number of nodes, N
    :param aggr: aggregation type, sum or mean
    :param mask: the edge mask (used in mean aggregation for counting degree)
    :return: The aggreagated node-wise information with shape [N, K]
    """
    result_shape = (n_node, message.shape[1])
    result = message.new_full(result_shape, 0)  # [N, K]
    row_index = row_index.unsqueeze(-1).expand(-1, message.shape[1])  # [M, K]
    result.scatter_add_(0, row_index, message)  # [N, K]
    if aggr == 'sum':
        pass
    elif aggr == 'mean':
        count = message.new_full(result_shape, 0)
        ones = torch.ones_like(message)
        if mask is not None:
            ones = ones * mask.unsqueeze(-1)
        count.scatter_add_(0, row_index, ones)
        result = result / count.clamp(min=1)
    else:
        raise NotImplementedError('Unknown aggregation method:', aggr)
    return result  # [N, K]


class BaseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, residual=False, last_act=False, flat=False):
        super(BaseMLP, self).__init__()
        self.residual = residual
        if flat:
            activation = nn.Tanh()
            hidden_dim = 4 * hidden_dim
        if residual:
            assert output_dim == input_dim
        if last_act:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, output_dim),
                activation
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x):
        return self.mlp(x) if not self.residual else self.mlp(x) + x


class EquivariantScalarNet(nn.Module):
    def __init__(self, n_vector_input, hidden_dim, activation, n_scalar_input=0, norm=True, flat=True):
        """
        The universal O(n) equivariant network using scalars.
        :param n_input: The total number of input vectors.
        :param hidden_dim: The hidden dim of the network.
        :param activation: The activation function.
        """
        super(EquivariantScalarNet, self).__init__()
        self.input_dim = n_vector_input * n_vector_input + n_scalar_input
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        # self.output_dim = n_vector_input
        self.activation = activation
        self.norm = norm
        self.in_scalar_net = BaseMLP(self.input_dim, self.hidden_dim, self.hidden_dim, self.activation, last_act=True,
                                     flat=flat)
        self.out_vector_net = BaseMLP(self.hidden_dim, self.hidden_dim, n_vector_input, self.activation, flat=flat)
        self.out_scalar_net = BaseMLP(self.hidden_dim, self.hidden_dim, self.output_dim, self.activation, flat=flat)

    def forward(self, vectors, scalars=None):
        """
        :param vectors: torch.Tensor with shape [N, 3, K] or a list of torch.Tensor
        :param scalars: torch.Tensor with shape [N, L] (Optional)
        :return: A vector that is equivariant to the O(n) transformations of input vectors with shape [N, 3]
        """
        if type(vectors) == list:
            Z = torch.stack(vectors, dim=-1)  # [N, 3, K]
        else:
            Z = vectors
        K = Z.shape[-1]
        Z_T = Z.transpose(-1, -2)  # [N, K, 3]
        scalar = torch.einsum('bij,bjk->bik', Z_T, Z)  # [N, K, K]
        scalar = scalar.reshape(-1, K * K)  # [N, KK]
        if self.norm:
            scalar = F.normalize(scalar, p=2, dim=-1)  # [N, KK]
        if scalars is not None:
            scalar = torch.cat((scalar, scalars), dim=-1)  # [N, KK + L]
        scalar = self.in_scalar_net(scalar)  # [N, K]
        vec_scalar = self.out_vector_net(scalar)  # [N, K]
        vector = torch.einsum('bij,bj->bi', Z, vec_scalar)  # [N, 3]
        scalar = self.out_scalar_net(scalar)  # [N, H]

        return vector, scalar


class InvariantScalarNet(nn.Module):
    def __init__(self, n_vector_input, hidden_dim, output_dim, activation, n_scalar_input=0, norm=True, last_act=False,
                 flat=False):
        """
        The universal O(n) invariant network using scalars.
        :param n_vector_input: The total number of input vectors.
        :param hidden_dim: The hidden dim of the network.
        :param activation: The activation function.
        """
        super(InvariantScalarNet, self).__init__()
        self.input_dim = n_vector_input * n_vector_input + n_scalar_input
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.norm = norm
        self.scalar_net = BaseMLP(self.input_dim, self.hidden_dim, self.output_dim, self.activation, last_act=last_act,
                                  flat=flat)

    def forward(self, vectors, scalars=None):
        """
        :param vectors: torch.Tensor with shape [N, 3, K] or a list of torch.Tensor with shape [N, 3]
        :param scalars: torch.Tensor with shape [N, L] (Optional)
        :return: A scalar that is invariant to the O(n) transformations of input vectors  with shape [N, K]
        """
        if type(vectors) == list:
            Z = torch.stack(vectors, dim=-1)  # [N, 3, K]
        else:
            Z = vectors
        K = Z.shape[-1]
        Z_T = Z.transpose(-1, -2)  # [N, K, 3]
        scalar = torch.einsum('bij,bjk->bik', Z_T, Z)  # [N, K, K]
        scalar = scalar.reshape(-1, K * K)  # [N, KK]
        if self.norm:
            scalar = F.normalize(scalar, p=2, dim=-1)  # [N, KK]
        if scalars is not None:
            scalar = torch.cat((scalar, scalars), dim=-1)  # [N, KK + L]
        scalar = self.scalar_net(scalar)  # [N, K]
        return scalar


class EGNN_Layer(nn.Module):
    def __init__(self, in_edge_nf, hidden_nf, activation=nn.SiLU(), with_v=False, flat=False, norm=False,
                 h_update=True):
        super(EGNN_Layer, self).__init__()
        self.with_v = with_v
        self.edge_message_net = InvariantScalarNet(n_vector_input=1, hidden_dim=hidden_nf, output_dim=hidden_nf,
                                                   activation=activation, n_scalar_input=2 * hidden_nf + in_edge_nf,
                                                   norm=norm, last_act=True, flat=flat)
        self.coord_net = BaseMLP(input_dim=hidden_nf, hidden_dim=hidden_nf, output_dim=1, activation=activation,
                                 flat=flat)
        if self.with_v:
            self.node_v_net = BaseMLP(input_dim=hidden_nf, hidden_dim=hidden_nf, output_dim=1, activation=activation,
                                      flat=flat)
        else:
            self.node_v_net = None
        self.h_update = h_update
        if self.h_update:
            self.node_net = BaseMLP(input_dim=hidden_nf + hidden_nf, hidden_dim=hidden_nf, output_dim=hidden_nf,
                                    activation=activation, flat=flat)

    def forward(self, x, h, edge_index, edge_fea, v=None):
        row, col = edge_index
        rij = x[row] - x[col]  # [BM, 3]
        hij = torch.cat((h[row], h[col], edge_fea), dim=-1)  # [BM, 2K+T]
        message = self.edge_message_net(vectors=[rij], scalars=hij)  # [BM, 3]
        coord_message = self.coord_net(message)  # [BM, 1]
        f = (x[row] - x[col]) * coord_message  # [BM, 3]
        tot_f = aggregate(message=f, row_index=row, n_node=x.shape[0], aggr='mean')  # [BN, 3]
        tot_f = torch.clamp(tot_f, min=-100, max=100)

        if v is not None:
            x = x + self.node_v_net(h) * v + tot_f
        else:
            x = x + tot_f  # [BN, 3]

        tot_message = aggregate(message=message, row_index=row, n_node=x.shape[0], aggr='sum')  # [BN, K]
        node_message = torch.cat((h, tot_message), dim=-1)  # [BN, K+K]
        if self.h_update:
            h = self.node_net(node_message)  # [BN, K]
        return x, v, h


class EGNN(nn.Module):
    def __init__(self, n_layers, in_node_nf, in_edge_nf, hidden_nf, activation=nn.SiLU(), device='cpu', with_v=False,
                 flat=False, norm=False):
        super(EGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.with_v = with_v
        # input feature mapping
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        for i in range(self.n_layers):
            layer = EGNN_Layer(in_edge_nf, hidden_nf, activation=activation, with_v=with_v, flat=flat, norm=norm)
            # if i == self.n_layers - 1:
            #     layer = EGNN_Layer(in_edge_nf, hidden_nf, activation=activation, with_v=with_v, flat=flat, norm=norm,
            #                        h_update=False)
            # else:
            #     layer = EGNN_Layer(in_edge_nf, hidden_nf, activation=activation, with_v=with_v, flat=flat, norm=norm)
            self.layers.append(layer)
        self.to(device)

    def forward(self, x, h, edge_index, edge_fea, v=None):
        h = self.embedding(h)
        for i in range(self.n_layers):
            x, v, h = self.layers[i](x, h, edge_index, edge_fea, v=v)
        return (x, v, h) if v is not None else (x, h)


class EGMN(nn.Module):
    def __init__(self, n_layers, n_vector_input, hidden_dim, n_scalar_input, activation=nn.SiLU(), norm=False, flat=False):
        super(EGMN, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        for i in range(self.n_layers):
            cur_layer = EquivariantScalarNet(n_vector_input=n_vector_input + i, hidden_dim=hidden_dim,
                                             activation=activation, n_scalar_input=n_scalar_input if i == 0 else hidden_dim,
                                             norm=norm, flat=flat)
            self.layers.append(cur_layer)

    def forward(self, vectors, scalars):
        cur_vectors = vectors
        for i in range(self.n_layers):
            vector, scalars = self.layers[i](cur_vectors, scalars)
            cur_vectors.append(vector)
        return cur_vectors[-1], scalars


class GNN_Layer(nn.Module):
    def __init__(self, in_edge_nf, hidden_nf, activation=nn.SiLU(), with_v=False, flat=False):
        super(GNN_Layer, self).__init__()
        self.with_v = with_v
        self.edge_message_net = BaseMLP(input_dim=in_edge_nf + 2 * hidden_nf, hidden_dim=hidden_nf, output_dim=hidden_nf,
                                        activation=activation, flat=flat)
        self.node_net = BaseMLP(input_dim=hidden_nf + hidden_nf, hidden_dim=hidden_nf, output_dim=hidden_nf,
                                activation=activation, flat=flat)

    def forward(self, h, edge_index, edge_fea):
        row, col = edge_index
        hij = torch.cat((h[row], h[col], edge_fea), dim=-1)  # [BM, 2K+T]
        message = self.edge_message_net(hij)  # [BM, K]
        agg = aggregate(message=message, row_index=row, n_node=h.shape[0], aggr='mean')  # [BN, K]
        h = h + self.node_net(torch.cat((agg, h), dim=-1))
        return h


class GNN(nn.Module):
    def __init__(self, n_layers, in_node_nf, in_edge_nf, hidden_nf, activation=nn.SiLU(), device='cpu', flat=False):
        super(GNN, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        # input feature mapping
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        for i in range(self.n_layers):
            layer = GNN_Layer(in_edge_nf, hidden_nf, activation=activation, flat=flat)
            self.layers.append(layer)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, 3)
        )
        self.to(device)

    def forward(self, h, edge_index, edge_fea):
        h = self.embedding(h)
        for i in range(self.n_layers):
            h = self.layers[i](h, edge_index, edge_fea)
        h = self.decoder(h)
        return h


class Linear_dynamics(nn.Module):
    def __init__(self, device='cpu'):
        super(Linear_dynamics, self).__init__()
        self.time = nn.Parameter(torch.ones(1))
        self.device = device
        self.to(self.device)

    def forward(self, x, v):
        return x + v * self.time


class RF_vel(nn.Module):
    def __init__(self, hidden_nf, edge_attr_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=4):
        super(RF_vel, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL_rf_vel(nf=hidden_nf, edge_attr_nf=edge_attr_nf, act_fn=act_fn))
        self.to(self.device)

    def forward(self, vel_norm, x, edges, vel, edge_attr):
        for i in range(0, self.n_layers):
            x, _ = self._modules["gcl_%d" % i](x, vel_norm, vel, edges, edge_attr)
        return x


class GCL_rf_vel(nn.Module):
    def __init__(self,  nf=64, edge_attr_nf=0, act_fn=nn.LeakyReLU(0.2), coords_weight=1.0):
        super(GCL_rf_vel, self).__init__()
        self.coords_weight = coords_weight
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(1, nf),
            act_fn,
            nn.Linear(nf, 1))

        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi = nn.Sequential(nn.Linear(1 + edge_attr_nf, nf),
                                 act_fn,
                                 layer,
                                 nn.Tanh())

    def forward(self, x, vel_norm, vel, edge_index, edge_attr=None):
        row, col = edge_index
        edge_m = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_m)
        x += vel * self.coord_mlp_vel(vel_norm)
        return x, edge_attr

    def edge_model(self, source, target, edge_attr):
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        return m_ij

    def node_model(self, x, edge_index, edge_m):
        row, col = edge_index
        agg = unsorted_segment_mean(edge_m, row, num_segments=x.size(0))
        x_out = x + agg * self.coords_weight
        return x_out


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class FullMLP(nn.ModuleList):
    def __init__(self, in_node_nf, hidden_nf, n_layers, activation=nn.SiLU(), flat=False, device='cpu'):
        super(FullMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        for i in range(n_layers):
            self.layers.append(BaseMLP(hidden_nf, hidden_nf, hidden_nf, activation,
                                       residual=True, last_act=True, flat=flat))
        self.output = nn.Linear(hidden_nf, 3)
        self.to(device)

    def forward(self, x):
        x = self.embedding(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return self.output(x)
