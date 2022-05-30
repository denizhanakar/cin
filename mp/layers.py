import torch

from typing import Callable, Optional, Tuple
from torch import Tensor
from zmq import device
from mp.cell_mp import CochainMessagePassing, CochainMessagePassingParams
from torch_geometric.nn.inits import reset
from torch.nn import Linear, Sequential, BatchNorm1d as BN, Identity
from data.complex import Cochain
from torch_scatter import scatter
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from abc import ABC, abstractmethod

from torch_scatter import gather_csr, scatter, segment_csr
from torch_geometric.nn.conv.utils.helpers import expand_left


class DummyCochainMessagePassing(CochainMessagePassing):
    """This is a dummy parameter-free message passing model used for testing."""
    def __init__(self, up_msg_size, down_msg_size, boundary_msg_size=None,
                 use_boundary_msg=False, use_down_msg=True):
        super(DummyCochainMessagePassing, self).__init__(up_msg_size, down_msg_size,
                                                       boundary_msg_size=boundary_msg_size,
                                                       use_boundary_msg=use_boundary_msg,
                                                       use_down_msg=use_down_msg)

    def message_up(self, up_x_j: Tensor, up_attr: Tensor) -> Tensor:
        # (num_up_adj, x_feature_dim) + (num_up_adj, up_feat_dim)
        # We assume the feature dim is the same across al levels
        return up_x_j + up_attr

    def message_down(self, down_x_j: Tensor, down_attr: Tensor) -> Tensor:
        # (num_down_adj, x_feature_dim) + (num_down_adj, down_feat_dim)
        # We assume the feature dim is the same across al levels
        return down_x_j + down_attr

    def forward(self, cochain: CochainMessagePassingParams):
        up_out, down_out, boundary_out = self.propagate(cochain.up_index, cochain.down_index,
                                                    cochain.boundary_index, x=cochain.x,
                                                    up_attr=cochain.kwargs['up_attr'],
                                                    down_attr=cochain.kwargs['down_attr'],
                                                    boundary_attr=cochain.kwargs['boundary_attr'])
        # down or boundary will be zero if one of them is not used.
        return cochain.x + up_out + down_out + boundary_out


class DummyCellularMessagePassing(torch.nn.Module):
    def __init__(self, input_dim=1, max_dim: int = 2, use_boundary_msg=False, use_down_msg=True):
        super(DummyCellularMessagePassing, self).__init__()
        self.max_dim = max_dim
        self.mp_levels = torch.nn.ModuleList()
        for dim in range(max_dim+1):
            mp = DummyCochainMessagePassing(input_dim, input_dim, boundary_msg_size=input_dim,
                                          use_boundary_msg=use_boundary_msg, use_down_msg=use_down_msg)
            self.mp_levels.append(mp)
    
    def forward(self, *cochain_params: CochainMessagePassingParams):
        assert len(cochain_params) <= self.max_dim+1

        out = []
        for dim in range(len(cochain_params)):
            out.append(self.mp_levels[dim].forward(cochain_params[dim]))
        return out


class CINCochainConv(CochainMessagePassing):
    """This is a dummy parameter-free message passing model used for testing."""
    def __init__(self, up_msg_size: int, down_msg_size: int,
                 msg_up_nn: Callable, msg_down_nn: Callable, update_nn: Callable,
                 eps: float = 0., train_eps: bool = False):
        super(CINCochainConv, self).__init__(up_msg_size, down_msg_size, use_boundary_msg=False)
        self.msg_up_nn = msg_up_nn
        self.msg_down_nn = msg_down_nn
        self.update_nn = update_nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def forward(self, cochain: CochainMessagePassingParams):
        out_up, out_down, _ = self.propagate(cochain.up_index, cochain.down_index,
                                             None, x=cochain.x,
                                             up_attr=cochain.kwargs['up_attr'],
                                             down_attr=cochain.kwargs['down_attr'])

        out_up += (1 + self.eps) * cochain.x
        out_down += (1 + self.eps) * cochain.x
        return self.update_nn(out_up + out_down)

    def reset_parameters(self):
        reset(self.msg_up_nn)
        reset(self.msg_down_nn)
        reset(self.update_nn)
        self.eps.data.fill_(self.initial_eps)

    def message_up(self, up_x_j: Tensor, up_attr: Tensor) -> Tensor:
        if up_attr is not None:
            x = torch.cat([up_x_j, up_attr], dim=-1)
            return self.msg_up_nn(x)
        else:
            return self.msg_up_nn(up_x_j)

    def message_down(self, down_x_j: Tensor, down_attr: Tensor) -> Tensor:
        x = torch.cat([down_x_j, down_attr], dim=-1)
        return self.msg_down_nn(x)


class CINConv(torch.nn.Module):
    def __init__(self, up_msg_size: int, down_msg_size: int,
                 msg_up_nn: Callable, msg_down_nn: Callable, update_nn: Callable,
                 eps: float = 0., train_eps: bool = False, max_dim: int = 2):
        super(CINConv, self).__init__()
        self.max_dim = max_dim
        self.mp_levels = torch.nn.ModuleList()
        for dim in range(max_dim+1):
            mp = CINCochainConv(up_msg_size, down_msg_size,
                              msg_up_nn, msg_down_nn, update_nn, eps, train_eps)
            self.mp_levels.append(mp)

    def forward(self, *cochain_params: CochainMessagePassingParams):
        assert len(cochain_params) <= self.max_dim+1

        out = []
        for dim in range(len(cochain_params)):
            out.append(self.mp_levels[dim].forward(cochain_params[dim]))
        return out


class EdgeCINConv(torch.nn.Module):
    """
    CIN convolutional layer which performs cochain message passing only
    _up to_ 1-dimensional cells (edges).
    """
    def __init__(self, up_msg_size: int, down_msg_size: int,
                 v_msg_up_nn: Callable, e_msg_down_nn: Callable, e_msg_up_nn: Callable,
                 v_update_nn: Callable, e_update_nn: Callable, eps: float = 0., train_eps=False):
        super(EdgeCINConv, self).__init__()
        self.max_dim = 1
        self.mp_levels = torch.nn.ModuleList()

        v_mp = CINCochainConv(up_msg_size, down_msg_size,
                            v_msg_up_nn, lambda *args: None, v_update_nn, eps, train_eps)
        e_mp = CINCochainConv(up_msg_size, down_msg_size,
                            e_msg_up_nn, e_msg_down_nn, e_update_nn, eps, train_eps)
        self.mp_levels.extend([v_mp, e_mp])

    def forward(self, *cochain_params: CochainMessagePassingParams):
        assert len(cochain_params) <= self.max_dim+1

        out = []
        for dim in range(len(cochain_params)):
            out.append(self.mp_levels[dim].forward(cochain_params[dim]))
        return out


class SparseCINCochainConv(CochainMessagePassing):
    """This is a CIN Cochain layer that operates of boundaries and upper adjacent cells."""
    def __init__(self, dim: int,
                 up_msg_size: int,
                 down_msg_size: int,
                 boundary_msg_size: Optional[int],
                 msg_up_nn: Callable,
                 msg_boundaries_nn: Callable,
                 update_up_nn: Callable,
                 update_boundaries_nn: Callable,
                 combine_nn: Callable,
                 eps: float = 0.,
                 train_eps: bool = False,
                 use_pos: bool = False,
                 use_complete: bool = False):
        super(SparseCINCochainConv, self).__init__(up_msg_size, down_msg_size, boundary_msg_size=boundary_msg_size,
                                                 use_down_msg=False)
        self.dim = dim
        self.use_pos = use_pos
        self.use_complete = use_complete
        self.msg_up_nn = msg_up_nn
        self.msg_boundaries_nn = msg_boundaries_nn
        self.update_up_nn = update_up_nn
        self.update_boundaries_nn = update_boundaries_nn
        self.combine_nn = combine_nn
        self.initial_eps = eps
        if train_eps:
            self.eps1 = torch.nn.Parameter(torch.Tensor([eps]))
            self.eps2 = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps1', torch.Tensor([eps]))
            self.register_buffer('eps2', torch.Tensor([eps]))
        self.reset_parameters()

    def forward(self, cochain: CochainMessagePassingParams):
        """
        The paper specifies (1+\epsilon) and also a sum across all \tau from the boundary of \sigma.
        We first get the boundary via self.propagate(?)
        Then we add cochain.x to out_up, an up message output.
        We finally combine both into an NN and return.
        """
        # breakpoint()
        out_up, _, out_boundaries = self.propagate(cochain.up_index, cochain.down_index,
                                              cochain.boundary_index, x=cochain.x,
                                              pos=cochain.pos,
                                              complete_graph_index=cochain.complete_graph_index,
                                              use_complete=self.use_complete,
                                              up_attr=cochain.kwargs['up_attr'],
                                              boundary_attr=cochain.kwargs['boundary_attr'],
                                              use_pos=self.use_pos)
        # As in GIN, we can learn an injective update function for each multi-set
        # Add 
        out_up += (1 + self.eps1) * cochain.x
        out_boundaries += (1 + self.eps2) * cochain.x
        out_up = self.update_up_nn(out_up)
        out_boundaries = self.update_boundaries_nn(out_boundaries)

        # We need to combine the two such that the output is injective
        # Because the cross product of countable spaces is countable, then such a function exists.
        # And we can learn it with another MLP.
        return self.combine_nn(torch.cat([out_up, out_boundaries], dim=-1))

    def reset_parameters(self):
        reset(self.msg_up_nn)
        reset(self.msg_boundaries_nn)
        reset(self.update_up_nn)
        reset(self.update_boundaries_nn)
        reset(self.combine_nn)
        self.eps1.data.fill_(self.initial_eps)
        self.eps2.data.fill_(self.initial_eps)

    def message_up(self, up_x_j: Tensor, up_pos_j: Tensor, up_pos_i: Tensor, up_attr: Tensor) -> Tensor:
        """
        up_x_j:
        up_pos_j:
        """
        # breakpoint()
        # WE CAN USE self.dim!
        if self.use_pos and self.dim == 0:
            distances = torch.linalg.norm(up_pos_i - up_pos_j, dim=1).unsqueeze(1)
            to_cat_zeros = torch.zeros(up_x_j.shape[0] - up_attr.shape[0], up_attr.shape[1]).to(device=up_x_j.device)
            up_attr = torch.cat((up_attr, to_cat_zeros), 0)
            return self.msg_up_nn((up_x_j, up_attr, distances))
            # return self.msg_up_nn((up_x_j, up_attr, torch.zeros(up_x_j.shape[0], 1).to(device=up_x_j.device)))
        else:
            return self.msg_up_nn((up_x_j, up_attr))
    
    def message_boundary(self, boundary_x_j: Tensor) -> Tensor:
        # breakpoint()
        return self.msg_boundaries_nn(boundary_x_j)


class SparseEquivCINCochainConv(CochainMessagePassing):
    """This is a CIN Cochain layer that operates of boundaries and upper adjacent cells."""
    def __init__(self, dim: int,
                 up_msg_size: int,
                 down_msg_size: int,
                 boundary_msg_size: Optional[int],
                 msg_up_nn: Callable,
                 msg_boundaries_nn: Callable,
                 update_up_nn: Callable,
                 update_boundaries_nn: Callable,
                 combine_nn: Callable,
                 eps: float = 0.,
                 train_eps: bool = False,
                 use_complete: bool = False):
        super(SparseEquivCINCochainConv, self).__init__(up_msg_size, down_msg_size, boundary_msg_size=boundary_msg_size,
                                                 use_down_msg=False)
        self.dim = dim
        self.use_complete = use_complete
        self.mlp_wght = Linear(up_msg_size, 1)
        self.msg_up_nn = msg_up_nn
        self.msg_boundaries_nn = msg_boundaries_nn
        self.update_up_nn = update_up_nn
        self.update_boundaries_nn = update_boundaries_nn
        self.combine_nn = combine_nn
        self.initial_eps = eps
        if train_eps:
            self.eps1 = torch.nn.Parameter(torch.Tensor([eps]))
            self.eps2 = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps1', torch.Tensor([eps]))
            self.register_buffer('eps2', torch.Tensor([eps]))

        # Store weights for each relative position.
        self.weights = None
        # Store weighted sum of all relative positions.
        self.weighted_relative_pos = None

        self.reset_parameters()

    def forward(self, cochain: CochainMessagePassingParams):
        """
        The paper specifies (1+\epsilon) and also a sum across all \tau from the boundary of \sigma.
        We first get the boundary via self.propagate(?)
        Then we add cochain.x to out_up, an up message output.
        We finally combine both into an NN and return.
        """
        # breakpoint()
        out_up, _, out_boundaries, pos_out = self.propagate(cochain.up_index, cochain.down_index,
                                              cochain.boundary_index, x=cochain.x,
                                              pos=cochain.pos,
                                              complete_graph_index=cochain.complete_graph_index,
                                              use_complete=self.use_complete,
                                              up_attr=cochain.kwargs['up_attr'],
                                              boundary_attr=cochain.kwargs['boundary_attr'],
                                              use_pos=True)
        # As in GIN, we can learn an injective update function for each multi-set
        # Add 
        out_up += (1 + self.eps1) * cochain.x
        out_boundaries += (1 + self.eps2) * cochain.x
        out_up = self.update_up_nn(out_up)
        out_boundaries = self.update_boundaries_nn(out_boundaries)

        # We need to combine the two such that the output is injective
        # Because the cross product of countable spaces is countable, then such a function exists.
        # And we can learn it with another MLP.
        return self.combine_nn(torch.cat([out_up, out_boundaries], dim=-1)), pos_out

    def reset_parameters(self):
        reset(self.msg_up_nn)
        reset(self.msg_boundaries_nn)
        reset(self.update_up_nn)
        reset(self.update_boundaries_nn)
        reset(self.combine_nn)
        self.eps1.data.fill_(self.initial_eps)
        self.eps2.data.fill_(self.initial_eps)

    def message_up(self, up_x_j: Tensor, up_pos_j: Tensor, up_pos_i: Tensor, up_attr: Tensor) -> Tensor:
        """
        up_x_j:
        up_pos_j:
        """
        # breakpoint()
        # WE CAN USE self.dim!
        if self.dim == 0:
            distances = torch.linalg.norm(up_pos_i - up_pos_j, dim=1).unsqueeze(1)
            to_cat_zeros = torch.zeros(up_x_j.shape[0] - up_attr.shape[0], up_attr.shape[1]).to(device=up_x_j.device)
            up_attr = torch.cat((up_attr, to_cat_zeros), 0)
            messages = self.msg_up_nn((up_x_j, up_attr, distances))
            self.weights = self.mlp_wght(messages)
            return messages
        else:
            return self.msg_up_nn((up_x_j, up_attr))

    def aggregate_up(self, inputs: Tensor, agg_up_index: Tensor,
                     up_pos_j: Tensor, up_pos_i: Tensor,
                     up_ptr: Optional[Tensor] = None,
                     up_dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from upper adjacent cells.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        self.weighted_relative_pos = scatter((up_pos_i - up_pos_j) * self.weights, agg_up_index, dim=self.node_dim, reduce="mean")
        if up_ptr is not None:
            up_ptr = expand_left(up_ptr, dim=self.node_dim, dims=inputs.dim())
            return segment_csr(inputs, up_ptr, reduce=self.aggr_up)
        else:
            return scatter(inputs, agg_up_index, dim=self.node_dim, dim_size=up_dim_size,
                           reduce=self.aggr_up)
    
    def message_boundary(self, boundary_x_j: Tensor) -> Tensor:
        # breakpoint()
        return self.msg_boundaries_nn(boundary_x_j)

    def update(self, up_inputs: Optional[Tensor], down_inputs: Optional[Tensor],
               boundary_inputs: Optional[Tensor], x: Tensor, pos: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Updates cell embeddings. Takes in the output of the aggregations from different
        adjacencies as the first three arguments and any argument which was initially passed to
        :meth:`propagate`.
        """
        # breakpoint()
        if up_inputs is None:
            up_inputs = torch.zeros(x.size(0), self.up_msg_size).to(device=x.device)
        if down_inputs is None:
            down_inputs = torch.zeros(x.size(0), self.down_msg_size).to(device=x.device)
        if boundary_inputs is None:
            boundary_inputs = torch.zeros(x.size(0), self.boundary_msg_size).to(device=x.device)

        pos_out = pos + self.weighted_relative_pos

        return up_inputs, down_inputs, boundary_inputs, pos_out


class Catter(torch.nn.Module):
    def __init__(self):
        super(Catter, self).__init__()

    def forward(self, x):
        # breakpoint()
        return torch.cat(x, dim=-1)


class SparseCINConv(torch.nn.Module):
    """A cellular version of GIN which performs message passing from cellular upper
    neighbors and boundaries, but not from lower neighbors (hence why "Sparse")
    """

    # TODO: Refactor the way we pass networks externally to allow for different networks per dim.
    def __init__(self, up_msg_size: int, down_msg_size: int, boundary_msg_size: Optional[int],
                 passed_msg_up_nn: Optional[Callable], passed_msg_boundaries_nn: Optional[Callable],
                 passed_update_up_nn: Optional[Callable],
                 passed_update_boundaries_nn: Optional[Callable],
                 eps: float = 0., train_eps: bool = False, max_dim: int = 2,
                 graph_norm=BN, use_coboundaries=False, use_pos=False, **kwargs):
        super(SparseCINConv, self).__init__()
        self.max_dim = max_dim
        self.mp_levels = torch.nn.ModuleList()
        for dim in range(max_dim+1):
            msg_up_nn = passed_msg_up_nn
            if msg_up_nn is None:
                if use_coboundaries:
                    if use_pos and dim == 0:
                        msg_up_nn = Sequential(
                                Catter(),
                                Linear(kwargs['layer_dim'] * 2 + 1, kwargs['layer_dim']),
                                kwargs['act_module']())
                    else:
                        msg_up_nn = Sequential(
                                Catter(),
                                Linear(kwargs['layer_dim'] * 2, kwargs['layer_dim']),
                                kwargs['act_module']())
                else:
                    msg_up_nn = lambda xs: xs[0]

            msg_boundaries_nn = passed_msg_boundaries_nn
            if msg_boundaries_nn is None:
                msg_boundaries_nn = lambda x: x

            update_up_nn = passed_update_up_nn
            if update_up_nn is None:
                update_up_nn = Sequential(
                    Linear(kwargs['layer_dim'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module'](),
                    Linear(kwargs['hidden'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module']()
                )

            update_boundaries_nn = passed_update_boundaries_nn
            if update_boundaries_nn is None:
                update_boundaries_nn = Sequential(
                    Linear(kwargs['layer_dim'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module'](),
                    Linear(kwargs['hidden'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module']()
                )
            combine_nn = Sequential(
                Linear(kwargs['hidden']*2, kwargs['hidden']),
                graph_norm(kwargs['hidden']),
                kwargs['act_module']())

            mp = SparseCINCochainConv(dim, up_msg_size, down_msg_size, boundary_msg_size=boundary_msg_size,
                msg_up_nn=msg_up_nn, msg_boundaries_nn=msg_boundaries_nn, update_up_nn=update_up_nn,
                update_boundaries_nn=update_boundaries_nn, combine_nn=combine_nn, eps=eps,
                train_eps=train_eps, use_pos=use_pos)
            self.mp_levels.append(mp)

    def forward(self, *cochain_params: CochainMessagePassingParams, start_to_process=0):
        assert len(cochain_params) <= self.max_dim+1

        out = []
        for dim in range(len(cochain_params)):
            if dim < start_to_process:
                out.append(cochain_params[dim].x)
            else:
                out.append(self.mp_levels[dim].forward(cochain_params[dim]))
        return out


class SparseInvarCINConv(torch.nn.Module):
    """A cellular version of GIN which performs message passing from cellular upper
    neighbors and boundaries, but not from lower neighbors (hence why "Sparse")
    """

    # TODO: Refactor the way we pass networks externally to allow for different networks per dim.
    def __init__(self, up_msg_size: int, down_msg_size: int, boundary_msg_size: Optional[int],
                 passed_msg_up_nn: Optional[Callable], passed_msg_boundaries_nn: Optional[Callable],
                 passed_update_up_nn: Optional[Callable],
                 passed_update_boundaries_nn: Optional[Callable],
                 eps: float = 0., train_eps: bool = False, max_dim: int = 2,
                 graph_norm=BN, use_coboundaries=False,
                 use_pos=False, use_complete=False, **kwargs):
        super(SparseInvarCINConv, self).__init__()
        self.max_dim = max_dim
        self.mp_levels = torch.nn.ModuleList()
        for dim in range(max_dim+1):
            msg_up_nn = passed_msg_up_nn
            if msg_up_nn is None:
                if use_coboundaries:
                    if use_pos and dim == 0:
                        msg_up_nn = Sequential(
                                Catter(),
                                Linear(kwargs['layer_dim'] * 2 + 1, kwargs['layer_dim']),
                                kwargs['act_module']())
                    else:
                        msg_up_nn = Sequential(
                                Catter(),
                                Linear(kwargs['layer_dim'] * 2, kwargs['layer_dim']),
                                kwargs['act_module']())
                else:
                    msg_up_nn = lambda xs: xs[0]

            msg_boundaries_nn = passed_msg_boundaries_nn
            if msg_boundaries_nn is None:
                msg_boundaries_nn = lambda x: x

            update_up_nn = passed_update_up_nn
            if update_up_nn is None:
                update_up_nn = Sequential(
                    Linear(kwargs['layer_dim'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module'](),
                    Linear(kwargs['hidden'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module']()
                )

            update_boundaries_nn = passed_update_boundaries_nn
            if update_boundaries_nn is None:
                update_boundaries_nn = Sequential(
                    Linear(kwargs['layer_dim'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module'](),
                    Linear(kwargs['hidden'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module']()
                )
            combine_nn = Sequential(
                Linear(kwargs['hidden']*2, kwargs['hidden']),
                graph_norm(kwargs['hidden']),
                kwargs['act_module']())

            mp = SparseCINCochainConv(dim, up_msg_size, down_msg_size, boundary_msg_size=boundary_msg_size,
                msg_up_nn=msg_up_nn, msg_boundaries_nn=msg_boundaries_nn, update_up_nn=update_up_nn,
                update_boundaries_nn=update_boundaries_nn, combine_nn=combine_nn, eps=eps,
                train_eps=train_eps, use_pos=use_pos, use_complete=use_complete)
            self.mp_levels.append(mp)

    def forward(self, *cochain_params: CochainMessagePassingParams, start_to_process=0):
        assert len(cochain_params) <= self.max_dim+1

        out = []
        for dim in range(len(cochain_params)):
            if dim < start_to_process:
                out.append(cochain_params[dim].x)
            else:
                out.append(self.mp_levels[dim].forward(cochain_params[dim]))
        return out


class SparseEquivCINConv(torch.nn.Module):
    """A cellular version of GIN which performs message passing from cellular upper
    neighbors and boundaries, but not from lower neighbors (hence why "Sparse")
    """

    # TODO: Refactor the way we pass networks externally to allow for different networks per dim.
    # This?
    def __init__(self, up_msg_size: int, down_msg_size: int, boundary_msg_size: Optional[int],
                 passed_msg_up_nn: Optional[Callable], passed_msg_boundaries_nn: Optional[Callable],
                 passed_update_up_nn: Optional[Callable],
                 passed_update_boundaries_nn: Optional[Callable],
                 eps: float = 0., train_eps: bool = False, max_dim: int = 2,
                 graph_norm=BN, use_coboundaries=False,
                 use_complete=False, **kwargs):
        super(SparseEquivCINConv, self).__init__()
        self.max_dim = max_dim
        self.mp_levels = torch.nn.ModuleList()
        for dim in range(max_dim+1):
            msg_up_nn = passed_msg_up_nn
            if msg_up_nn is None:
                if use_coboundaries:
                    if dim == 0:
                        # breakpoint()
                        msg_up_nn = Sequential(
                                Catter(),
                                Linear(kwargs['layer_dim'] * 2 + 1, kwargs['layer_dim']),
                                kwargs['act_module']())
                    else:
                        msg_up_nn = Sequential(
                                Catter(),
                                Linear(kwargs['layer_dim'] * 2, kwargs['layer_dim']),
                                kwargs['act_module']())
                else:
                    msg_up_nn = lambda xs: xs[0]

            msg_boundaries_nn = passed_msg_boundaries_nn
            if msg_boundaries_nn is None:
                msg_boundaries_nn = lambda x: x

            update_up_nn = passed_update_up_nn
            if update_up_nn is None:
                update_up_nn = Sequential(
                    Linear(kwargs['layer_dim'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module'](),
                    Linear(kwargs['hidden'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module']()
                )

            update_boundaries_nn = passed_update_boundaries_nn
            if update_boundaries_nn is None:
                update_boundaries_nn = Sequential(
                    Linear(kwargs['layer_dim'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module'](),
                    Linear(kwargs['hidden'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module']()
                )
            combine_nn = Sequential(
                Linear(kwargs['hidden']*2, kwargs['hidden']),
                graph_norm(kwargs['hidden']),
                kwargs['act_module']())

            if dim == 0:
                mp = SparseEquivCINCochainConv(dim, up_msg_size, down_msg_size, boundary_msg_size=boundary_msg_size,
                    msg_up_nn=msg_up_nn, msg_boundaries_nn=msg_boundaries_nn, update_up_nn=update_up_nn,
                    update_boundaries_nn=update_boundaries_nn, combine_nn=combine_nn, eps=eps,
                    train_eps=train_eps, use_complete=use_complete)
            else:
                mp = SparseCINCochainConv(dim, up_msg_size, down_msg_size, boundary_msg_size=boundary_msg_size,
                    msg_up_nn=msg_up_nn, msg_boundaries_nn=msg_boundaries_nn, update_up_nn=update_up_nn,
                    update_boundaries_nn=update_boundaries_nn, combine_nn=combine_nn, eps=eps,
                    train_eps=train_eps)
            self.mp_levels.append(mp)

    def forward(self, *cochain_params: CochainMessagePassingParams, start_to_process=0):
        assert len(cochain_params) <= self.max_dim+1

        out = []
        for dim in range(len(cochain_params)):
            if dim < start_to_process:
                out.append(cochain_params[dim].x)
            else:
                out.append(self.mp_levels[dim].forward(cochain_params[dim]))
        return out


class OrientedConv(CochainMessagePassing):
    def __init__(self, dim: int, up_msg_size: int, down_msg_size: int,
                 update_up_nn: Optional[Callable], update_down_nn: Optional[Callable],
                 update_nn: Optional[Callable], act_fn, orient=True):
        super(OrientedConv, self).__init__(up_msg_size, down_msg_size, use_boundary_msg=False)
        self.dim = dim
        self.update_up_nn = update_up_nn
        self.update_down_nn = update_down_nn
        self.update_nn = update_nn
        self.act_fn = act_fn
        self.orient = orient

    def forward(self, cochain: Cochain):
        assert len(cochain.upper_orient) == cochain.upper_index.size(1)
        assert len(cochain.lower_orient) == cochain.lower_index.size(1)
        assert cochain.upper_index.max() < len(cochain.x)
        assert cochain.lower_index.max() < len(cochain.x)

        out_up, out_down, _ = self.propagate(cochain.upper_index, cochain.lower_index, None, x=cochain.x,
            up_attr=cochain.upper_orient.view(-1, 1), down_attr=cochain.lower_orient.view(-1, 1))

        out_up = self.update_up_nn(out_up)
        out_down = self.update_down_nn(out_down)
        x = self.update_nn(cochain.x)
        return self.act_fn(x + out_up + out_down)

    def reset_parameters(self):
        reset(self.update_up_nn)
        reset(self.update_down_nn)
        reset(self.update_nn)

    # TODO: As a temporary hack, we pass the orientation through the up and down attributes.
    def message_up(self, up_x_j: Tensor, up_attr: Tensor) -> Tensor:
        if self.orient:
            return up_x_j * up_attr
        return up_x_j

    def message_down(self, down_x_j: Tensor, down_attr: Tensor) -> Tensor:
        if self.orient:
            return down_x_j * down_attr
        return down_x_j


class InitReduceConv(torch.nn.Module):

    def __init__(self, reduce='add'):
        """

        Args:
            reduce (str): Way to aggregate boundaries. Can be "sum, add, mean, min, max"
        """
        super(InitReduceConv, self).__init__()
        self.reduce = reduce

    def forward(self, boundary_x, boundary_index):
        features = boundary_x.index_select(0, boundary_index[0])
        out_size = boundary_index[1, :].max() + 1
        return scatter(features, boundary_index[1], dim=0, dim_size=out_size, reduce=self.reduce)

    
class AbstractEmbedVEWithReduce(torch.nn.Module, ABC):
    
    def __init__(self,
                 v_embed_layer: Callable,
                 e_embed_layer: Optional[Callable],
                 init_reduce: InitReduceConv):
        """

        Args:
            v_embed_layer: Layer to embed the integer features of the vertices
            e_embed_layer: Layer (potentially None) to embed the integer features of the edges.
            init_reduce: Layer to initialise the 2D cell features and potentially the edge features.
        """
        super(AbstractEmbedVEWithReduce, self).__init__()
        self.v_embed_layer = v_embed_layer
        self.e_embed_layer = e_embed_layer
        self.init_reduce = init_reduce
    
    @abstractmethod
    def _prepare_v_inputs(self, v_params):
        pass
    
    @abstractmethod
    def _prepare_e_inputs(self, e_params):
        pass
    
    def forward(self, *cochain_params: CochainMessagePassingParams):
        assert 1 <= len(cochain_params) <= 3
        v_params = cochain_params[0]
        e_params = cochain_params[1] if len(cochain_params) >= 2 else None
        c_params = cochain_params[2] if len(cochain_params) == 3 else None

        vx = self.v_embed_layer(self._prepare_v_inputs(v_params))
        out = [vx]

        if e_params is None:
           assert c_params is None
           return out

        reduced_ex = self.init_reduce(vx, e_params.boundary_index)
        ex = reduced_ex
        if e_params.x is not None:
            ex = self.e_embed_layer(self._prepare_e_inputs(e_params))
            # The output of this should be the same size as the vertex features.
            assert ex.size(1) == vx.size(1)
        out.append(ex)

        if c_params is not None:
            # We divide by two in case this was obtained from node aggregation.
            # The division should not do any harm if this is an aggregation of learned embeddings.
            cx = self.init_reduce(reduced_ex, c_params.boundary_index) / 2.
            out.append(cx)

        return out
    
    def reset_parameters(self):
        reset(self.v_embed_layer)
        reset(self.e_embed_layer)

    
class EmbedVEWithReduce(AbstractEmbedVEWithReduce):

    def __init__(self,
                 v_embed_layer: torch.nn.Embedding,
                 e_embed_layer: Optional[torch.nn.Embedding],
                 init_reduce: InitReduceConv):
        super(EmbedVEWithReduce, self).__init__(v_embed_layer, e_embed_layer, init_reduce)
        
    def _prepare_v_inputs(self, v_params):
        assert v_params.x is not None
        assert v_params.x.dim() == 2
        assert v_params.x.size(1) == 1
        # The embedding layer expects integers so we convert the tensor to int.
        return v_params.x.squeeze(1).to(dtype=torch.long)
    
    def _prepare_e_inputs(self, e_params):
        assert self.e_embed_layer is not None
        assert e_params.x.dim() == 2
        assert e_params.x.size(1) == 1
        # The embedding layer expects integers so we convert the tensor to int.
        return e_params.x.squeeze(1).to(dtype=torch.long)


class OGBEmbedVEWithReduce(AbstractEmbedVEWithReduce):
    
    def __init__(self,
                 v_embed_layer,
                 e_embed_layer,
                 init_reduce):
        super(OGBEmbedVEWithReduce, self).__init__(v_embed_layer, e_embed_layer, init_reduce)

    def _prepare_v_inputs(self, v_params):
        assert v_params.x is not None
        assert v_params.x.dim() == 2
        # Inputs in ogbg-mol* datasets are already long.
        # This is to test the layer with other datasets.
        return v_params.x#.to(dtype=torch.long)
    
    def _prepare_e_inputs(self, e_params):
        assert self.e_embed_layer is not None
        assert e_params.x.dim() == 2
        # Inputs in ogbg-mol* datasets are already long.
        # This is to test the layer with other datasets.
        return e_params.x#.to(dtype=torch.long)
