import types

import jax
import jaxtyping as jt
import jraph

from ... import utils as tensorial_utils


def get_number_of_padding_with_graphs_graphs(
    padded_graph: jraph.GraphsTuple, np_: types.ModuleType | None = None
) -> int:
    """Returns number of padding graphs in padded_graph.

    Warning: This method only gives results for graphs that have been padded with
    ``pad_with_graphs``.

    Args:
      padded_graph: a ``GraphsTuple`` that has been padded with
        ``pad_with_graphs``.

    Returns:
      The number of padding graphs.
    """
    # The first_padding graph always has at least one padding node, and
    # all padding graphs that follow have 0 nodes. We can count how many
    # trailing graphs have 0 nodes, and add one.
    if np_ is None:
        np_ = tensorial_utils.infer_backend(jax.tree.leaves(padded_graph))

    n_trailing_empty_padding_graphs = np_.argmin(padded_graph.n_node[::-1] == 0)
    return n_trailing_empty_padding_graphs + 1


def get_node_padding_mask(
    padded_graph: jraph.GraphsTuple, np_: types.ModuleType | None = None
) -> jt.Array:
    """Returns a mask for the nodes of a padded graph.

    Args:
      padded_graph: ``GraphsTuple`` padded using ``pad_with_graphs``. This graph
        must contain at least one array of node features so the total static
        number of nodes can be inferred statically from the shape, and the method
        can be jitted.

    Returns:
      Boolean array of shape [total_num_nodes] containing True for real nodes,
      and False for padding nodes.
    """
    if np_ is None:
        np_ = tensorial_utils.infer_backend(jax.tree.leaves(padded_graph))

    n_padding_node = get_number_of_padding_with_graphs_nodes(padded_graph, np_)
    flat_node_features = jax.tree.leaves(padded_graph.nodes)

    if not flat_node_features:
        raise ValueError("`padded_graph` must have at least one array of node features")
    total_num_nodes = flat_node_features[0].shape[0]
    return _get_mask(padding_length=n_padding_node, full_length=total_num_nodes, np_=np_)


def get_edge_padding_mask(padded_graph: jraph.GraphsTuple, np_=None) -> jt.Array:
    """Returns a mask for the edges of a padded graph.

    Args:
      padded_graph: ``GraphsTuple`` padded using ``pad_with_graphs``.

    Returns:
      Boolean array of shape [total_num_edges] containing True for real edges,
      and False for padding edges.
    """
    if np_ is None:
        np_ = tensorial_utils.infer_backend(jax.tree.leaves(padded_graph))

    n_padding_edge = get_number_of_padding_with_graphs_edges(padded_graph, np_)
    total_num_edges = padded_graph.senders.shape[0]
    return _get_mask(padding_length=n_padding_edge, full_length=total_num_edges, np_=np_)


def get_graph_padding_mask(padded_graph: jraph.GraphsTuple, np_=None) -> jt.Array:
    """Returns a mask for the graphs of a padded graph.

    Args:
      padded_graph: ``GraphsTuple`` padded using ``pad_with_graphs``.

    Returns:
      Boolean array of shape [total_num_graphs] containing True for real graphs,
      and False for padding graphs.
    """
    if np_ is None:
        np_ = tensorial_utils.infer_backend(jax.tree.leaves(padded_graph))

    n_padding_graph = get_number_of_padding_with_graphs_graphs(padded_graph, np_)
    total_num_graphs = padded_graph.n_node.shape[0]
    return _get_mask(padding_length=n_padding_graph, full_length=total_num_graphs, np_=np_)


def get_number_of_padding_with_graphs_nodes(
    padded_graph: jraph.GraphsTuple, np_: types.ModuleType
) -> int:
    """Returns number of padding nodes in given padded_graph.

    Warning: This method only gives results for graphs that have been padded with
    ``pad_with_graphs``.

    Args:
      padded_graph: a ``GraphsTuple`` that has been padded with
        ``pad_with_graphs``.

    Returns:
      The number of padding nodes.
    """
    return padded_graph.n_node[-get_number_of_padding_with_graphs_graphs(padded_graph, np_)]


def get_number_of_padding_with_graphs_edges(
    padded_graph: jraph.GraphsTuple, np_: types.ModuleType
) -> int:
    """Returns number of padding edges in given padded_graph.

    Warning: This method only gives results for graphs that have been padded with
    ``pad_with_graphs``.

    Args:
      padded_graph: a ``GraphsTuple`` that has been padded with
        ``pad_with_graphs``.

    Returns:
      The number of padding edges.
    """
    return padded_graph.n_edge[-get_number_of_padding_with_graphs_graphs(padded_graph, np_)]


def _get_mask(padding_length, full_length, np_: types.ModuleType) -> jt.Array:
    valid_length = full_length - padding_length
    return np_.arange(full_length, dtype=np_.int32) < valid_length
