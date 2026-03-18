import jraph

from .gcnn import keys


def determine_graph_batch_size(graphs: jraph.GraphsTuple):
    if len(graphs.n_node.shape) > 1:
        # explicit batching
        if isinstance(graphs.globals, dict) and (keys.MASK in graphs.globals):
            return (graphs.globals[keys.MASK].sum(),)

        return (graphs.n_node.shape[0],)

    return (len(graphs.n_node) - jraph.get_number_of_padding_with_graphs_graphs(graphs),)


def get_batch_sizers() -> list:
    return [(jraph.GraphsTuple, determine_graph_batch_size)]
