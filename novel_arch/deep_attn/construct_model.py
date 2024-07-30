from novel_arch.deep_attn.model import DeepBDE
from novel_arch.deep_attn.feat_type_updaters import concat_sum_atom_edge_feat, aggreg_atom_edge_no_repeat, AttnNodeEdgeAggreg, AtomEdgeReducer, bond_mean, atom_mean, bond_sum, atom_sum, A2GReducer, B2GReducer

def get_std_model(
        atom_aggregators=concat_sum_atom_edge_feat, 
        b2g_aggregators=bond_mean(), a2g_aggregators=atom_mean(),
        fc_readout_sizes=[128]+[64]*4, 
        graph_inner_layer_sizes=[[64]*3]*4, 
        graph_hidden_size=32,
        in_feat_sizes={'atom': 12, 'bond': 7, 'global': 3},
        dropout=0.0,
        **kwargs):
    model = DeepBDE(
            atom_aggregators=atom_aggregators,
            b2g_aggregators=b2g_aggregators,
            a2g_aggregators=a2g_aggregators,
            in_feat_sizes=in_feat_sizes,
            graph_hidden_size=graph_hidden_size,
            graph_layers=len(graph_inner_layer_sizes),
            graph_inner_layer_sizes=graph_inner_layer_sizes,
            residual=True,
            fc_readout_sizes=fc_readout_sizes,
            dropout=dropout,
            **kwargs,
        )
    return model

def get_std_sum(b2g_aggregators=bond_sum(), a2g_aggregators=atom_sum(), **kwargs):
    return get_std_model(b2g_aggregators=b2g_aggregators, a2g_aggregators=a2g_aggregators, **kwargs)

def get_std_sum_full(b2g_aggregators=bond_sum(), a2g_aggregators=atom_sum(), # this function is almost always used to construct DeepDBE
                graph_inner_layer_sizes=[[128]*4]*5, 
                graph_hidden_size=64, 
                fc_readout_sizes=[256]+[128]*3, 
                **kwargs):
    return get_std_model(graph_hidden_size=graph_hidden_size, 
                        fc_readout_sizes=fc_readout_sizes,
                        graph_inner_layer_sizes=graph_inner_layer_sizes, 
                        b2g_aggregators=b2g_aggregators, a2g_aggregators=a2g_aggregators, 
                        **kwargs)

def get_std_sum_full_deeper(b2g_aggregators=bond_sum(), a2g_aggregators=atom_sum(), 
                graph_inner_layer_sizes=[[128]*6]*6, 
                graph_hidden_size=64, 
                fc_readout_sizes=[256]+[128]*4, 
                **kwargs):
    return get_std_model(graph_hidden_size=graph_hidden_size, 
                        fc_readout_sizes=fc_readout_sizes,
                        graph_inner_layer_sizes=graph_inner_layer_sizes, 
                        b2g_aggregators=b2g_aggregators, a2g_aggregators=a2g_aggregators, 
                        **kwargs)

def get_std_sum_full_attn(sum_like=True,
            zero_attn_scalar_map=False,
            internal_attn_size=16, 
            graph_hidden_size=64,
            graph_inner_layer_sizes=[[128]*4]*5,
            attn_layers=1, **kwargs):
    graph_layers = len(graph_inner_layer_sizes)
    aggregs = [concat_sum_atom_edge_feat] * (graph_layers - attn_layers) + [AtomEdgeReducer(AttnNodeEdgeAggreg(graph_hidden_size, internal_attn_size, sum_like=sum_like, zero_scalar_map=zero_attn_scalar_map))] * attn_layers
    a2g_aggregs = [atom_sum()] * (graph_layers - attn_layers) + [A2GReducer(AttnNodeEdgeAggreg(graph_hidden_size, internal_attn_size, include_attn_edges=False, sum_like=sum_like, zero_scalar_map=zero_attn_scalar_map))] * attn_layers
    b2g_aggregs = [bond_sum()] * (graph_layers - attn_layers) + [B2GReducer(AttnNodeEdgeAggreg(graph_hidden_size, internal_attn_size, include_attn_edges=False, sum_like=sum_like, zero_scalar_map=zero_attn_scalar_map))] * attn_layers
    model = get_std_model(
        graph_inner_layer_sizes=graph_inner_layer_sizes,
        fc_readout_sizes=[256]+[128]*3,
        atom_aggregators=aggregs, 
        b2g_aggregators=b2g_aggregs, 
        a2g_aggregators=a2g_aggregs, 
        graph_hidden_size=graph_hidden_size)
    return model

def get_attn_model(sum_like=False,
            internal_attn_size=16, 
            graph_hidden_size=32,
            graph_inner_layer_sizes=[[64]*3]*4, **kwargs):
    graph_layers = len(graph_inner_layer_sizes)
    attn_aggregs = [AtomEdgeReducer(AttnNodeEdgeAggreg(graph_hidden_size, internal_attn_size, sum_like=sum_like))] * graph_layers
    a2g_aggregs = [A2GReducer(AttnNodeEdgeAggreg(graph_hidden_size, internal_attn_size, include_attn_edges=False, sum_like=sum_like))] * graph_layers
    b2g_aggregs = [B2GReducer(AttnNodeEdgeAggreg(graph_hidden_size, internal_attn_size, include_attn_edges=False, sum_like=sum_like))] * graph_layers
    model = get_std_model(
        graph_inner_layer_sizes=graph_inner_layer_sizes,
        # fc_readout_sizes=fc_readout_sizes,
        atom_aggregators=attn_aggregs, 
        b2g_aggregators=b2g_aggregs, 
        a2g_aggregators=a2g_aggregs, 
        graph_hidden_size=graph_hidden_size)
    return model