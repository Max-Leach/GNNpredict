from architecture.model import DeepBDE
from architecture.feat_type_updaters import concat_sum_atom_edge_feat, bond_mean, atom_mean, bond_sum, atom_sum, A2GReducer, B2GReducer

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