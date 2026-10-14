import argparse
import sys
import torch
import os


COMMON_BEST_CONFIGS = {
    'num_layers': 2,
    'num_heads': 4,
    'dropout': 0.2,
    'max_input_sequence_length': 512,
    'channel_embedding_dim': 32,
    'patch_size': 4,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'batch_size': 200,
    'num_epochs': 100,
    'patience': 20,
}

TRAINABLE_MODELS = [
    'JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'RepeatMixer',
    'DyGFormer', 'DyGFormerF', 'DyGFormerFF', 'DyGFormerC', 'DyGFormerD', 'DyGFormerP',
    'DyGFormerR', 'DyGFormerST', 'NPTDyG', 'NPTSTDyG', 'DyGFormerWSDMP', 'NPTWSDMP',
    'DyGFormerWSDMPF', 'DyGFormerSPSE',
]


def apply_common_best_configs(args):
    """Apply the unified experiment settings to every model (model-specific args are untouched)."""
    if args.model_name not in TRAINABLE_MODELS:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")
    for name, value in COMMON_BEST_CONFIGS.items():
        setattr(args, name, value)

    # model-specific settings that must be kept
    if args.model_name == 'TGAT':
        args.time_feat_dim = 16

    # dimension checks
    attention_dim = 4 * args.channel_embedding_dim
    if args.model_name in ('DyGFormerFF', 'DyGFormerWSDMPF') and attention_dim % (2 * args.num_heads) != 0:
        raise ValueError(
            f"{args.model_name} dimension config error: 4 * channel_embedding_dim ({attention_dim}) "
            f"must be divisible by 2 * num_heads ({2 * args.num_heads})"
        )
    if args.model_name == 'DyGFormerFF' and args.channel_embedding_dim % args.num_heads != 0:
        raise ValueError(
            f"DyGFormerFF (RoPE) requires channel_embedding_dim ({args.channel_embedding_dim}) "
            f"to be divisible by num_heads ({args.num_heads})"
        )
    if args.model_name in ('DyGFormerWSDMP', 'NPTWSDMP') and attention_dim % args.num_heads != 0:
        raise ValueError(
            f"{args.model_name} dimension config error: 4 * channel_embedding_dim ({attention_dim}) "
            f"must be divisible by num_heads ({args.num_heads})"
        )


def build_run_name_suffix(args, include_batch: bool = True) -> str:
    """Common suffix appended to save/load paths (excluding seed / task prefix).

    Aligned with the naming rules of train_link_prediction, shared by training/evaluation/node classification
    to avoid path mismatches caused by each script building paths on its own.
    DyGFormerD automatically includes dim{rwpe_dim} to distinguish different RWPE precomputation dimensions.
    """
    parts = []
    if getattr(args, 'max_input_sequence_length', None) is not None:
        parts.append(f"seq{args.max_input_sequence_length}")
    if getattr(args, 'patch_size', None) is not None:
        parts.append(f"patch{args.patch_size}")
    if args.model_name in ('DyGFormerFF'):
        if getattr(args, 'max_second_hop_length', 0):
            parts.append(f"hop{args.max_second_hop_length}")
    if include_batch:
        if getattr(args, 'batch_size', None) is not None:
            parts.append(f"bs{args.batch_size}")
        if getattr(args, 'gradient_accumulation_steps', 1) > 1:
            parts.append(f"accum{args.gradient_accumulation_steps}")
    if args.model_name == 'DyGFormerD':
        parts.append(f"dim{getattr(args, 'rwpe_dim', 8)}")
    if getattr(args, 'exp_suffix', None):
        parts.append(args.exp_suffix)
    return '_'.join(parts)


def get_link_prediction_args(is_evaluation: bool = False):
    """
    Get the args for the link prediction task.
    """
    parser = argparse.ArgumentParser(description="Interface for the link prediction task")
    
    # Ensure parameters are only defined once
    existing_args = [action.dest for action in parser._actions]
    if 'channel_embedding_dim' not in existing_args:
        parser.add_argument('--channel_embedding_dim', type=int, default=32, help='dimension of each channel embedding')
    if 'num_heads' not in existing_args:
        parser.add_argument('--num_heads', type=int, default=4, help='number of heads used in attention layer')
    if 'num_kv_heads' not in existing_args:
        parser.add_argument('--num_kv_heads', type=int, help='number of key-value heads for GQA (if not specified, use MHA with num_kv_heads=num_heads)')
     # Add task_name parameter
    parser.add_argument('--task_name', type=str, default='link_prediction', choices=['link_prediction', 'node_classification'],
                        help='task to perform, either link_prediction or node_classification')
    # Other parameters...
    parser.add_argument('--model_name', type=str,                         choices=['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'EdgeBank', 'TCL', 'GraphMixer', 'RepeatMixer', 'DyGFormer', 'DyGFormerF', 'DyGFormerFF', 'DyGFormerC', 'DyGFormerD', 'DyGFormerP', 'DyGFormerR', 'DyGFormerST', 'NPTDyG', 'NPTSTDyG','DyGFormerWSDMP', 'NPTWSDMP', 'DyGFormerWSDMPF', 'DyGFormerSPSE'],
                        help='name of the model, note that EdgeBank is only applicable for evaluation')
    parser.add_argument('--spse_dist_len', type=int, default=2,
                        help='DyGFormerSPSE: K (recommended 2, minimum 2)')
    parser.add_argument('--spse_root_nodes_fact', type=float, default=0.075,
                        help='DyGFormerSPSE: R, root ratio, typically about 0.05-0.1 (default is the midpoint 0.075)')
    parser.add_argument('--spse_min_root_nodes', type=int, default=4,
                        help='DyGFormerSPSE: lower bound on the number of roots; used together with max(N\'*R)')
    parser.add_argument('--spse_max_root_nodes', type=int, default=0,
                        help='DyGFormerSPSE: upper bound on the number of roots; 0 means unlimited')
    parser.add_argument('--spse_max_bfs_depth', type=int, default=0,
                        help='DyGFormerSPSE: D_DFS (max_bfs_depth), default 0')
    parser.add_argument('--spse_max_tries_per_depth', type=int, default=1,
                        help='DyGFormerSPSE: N (max_tries_per_depth), default 1')
    parser.add_argument('--spse_use_full_spse', action='store_true', default=False,
                        help='DyGFormerSPSE: use the original get_simple_paths_count (slow); off by default, using walk counts of A and A^2 instead (fast)')
    parser.add_argument('--spse_max_subgraph_nodes', type=int, default=200,
                        help='DyGFormerSPSE: upper bound on the number of nodes in the src∪dst induced subgraph; samples exceeding it skip SPSE (bias is 0); 0 means unlimited')
    parser.add_argument('--spse_precomputed_dir', type=str, default=None,
                        help='DyGFormerSPSE: precomputed SPSE data directory (e.g. processed_data/wikipedia/spse_precomputed_ps4_dl8); '
                             'when provided, positive samples are looked up directly instead of computed online, and dist_len is read automatically from meta.json')
    # DyGFormerD RWPE-specific arguments
    parser.add_argument('--rwpe_dim', type=int, default=8,
                        help='DyGFormerD: RWPE dimension (must match the precomputation)')
    parser.add_argument('--rwpe_max_subgraph_size', type=int, default=512,
                        help='DyGFormerD: upper bound on unique nodes in the RWPE subgraph (must match the precomputation)')
    parser.add_argument('--rwpe_precomputed_dir', type=str, default=None,
                        help='DyGFormerD: precomputed RWPE directory. node_t is recommended: '
                             'processed_data/<ds>/rwpe_node_t_seq512_dim8_ms512 (both positive and negative samples look up by (node,t), misses are lazily cached); '
                             'old edge_id: rwpe_precomputed_* (only positive samples are looked up, causing distribution asymmetry)')
    parser.add_argument('--rwpe_lazy_load', action='store_true', default=False,
                        help='DyGFormerD: lazily load precomputed RWPE chunks (large datasets)')
    # DyGFormerST SPD (shortest path distance)-specific arguments
    parser.add_argument('--spd_max_subgraph_size', type=int, default=512,
                        help='DyGFormerST: upper bound on unique nodes in the SPD subgraph (must match the precomputation), defaults to max_input_sequence_length')
    parser.add_argument('--spd_precomputed_dir', type=str, default=None,
                        help='DyGFormerST: precomputed SPD directory (e.g. processed_data/<ds>/spd_precomputed_modelfn_seq512_ps4_ms512), '
                             'keyed by edge_id: covers only positive samples, misses fall back to online computation; negative samples (edge_ids=None) are always fully computed online')
    parser.add_argument('--spd_lazy_load', action='store_true', default=False,
                        help='DyGFormerST: lazily load precomputed SPD chunks (large datasets)')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='wikipedia',
                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'SocialEvo', 'uci', 'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts'])
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='number of gradient accumulation steps before optimizer update, used to simulate larger batch size')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id to use, e.g. 0')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--edge_bank_memory_mode', type=str, default='unlimited_memory', help='how memory of EdgeBank works',
                        choices=['unlimited_memory', 'time_window_memory', 'repeat_threshold_memory'])
    parser.add_argument('--time_window_mode', type=str, default='fixed_proportion', help='how to select the time window size for time window memory',
                        choices=['fixed_proportion', 'repeat_interval'])
    parser.add_argument('--patch_size', type=int, default=4, help='patch size')
    parser.add_argument('--max_input_sequence_length', type=int, default=512, help='maximal length of the input sequence of each node')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=3, help='number of runs')
    parser.add_argument('--test_interval_epochs', type=int, default=10, help='how many epochs to perform testing once')
    parser.add_argument('--negative_sample_strategy', type=str, default='random', choices=['random', 'historical', 'inductive'],
                        help='strategy for the negative edge sampling')
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')
    parser.add_argument('--exp_suffix', type=str, default=None,
                        help='optional extra suffix for save/load paths to avoid overwriting '
                             '(e.g. dim24); DyGFormerD already auto-appends dim{rwpe_dim}')
    parser.add_argument('--max_second_hop_length', type=int, default=None, help='max 2-hop neighbors per node (DyGFormerFF)')

    try:
        args = parser.parse_args()
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        parser.print_help()
        sys.exit()
    if args.task_name == 'link_prediction':
       print("Running link prediction task...")
    # Logic for the link prediction task
    elif args.task_name == 'node_classification':
        print("Running node classification task...")
    # Logic for the node classification task
    if args.model_name == 'EdgeBank':
        assert is_evaluation, 'EdgeBank is only applicable for evaluation!'

    if args.load_best_configs:
        load_link_prediction_best_configs(args=args)

    return args


def load_link_prediction_best_configs(args):
    """
    Load the best configurations for the link prediction task.
    """
    apply_common_best_configs(args)







def get_node_classification_args():
    """
    get the args for the node classification task
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the node classification task')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='wikipedia', choices=['wikipedia', 'reddit'])
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--model_name', type=str, default='DyGFormer', help='name of the model',
                        choices=['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'RepeatMixer', 'DyGFormer', 'DyGFormerF', 'DyGFormerFF', 'DyGFormerC', 'DyGFormerD', 'DyGFormerP', 'DyGFormerR', 'DyGFormerST', 'NPTDyG', 'NPTSTDyG','DyGFormerWSDMP', 'NPTWSDMP', 'DyGFormerWSDMPF', 'DyGFormerSPSE'])
    parser.add_argument('--gpu', type=int, default=0, help='gpu id to use, e.g. 0')
    parser.add_argument('--pretrained_model_name', type=str, help='Name of the pretrained model to load (e.g., DyGFormer_seed1)')
    parser.add_argument('--use_lp_pretrained_backbone', action='store_true', default=False,
                        help='whether to initialize node classification backbone from a pretrained link prediction checkpoint')
    parser.add_argument('--lp_pretrained_model_name', type=str, default=None,
                        help='explicit link prediction checkpoint name to load, e.g. DyGFormer_seed0')
    parser.add_argument('--node_cls_train_mode', type=str, default='classifier_only',
                        choices=['classifier_only', 'random_finetune'],
                        help='classifier_only freezes LP-pretrained backbone and trains the classifier only; '
                             'random_finetune trains randomly initialized backbone and classifier together')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_heads', type=int, default=4, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--patch_size', type=int, default=4, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=32, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=512, help='maximal length of the input sequence of each node')
    # NPTDyG-specific arguments
    parser.add_argument('--mask_prob', type=float, default=0.15, help='NPTDyG: masking probability')
    parser.add_argument('--augmentation_mask_prob', type=float, default=0.15, help='NPTDyG: masking probability for data augmentation')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=5, help='number of runs')
    parser.add_argument('--start_seed', type=int, default=0, help='starting seed index (inclusive), evaluates seeds [start_seed, start_seed+num_runs)')
    parser.add_argument('--test_interval_epochs', type=int, default=10, help='how many epochs to perform testing once')
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')
    parser.add_argument('--exp_suffix', type=str, default=None,
                        help='optional extra suffix for save/load paths to avoid overwriting '
                             '(e.g. dim24); DyGFormerD already auto-appends dim{rwpe_dim}')
    parser.add_argument('--max_second_hop_length', type=int, default=None, help='max 2-hop neighbors per node (DyGFormerFF)')
    # DyGFormerSPSE-specific arguments (node classification task, must match LP training)
    parser.add_argument('--spse_dist_len', type=int, default=2,
                        help='DyGFormerSPSE: SPSE distance vector length, must match LP training')
    parser.add_argument('--spse_precomputed_dir', type=str, default=None,
                        help='DyGFormerSPSE: precomputed SPSE data directory, must match LP training')
    # DyGFormerD RWPE (must match LP training)
    parser.add_argument('--rwpe_dim', type=int, default=8,
                        help='DyGFormerD: RWPE dimension, must match LP training')
    parser.add_argument('--rwpe_max_subgraph_size', type=int, default=512,
                        help='DyGFormerD: RWPE subgraph upper bound, must match LP training')
    parser.add_argument('--rwpe_precomputed_dir', type=str, default=None,
                        help='DyGFormerD: precomputed RWPE directory, must match LP training; '
                             'rwpe_node_t_* is recommended (positive and negative samples uniformly looked up by (node,t))')
    parser.add_argument('--rwpe_lazy_load', action='store_true', default=False,
                        help='DyGFormerD: lazily load precomputed RWPE chunks')
    # DyGFormerST SPD (must match LP training)
    parser.add_argument('--spd_max_subgraph_size', type=int, default=512,
                        help='DyGFormerST: SPD subgraph upper bound, must match LP training')
    parser.add_argument('--spd_precomputed_dir', type=str, default=None,
                        help='DyGFormerST: precomputed SPD directory, must match LP training')
    parser.add_argument('--spd_lazy_load', action='store_true', default=False,
                        help='DyGFormerST: lazily load precomputed SPD chunks')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='number of gradient accumulation steps (default 1 = no accumulation)')

    try:
        args = parser.parse_args()
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        parser.print_help()
        sys.exit()

    assert args.dataset_name in ['wikipedia', 'reddit'], f'Wrong value for dataset_name {args.dataset_name}!'
    if args.load_best_configs:
        load_node_classification_best_configs(args=args)

    return args


def load_node_classification_best_configs(args: argparse.Namespace):
    """
    load the best configurations for the node classification task
    :param args: argparse.Namespace
    :return:
    """
    apply_common_best_configs(args)
