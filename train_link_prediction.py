# -*- coding: utf-8 -*-
import logging
import time
import sys
import os
import argparse
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend to avoid GUI issues
import matplotlib.pyplot as plt
from models.NPTDyG import NPTDyG
from models.NPTSTDyG import NPTSTDyG
from models.NPTWSDMP import NPTWSDMP
from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
# DyGFormerD is imported lazily when needed to avoid dgl dependency issues
from models.DyGFormerP import DyGFormerP
from models.DyGFormerR import DyGFormerR
from models.DyGFormerC import DyGFormerC


from models.DyGFormerWSDMP import DyGFormerWSDMP
from models.DyGFormerWSDMPF import DyGFormerWSDMPF
from models.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args, load_link_prediction_best_configs, build_run_name_suffix
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

from models.DyGFormerSPSE import DyGFormerSPSE



import inspect
import ast


def get_model_submodule(model, idx):
    """Safely get a submodule of the model, compatible with DataParallel and Sequential"""
    if isinstance(model, torch.nn.DataParallel):
        return model.module[idx]
    else:
        return model[idx]

def setup_distributed():
    dist.init_process_group(backend='nccl', init_method='env://')

def get_link_prediction_args(argv=None):
    # Directly use the functions in utils/load_configs.py
    from utils.load_configs import get_link_prediction_args as get_args
    return get_args(is_evaluation=False)

def get_idx_data_loader(indices_list, batch_size, shuffle=False, distributed=False, sampler=None):
    if distributed:
        sampler = DistributedSampler(indices_list, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        shuffle = False  # DistributedSampler handles shuffling
    return torch.utils.data.DataLoader(
        indices_list,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler
    )

def call_classifier(classifier, src_emb, dst_emb):
    """Smartly call the classifier, automatically adapting to different forward signatures"""
    try:
        # First try the input_1, input_2 style
        return classifier(input_1=src_emb, input_2=dst_emb)
    except TypeError:
        try:
            # Then try the x style, passing in the difference
            return classifier(src_emb - dst_emb)
        except TypeError:
            try:
                # Finally try passing two arguments directly
                return classifier(src_emb, dst_emb)
            except TypeError:
                # If none of the above work, try passing the concatenated features
                combined = torch.cat([src_emb, dst_emb], dim=-1)
                return classifier(combined)

def train(rank, world_size, args):
    # Only initialize distributed training when using multiple GPUs
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        args.device = torch.device(f'cuda:{rank}')
    else:
        # Single-GPU training, use the default device directly
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get data
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)


    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)
    val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0)
    new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
    test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
    new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(train_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
        distributed=world_size > 1,
        sampler=DistributedSampler(train_data.src_node_ids, num_replicas=world_size, rank=rank) if world_size > 1 else None
    )
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False, distributed=False)
    new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False, distributed=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False, distributed=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False, distributed=False)

    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_model_name = f'{args.model_name}_seed{args.seed}'
        suffix = build_run_name_suffix(args, include_batch=True)
        if suffix:
            args.save_model_name += '_' + suffix

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # create model
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                           time_feat_dim=args.time_feat_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        elif args.model_name == 'DyGFormerF':
            from models.DyGFormerF import DyGFormerF
            
            # Validate dimension constraints
            attention_dim = 4 * args.channel_embedding_dim
            if attention_dim % (2 * args.num_heads) != 0:
                # Automatically adjust channel_embedding_dim
                old_dim = args.channel_embedding_dim
                required_divisor = 2 * args.num_heads
                remainder = attention_dim % required_divisor
                if remainder > 0:
                    args.channel_embedding_dim += (required_divisor - remainder) // 4
                logger.warning(
                    f"Automatically adjusted channel_embedding_dim from {old_dim} to {args.channel_embedding_dim} "
                    f"to satisfy the dimension requirement of differential attention"
                )
            
            dynamic_backbone = DyGFormerF(
                node_raw_features=node_raw_features, 
                edge_raw_features=edge_raw_features, 
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim, 
                channel_embedding_dim=args.channel_embedding_dim, 
                patch_size=args.patch_size,
                num_layers=args.num_layers, 
                num_heads=args.num_heads,
                num_kv_heads=getattr(args, 'num_kv_heads', None),
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length, 
                device=args.device
            )
        elif args.model_name == 'DyGFormerFF':
            from models.DyGFormerFF import DyGFormerFF
            attention_dim = 4 * args.channel_embedding_dim
            if attention_dim % (2 * args.num_heads) != 0:
                old_dim = args.channel_embedding_dim
                required_divisor = 2 * args.num_heads
                remainder = attention_dim % required_divisor
                if remainder > 0:
                    args.channel_embedding_dim += (required_divisor - remainder) // 4
                logger.warning(
                    f"Automatically adjusted channel_embedding_dim from {old_dim} to {args.channel_embedding_dim} "
                    f"to satisfy attention_dim % (2*num_heads) == 0"
                )
            if args.channel_embedding_dim % args.num_heads != 0:
                old_dim = args.channel_embedding_dim
                args.channel_embedding_dim = (args.channel_embedding_dim // args.num_heads) * args.num_heads
                if args.channel_embedding_dim == 0:
                    args.channel_embedding_dim = args.num_heads
                logger.warning(
                    f"Automatically adjusted channel_embedding_dim from {old_dim} to {args.channel_embedding_dim} "
                    f"to satisfy the RoPE requirement (channel_embedding_dim % num_heads == 0)"
                )
            dynamic_backbone = DyGFormerFF(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                num_kv_heads=getattr(args, 'num_kv_heads', None),
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device,
                max_second_hop_length=getattr(args, 'max_second_hop_length', 0),
            )
        
        
        
        
        
        elif args.model_name == 'DyGFormerWSDMP':
            dynamic_backbone = DyGFormerWSDMP(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device
            )





        elif args.model_name == 'DyGFormerWSDMPF':
            dynamic_backbone = DyGFormerWSDMPF(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device
            )
        
        
        elif args.model_name == 'DyGFormerC':
            dynamic_backbone = DyGFormerC(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        
        
        elif args.model_name == 'DyGFormerD':
            from models.DyGFormerD import DyGFormerD
            dynamic_backbone = DyGFormerD(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length,
                                         use_lpu=getattr(args, 'use_lpu', True), use_rpe=getattr(args, 'use_rpe', True),
                                         rwpe_dim=getattr(args, 'rwpe_dim', 8),
                                         rwpe_precomputed_dir=getattr(args, 'rwpe_precomputed_dir', None),
                                         rwpe_lazy_load=getattr(args, 'rwpe_lazy_load', False),
                                         max_subgraph_size=getattr(args, 'rwpe_max_subgraph_size', 512),
                                         device=args.device)
        
        elif args.model_name == 'DyGFormerST':
            from models.DyGFormerST import DyGFormerST
            dynamic_backbone = DyGFormerST(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length,
                                         max_subgraph_size=getattr(args, 'spd_max_subgraph_size', 512),
                                         spd_precomputed_dir=getattr(args, 'spd_precomputed_dir', None),
                                         spd_lazy_load=getattr(args, 'spd_lazy_load', False),
                                         device=args.device)
        
        elif args.model_name == 'DyGFormerP':
            dynamic_backbone = DyGFormerP(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        
        elif args.model_name == 'DyGFormerR':
            dynamic_backbone = DyGFormerR(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        
        


        elif args.model_name == 'DyGFormerSPSE':
            dynamic_backbone = DyGFormerSPSE(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device,
                spse_dist_len=getattr(args, 'spse_dist_len', 2),
                spse_root_nodes_fact=getattr(args, 'spse_root_nodes_fact', 0.075),
                spse_min_root_nodes=getattr(args, 'spse_min_root_nodes', 4),
                spse_max_root_nodes=getattr(args, 'spse_max_root_nodes', 0),
                spse_max_bfs_depth=getattr(args, 'spse_max_bfs_depth', 0),
                spse_max_tries_per_depth=getattr(args, 'spse_max_tries_per_depth', 1),
                spse_use_adjacency_walk_counts=not getattr(args, 'spse_use_full_spse', False),
                spse_max_subgraph_nodes=getattr(args, 'spse_max_subgraph_nodes', 200),
                spse_precomputed_dir=getattr(args, 'spse_precomputed_dir', None),
                spse_lazy_load=(args.dataset_name == 'UNvote'),
            )


        elif args.model_name == 'NPTDyG':
            dynamic_backbone = NPTDyG(
                node_raw_features=node_raw_features,edge_raw_features=edge_raw_features,neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,channel_embedding_dim=args.channel_embedding_dim,num_layers=args.num_layers,
                num_heads=args.num_heads,dropout=args.dropout,max_input_sequence_length=args.max_input_sequence_length,device=args.device)
        elif args.model_name == 'NPTSTDyG':
            dynamic_backbone = NPTSTDyG(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device)
        elif args.model_name == 'NPTWSDMP':
            dynamic_backbone = NPTWSDMP(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device
            )
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")

        # Create link_predictor - use different input dimensions for different models
        if args.model_name in ['DyGFormer', 'DyGFormerF', 'DyGFormerFF', 'DyGFormerC', 'DyGFormerD', 'DyGFormerP', 'DyGFormerR', 'DyGFormerST', 'DyGFormerWSDMP', 'NPTWSDMP', 'DyGFormerWSDMPF', 'DyGFormerSPSE']:            # The output_layer of the DyGFormer family (including DyGFormerF) maps features to node_feat_dim
            input_dim1 = dynamic_backbone.node_feat_dim
            input_dim2 = dynamic_backbone.node_feat_dim
            link_predictor = MergeLayer(
                input_dim1=input_dim1,
                input_dim2=input_dim2,
                hidden_dim=128,
                output_dim=1
            )
        elif args.model_name in ['NPTDyG', 'NPTSTDyG']:
            link_predictor = MergeLayer(
                input_dim1=dynamic_backbone.node_feat_dim,
                input_dim2=dynamic_backbone.node_feat_dim,
                hidden_dim=128,
                output_dim=1
             )
        else:
            # Other models use 3 channels by default
            link_predictor = MergeLayer(
                input_dim1=dynamic_backbone.node_feat_dim,
                input_dim2=dynamic_backbone.node_feat_dim,
                hidden_dim=128,
                output_dim=1
             )

        # Combine the dynamic backbone and the link predictor into one model
        model = nn.Sequential(dynamic_backbone, link_predictor)
        model = model.to(args.device)

        # Show model parameter size statistics
        num_params = get_parameter_sizes(model)
        param_size_bytes = num_params * 4  # Assume float32, 4 bytes per parameter
        param_size_kb = param_size_bytes / 1024
        param_size_mb = param_size_kb / 1024
        logger.info(f'Model: {args.model_name}, Total trainable parameters: {num_params:,}')
        logger.info(f'Model size: {param_size_bytes:,.0f} B, {param_size_kb:.4f} KB, {param_size_mb:.4f} MB')
        print(f"\n{'='*80}")
        print(f"Model Parameter Statistics:")
        print(f"{'='*80}")
        print(f"Model: {args.model_name}")
        print(f"Total trainable parameters: {num_params:,}")
        print(f"Model size: {param_size_bytes:,.0f} B ({param_size_kb:.4f} KB / {param_size_mb:.4f} MB)")
        print(f"{'='*80}\n")

        # Choose whether to use multi-GPU training based on the number of GPUs
        if world_size > 1:
            if dist.is_initialized():
                model = DDP(model, device_ids=[rank], output_device=rank)
            else:
                model = DP(model)
            print(f"Using {world_size} GPUs! DataParallel/DistributedDataParallel enabled.")
        else:
            print('Single GPU or CPU, DataParallel not used.')
        print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))
        print('torch.cuda.device_count():', torch.cuda.device_count())
        print('model is DataParallel:', isinstance(model, torch.nn.DataParallel))

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        loss_func = nn.BCEWithLogitsLoss()

        # Initialize loss history
        epoch_train_losses = []
        epoch_val_losses = []
        epoch_new_node_val_losses = []

        for epoch in range(args.num_epochs):

            if args.model_name == 'DyGFormerFF':
                epoch_start_time = time.time()
                dyff_batch_peak_allocated_mb_list = []
                dyff_batch_peak_reserved_mb_list = []
                if torch.cuda.is_available():
                    torch.cuda.synchronize(args.device)
                    torch.cuda.reset_peak_memory_stats(args.device)
                backbone_module = get_model_submodule(model, 0)
                if hasattr(backbone_module, 'reset_debug_input_shape_stats'):
                    backbone_module.reset_debug_input_shape_stats()

            model.train()
            if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'DyGFormerF', 'DyGFormerFF', 'NPTDyG', 'NPTSTDyG', 'NPTWSDMP', 'DyGFormerST', 'DyGFormerWSDMP', 'DyGFormerWSDMPF', 'DyGFormerSPSE']:
                # training, only use training graph
                get_model_submodule(model, 0).set_neighbor_sampler(train_neighbor_sampler)
            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # reinitialize memory of memory-based models at the start of each epoch
                get_model_submodule(model, 0).memory_bank.__init_memory_bank__()

            # store train losses and metrics
            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, 
                                            desc=f'Epoch {epoch + 1}/{args.num_epochs}', 
                                            ncols=120, 
                                            leave=True, 
                                            ascii=True)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                if args.model_name == 'DyGFormerFF' and torch.cuda.is_available():
                    torch.cuda.synchronize(args.device)
                    torch.cuda.reset_peak_memory_stats(args.device)

                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                    train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

                # Make sure all input tensors are on the main GPU
                batch_src_node_ids = torch.LongTensor(batch_src_node_ids).to(args.device)
                batch_dst_node_ids = torch.LongTensor(batch_dst_node_ids).to(args.device)
                batch_node_interact_times = torch.FloatTensor(batch_node_interact_times).to(args.device)

                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

                # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
                # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
                if args.model_name in ['TGAT', 'CAWN', 'TCL']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch = {
                        'src_node_ids': batch_src_node_ids,
                        'dst_node_ids': batch_dst_node_ids,
                        'node_interact_times': batch_node_interact_times
                    }
                    batch_neg = {
                        'src_node_ids': batch_neg_src_node_ids,
                        'dst_node_ids': batch_neg_dst_node_ids,
                        'node_interact_times': batch_node_interact_times
                    }
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)
                elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # note that negative nodes do not change the memories while the positive nodes change the memories,
                    # we need to first compute the embeddings of negative nodes for memory-based models
                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch = {
                        'src_node_ids': batch_neg_src_node_ids,
                        'dst_node_ids': batch_neg_dst_node_ids,
                        'node_interact_times': batch_node_interact_times,
                        'edge_ids': None,
                        'edges_are_positive': False,
                        'num_neighbors': args.num_neighbors
                    }
                    batch_neg = {
                        'src_node_ids': batch_src_node_ids,
                        'dst_node_ids': batch_dst_node_ids,
                        'node_interact_times': batch_node_interact_times,
                        'edge_ids': batch_edge_ids,
                        'edges_are_positive': True,
                        'num_neighbors': args.num_neighbors
                    }
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          edge_ids=None,
                                                                          edges_are_positive=False,
                                                                          num_neighbors=args.num_neighbors)

                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          edge_ids=batch_edge_ids,
                                                                          edges_are_positive=True,
                                                                          num_neighbors=args.num_neighbors)
                elif args.model_name in ['GraphMixer']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch = {
                        'src_node_ids': batch_src_node_ids,
                        'dst_node_ids': batch_dst_node_ids,
                        'node_interact_times': batch_node_interact_times,
                        'num_neighbors': args.num_neighbors,
                        'time_gap': args.time_gap
                    }
                    batch_neg = {
                        'src_node_ids': batch_neg_src_node_ids,
                        'dst_node_ids': batch_neg_dst_node_ids,
                        'node_interact_times': batch_node_interact_times,
                        'num_neighbors': args.num_neighbors,
                        'time_gap': args.time_gap
                    }
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors,
                                                                          time_gap=args.time_gap)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors,
                                                                          time_gap=args.time_gap)
                elif args.model_name in ['DyGFormer', 'DyGFormerF', 'DyGFormerFF', 'DyGFormerC', 'DyGFormerD', 'DyGFormerP', 'DyGFormerR', 'NPTDyG', 'NPTSTDyG', 'NPTWSDMP', 'DyGFormerST', 'DyGFormerWSDMP', 'DyGFormerWSDMPF', 'DyGFormerSPSE']:
                    # Decide whether to pass the num_neighbors argument based on the model type
                    if args.model_name == 'DyGFormerSPSE':
                        # Pass batch_edge_ids for positive samples to support precomputed SPSE lookup
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                                src_node_ids=batch_src_node_ids,
                                dst_node_ids=batch_dst_node_ids,
                                node_interact_times=batch_node_interact_times,
                                edge_ids=batch_edge_ids,
                            )
                        # Negative samples have no real edge_id, pass None (bias is 0 or computed online)
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                            get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                                src_node_ids=batch_neg_src_node_ids,
                                dst_node_ids=batch_neg_dst_node_ids,
                                node_interact_times=batch_node_interact_times,
                                edge_ids=None,
                            )
                    elif args.model_name == 'DyGFormerD':
                        # node_t precomputation: both positive and negative samples look up by (node,t) (edge_ids can be ignored); the old edge_id mode still uses edge_ids/None
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                                src_node_ids=batch_src_node_ids,
                                dst_node_ids=batch_dst_node_ids,
                                node_interact_times=batch_node_interact_times,
                                edge_ids=batch_edge_ids,
                            )
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                            get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                                src_node_ids=batch_neg_src_node_ids,
                                dst_node_ids=batch_neg_dst_node_ids,
                                node_interact_times=batch_node_interact_times,
                                edge_ids=None,
                            )
                    elif args.model_name == 'DyGFormerST':
                        # edge_id precomputed SPD: positive samples pass batch_edge_ids for lookup (misses computed online),
                        # negative samples pass None, both src/dst sides computed fully online (same as the old edge_id mode of DyGFormerD)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                                src_node_ids=batch_src_node_ids,
                                dst_node_ids=batch_dst_node_ids,
                                node_interact_times=batch_node_interact_times,
                                edge_ids=batch_edge_ids,
                            )
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                            get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                                src_node_ids=batch_neg_src_node_ids,
                                dst_node_ids=batch_neg_dst_node_ids,
                                node_interact_times=batch_node_interact_times,
                                edge_ids=None,
                            )
                    elif args.model_name in ['DyGFormer', 'DyGFormerF', 'DyGFormerFF', 'DyGFormerC', 'DyGFormerP', 'DyGFormerR', 'NPTDyG', 'NPTSTDyG', 'NPTWSDMP', 'DyGFormerWSDMP', 'DyGFormerWSDMPF']:
                        # The DyGFormer family, NPTDyG, NPTSTDyG and NPTWSDMP do not accept the num_neighbors argument, only the three basic arguments
                        # (DyGFormerST has its own branch and no longer goes through here)
                        # The DyGFormer family returns only 2 values and does not collect attention weights
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times)

                        # get temporal embedding of negative source and negative destination nodes
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                            get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times)
                    else:
                        # Other models (including DyGFormer) may return 2 or 3 values
                        _out_pos = get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)
                        if len(_out_pos) == 3:
                            batch_src_node_embeddings, batch_dst_node_embeddings, _ = _out_pos
                        else:
                            batch_src_node_embeddings, batch_dst_node_embeddings = _out_pos

                        # get temporal embedding of negative source and negative destination nodes
                        _out_neg = get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)
                        if len(_out_neg) == 3:
                            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, _ = _out_neg
                        else:
                            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = _out_neg
                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")

                # Compute logits first (no sigmoid, let BCEWithLogitsLoss handle it internally)
                positive_probabilities = call_classifier(get_model_submodule(model, 1), batch_src_node_embeddings, batch_dst_node_embeddings).squeeze(dim=-1)
                negative_probabilities = call_classifier(get_model_submodule(model, 1), batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings).squeeze(dim=-1)

                # Then align shapes
                if positive_probabilities.dim() == 2 and negative_probabilities.dim() == 2:
                    min_seq_len = min(positive_probabilities.size(1), negative_probabilities.size(1))
                    positive_probabilities = positive_probabilities[:, :min_seq_len]
                    negative_probabilities = negative_probabilities[:, :min_seq_len]

                predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

                loss = loss_func(input=predicts, target=labels)
                loss = loss.mean()  # Average the loss in the multi-GPU setting

                train_losses.append(loss.item())

                # Use probabilities when computing metrics (convert logits via sigmoid)
                predicts_sigmoid = predicts.sigmoid()

                # Support gradient accumulation (effective when gradient_accumulation_steps > 1)
                train_metrics.append(get_link_prediction_metrics(predicts=predicts_sigmoid, labels=labels))
                accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
                if accumulation_steps > 1:
                    # Zero gradients on the first mini-batch, then accumulate
                    if batch_idx % accumulation_steps == 0:
                        optimizer.zero_grad()
                    scaled_loss = loss / accumulation_steps
                    scaled_loss.backward()
                    # Only update parameters once enough steps are accumulated or the last batch is reached
                    is_last_batch = (batch_idx + 1) == len(train_idx_data_loader)
                    if (batch_idx + 1) % accumulation_steps == 0 or is_last_batch:
                        optimizer.step()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if args.model_name == 'DyGFormerFF' and torch.cuda.is_available():
                    torch.cuda.synchronize(args.device)
                    batch_peak_allocated_mb = torch.cuda.max_memory_allocated(args.device) / 1024 ** 2
                    batch_peak_reserved_mb = torch.cuda.max_memory_reserved(args.device) / 1024 ** 2
                    dyff_batch_peak_allocated_mb_list.append(batch_peak_allocated_mb)
                    dyff_batch_peak_reserved_mb_list.append(batch_peak_reserved_mb)

                # Update the progress bar description with more concise information
                if batch_idx % 10 == 0 or batch_idx == len(train_idx_data_loader) - 1:
                    train_idx_data_loader_tqdm.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Batch': f'{batch_idx + 1}/{len(train_idx_data_loader)}'
                    })

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                    get_model_submodule(model, 0).memory_bank.detach_memory_bank()

            # --------------  Record DyGFormer pairwise averages after each training epoch --------------
            if args.model_name == 'DyGFormer':
                backbone_module = get_model_submodule(model, 0)
                if hasattr(backbone_module, 'get_avg_pairwise_metrics'):
                    avg_l2, avg_cos = backbone_module.get_avg_pairwise_metrics()
                    if avg_l2 is not None and avg_cos is not None:
                        logger.info(f'PAIRWISE(train) L2_mean={avg_l2:.5f}, Cos_mean={avg_cos:.5f}')
                        # Save to backbone_module for use in the final result_json
                        backbone_module.latest_avg_pairwise_l2 = avg_l2
                        backbone_module.latest_avg_pairwise_cos = avg_cos
                        print(f"Training Pairwise L2 mean: {avg_l2:.4f}, Cosine mean: {avg_cos:.4f}")
            # -------------------------------------------------------------------------

            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # backup memory bank after training so it can be used for new validation nodes
                train_backup_memory_bank = get_model_submodule(model, 0).memory_bank.backup_memory_bank()

            if args.model_name == 'DyGFormerFF' and torch.cuda.is_available():
                backbone_module = get_model_submodule(model, 0)
                if hasattr(backbone_module, 'get_debug_input_shape_summary'):
                    input_shape_summary = backbone_module.get_debug_input_shape_summary()
                    if input_shape_summary is not None and rank == 0:
                        logger.info(input_shape_summary)
                        print(input_shape_summary)

                torch.cuda.synchronize(args.device)
                avg_batch_peak_allocated_mb = float(np.mean(dyff_batch_peak_allocated_mb_list)) if dyff_batch_peak_allocated_mb_list else 0.0
                max_batch_peak_allocated_mb = float(np.max(dyff_batch_peak_allocated_mb_list)) if dyff_batch_peak_allocated_mb_list else 0.0
                avg_batch_peak_reserved_mb = float(np.mean(dyff_batch_peak_reserved_mb_list)) if dyff_batch_peak_reserved_mb_list else 0.0
                max_batch_peak_reserved_mb = float(np.max(dyff_batch_peak_reserved_mb_list)) if dyff_batch_peak_reserved_mb_list else 0.0
                train_time = time.time() - epoch_start_time
                memory_msg = (
                    f'[DyGFormerFF][Epoch {epoch + 1}] train CUDA memory/time: '
                    f'avg_batch_peak_allocated={avg_batch_peak_allocated_mb:.2f} MB, '
                    f'max_batch_peak_allocated={max_batch_peak_allocated_mb:.2f} MB, '
                    f'avg_batch_peak_reserved={avg_batch_peak_reserved_mb:.2f} MB, '
                    f'max_batch_peak_reserved={max_batch_peak_reserved_mb:.2f} MB, '
                    f'train_time={train_time:.2f}s, '
                    f'patch_size={args.patch_size}, '
                    f'max_second_hop_length={getattr(args, "max_second_hop_length", None)}'
                )
                if rank == 0:
                    logger.info(memory_msg)
                    print(memory_msg)

            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap)

            # Compute validation pairwise metrics (DyGFormer only) - temporarily removed

            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # backup memory bank after validating so it can be used for testing nodes (since test edges are strictly later in time than validation edges)
                val_backup_memory_bank = get_model_submodule(model, 0).memory_bank.backup_memory_bank()

                # reload training memory bank for new validation nodes
                get_model_submodule(model, 0).memory_bank.reload_memory_bank(train_backup_memory_bank)

            new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                       model=model,
                                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                                       evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                       evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                       evaluate_data=new_node_val_data,
                                                                                       loss_func=loss_func,
                                                                                       num_neighbors=args.num_neighbors,
                                                                                       time_gap=args.time_gap)

            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # reload validation memory bank for new testing nodes
                get_model_submodule(model, 0).memory_bank.reload_memory_bank(val_backup_memory_bank)

            if rank == 0:
                logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
                for metric_name in train_metrics[0].keys():
                    logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
            
            # Print epoch summary
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{args.num_epochs} Summary:")
            print(f"{'='*80}")
            print(f"Train Loss: {np.mean(train_losses):.4f}")
            for metric_name in train_metrics[0].keys():
                print(f"Train {metric_name}: {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}")

            # Per-stage timing statistics for DyGFormerSPSE
            _backbone = get_model_submodule(model, 0)
            if args.model_name == 'DyGFormerSPSE' and hasattr(_backbone, '_timers'):
                _t = _backbone._timers
                _total = sum(_t.values())
                if _total > 0:
                    print(f"\n[DyGFormerSPSE] Epoch {epoch + 1} forward timing (total={_total:.2f}s):")
                    for _k, _v in sorted(_t.items(), key=lambda x: -x[1]):
                        print(f"  {_k:25s}: {_v:7.2f}s  ({100*_v/_total:5.1f}%)")
                _backbone.reset_timers()
            
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            print(f"Validation Loss: {np.mean(val_losses):.4f}")
            for metric_name in val_metrics[0].keys():
                val_metric_value = np.mean([val_metric[metric_name] for val_metric in val_metrics])
                logger.info(f'validate {metric_name}, {val_metric_value:.4f}')
                print(f"Validation {metric_name}: {val_metric_value:.4f}")
            
            logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            print(f"New Node Validation Loss: {np.mean(new_node_val_losses):.4f}")
            for metric_name in new_node_val_metrics[0].keys():
                new_node_val_metric_value = np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics])
                logger.info(f'new node validate {metric_name}, {new_node_val_metric_value:.4f}')
                print(f"New Node Validation {metric_name}: {new_node_val_metric_value:.4f}")
            print(f"{'='*80}\n")

            # Record epoch-level loss
            epoch_train_losses.append(np.mean(train_losses))
            epoch_val_losses.append(np.mean(val_losses))
            epoch_new_node_val_losses.append(np.mean(new_node_val_losses))

            # Plot loss curves - disabled
            # plt.figure(figsize=(12, 5))
            # 
            # # Plot training loss
            # plt.subplot(1, 2, 1)
            # plt.plot(range(1, len(epoch_train_losses) + 1), epoch_train_losses, 'b-', label='Train Loss', linewidth=2, marker='o')
            # plt.xlabel('Epoch', fontsize=12)
            # plt.ylabel('Loss', fontsize=12)
            # plt.title('Training Loss Curve', fontsize=14)
            # plt.legend()
            # plt.grid(True, alpha=0.3)
            # 
            # # Plot validation loss
            # plt.subplot(1, 2, 2)
            # plt.plot(range(1, len(epoch_val_losses) + 1), epoch_val_losses, 'g-', label='Val Loss', linewidth=2, marker='s')
            # plt.plot(range(1, len(epoch_new_node_val_losses) + 1), epoch_new_node_val_losses, 'r--', label='New Node Val Loss', linewidth=2, marker='^')
            # plt.xlabel('Epoch', fontsize=12)
            # plt.ylabel('Loss', fontsize=12)
            # plt.title('Validation Loss Curve', fontsize=14)
            # plt.legend()
            # plt.grid(True, alpha=0.3)
            # 
            # plt.tight_layout()
            # 
            # # Save loss curve figure
            # loss_curve_folder = f"./loss_curves/{args.model_name}/{args.dataset_name}/"
            # os.makedirs(loss_curve_folder, exist_ok=True)
            # plt.savefig(f"{loss_curve_folder}/{args.save_model_name}_epoch{epoch+1}.png", dpi=150, bbox_inches='tight')
            # plt.close()
            # 
            # # Also save loss data to a JSON file
            # loss_data = {
            #     'epoch': list(range(1, len(epoch_train_losses) + 1)),
            #     'train_loss': [float(x) for x in epoch_train_losses],
            #     'val_loss': [float(x) for x in epoch_val_losses],
            #     'new_node_val_loss': [float(x) for x in epoch_new_node_val_losses]
            # }
            # with open(f"{loss_curve_folder}/{args.save_model_name}_loss_data.json", 'w') as f:
            #     json.dump(loss_data, f, indent=4)
            # 
            # logger.info(f'Loss curve saved to {loss_curve_folder}/{args.save_model_name}_epoch{epoch+1}.png')

            # Show attention weight statistics (attention weight collection is disabled)
            # if args.model_name in ['DyGFormer']:
            #     print(f"\n{'='*60}")
            #     print(f"Epoch {epoch + 1} - {args.model_name} attention weight statistics")
            #     print(f"{'='*60}")
            #     
            #     # Collect attention weight statistics
            #     try:
            #         # Use all validation data to collect attention weight statistics
            #         all_attention_stats = []
            #         
            #         model.eval()
            #         with torch.no_grad():
            #             # Use the validation data loader to batch-process all validation data
            #             print("Collecting attention weight statistics...")
            #             for batch_idx, val_data_indices in enumerate(val_idx_data_loader):
            #                 val_data_indices = val_data_indices.numpy()
            #                 batch_src_node_ids = val_data.src_node_ids[val_data_indices]
            #                 batch_dst_node_ids = val_data.dst_node_ids[val_data_indices]
            #                 batch_node_interact_times = val_data.node_interact_times[val_data_indices]
            #                 
            #                 if args.model_name == 'DyGFormer':
            #                     _out = get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
            #                         src_node_ids=batch_src_node_ids,
            #                         dst_node_ids=batch_dst_node_ids,
            #                         node_interact_times=batch_node_interact_times
            #                     )
            #                     
            #                     if len(_out) == 3:
            #                         _, _, extra = _out
            #                         if extra and 'attn_weights' in extra:
            #                             all_attention_stats.append(extra['attn_weights'])
            #         
            #         # Aggregate attention weight statistics across all samples
            #         if all_attention_stats:
            #             category_sums = {'self': 0.0, 'key': 0.0, 'other': 0.0, 'pad': 0.0}
            #             count = 0
            #             
            #             for batch_summary in all_attention_stats:
            #                 if isinstance(batch_summary, dict):
            #                     # Handle the DyGFormer data structure
            #                     if args.model_name == 'DyGFormer':
            #                         for side in ['src', 'dst']:
            #                             if side in batch_summary:
            #                                 side_data = batch_summary[side]
            #                                 if isinstance(side_data, dict) and 'A' in side_data and isinstance(side_data['A'], dict) and 'share' in side_data['A']:
            #                                     share_data = side_data['A']['share']
            #                                     if isinstance(share_data, dict):
            #                                         for cat, weight in share_data.items():
            #                                             if cat in category_sums:
            #                                                 category_sums[cat] += float(weight)
            #                                         count += 1
            #         
            #         # Show statistics
            #         if count > 0:
            #             # Normalize
            #             total = sum(category_sums.values())
            #             if total > 0:
            #                 print(f"Attention weight statistics based on {len(all_attention_stats)} batches:")
            #                 print("Average attention weight distribution (weights of the four node types sum to 1):")
            #                 for cat in ['self', 'key', 'other', 'pad']:
            #                     weight = category_sums[cat] / total
            #                     print(f"  {cat}: {weight:.3f} ({weight*100:.1f}%)")
            #             else:
            #                 print("No valid attention weight data collected")
            #         else:
            #             print("Failed to aggregate attention weight data")
            #     else:
            #         print("No attention weight data collected")
            #         
            #     except Exception as e:
            #         print(f"Error while getting attention weight statistics: {str(e)}")
            #     
            #     print(f"{'='*60}\n")

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                           model=model,
                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                           evaluate_idx_data_loader=test_idx_data_loader,
                                                                           evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                           evaluate_data=test_data,
                                                                           loss_func=loss_func,
                                                                           num_neighbors=args.num_neighbors,
                                                                           time_gap=args.time_gap)

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # reload validation memory bank for new testing nodes
                    get_model_submodule(model, 0).memory_bank.reload_memory_bank(val_backup_memory_bank)

                new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                             model=model,
                                                                                             neighbor_sampler=full_neighbor_sampler,
                                                                                             evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                             evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                             evaluate_data=new_node_test_data,
                                                                                             loss_func=loss_func,
                                                                                             num_neighbors=args.num_neighbors,
                                                                                             time_gap=args.time_gap)

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # reload validation memory bank for testing nodes or saving models
                    # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                    get_model_submodule(model, 0).memory_bank.reload_memory_bank(val_backup_memory_bank)

                logger.info(f'test loss: {np.mean(test_losses):.4f}')
                print(f"\n{'='*60}")
                print(f"Test Results (Epoch {epoch + 1}):")
                print(f"{'='*60}")
                print(f"Test Loss: {np.mean(test_losses):.4f}")
                for metric_name in test_metrics[0].keys():
                    test_metric_value = np.mean([test_metric[metric_name] for test_metric in test_metrics])
                    logger.info(f'test {metric_name}, {test_metric_value:.4f}')
                    print(f"Test {metric_name}: {test_metric_value:.4f}")
                
                logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
                print(f"New Node Test Loss: {np.mean(new_node_test_losses):.4f}")
                for metric_name in new_node_test_metrics[0].keys():
                    new_node_test_metric_value = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
                    logger.info(f'new node test {metric_name}, {new_node_test_metric_value:.4f}')
                    print(f"New Node Test {metric_name}: {new_node_test_metric_value:.4f}")
                print(f"{'='*60}\n")

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
            early_stop = early_stopping.step(val_metric_indicator, model)

            # Persist newly computed (node,t) RWPE from negatives / misses
            if args.model_name == 'DyGFormerD':
                backbone = get_model_submodule(model, 0)
                if hasattr(backbone, 'flush_rwpe_lazy_cache'):
                    backbone.flush_rwpe_lazy_cache()

            if early_stop:
                break

        if args.model_name == 'DyGFormerD':
            backbone = get_model_submodule(model, 0)
            if hasattr(backbone, 'flush_rwpe_lazy_cache'):
                backbone.flush_rwpe_lazy_cache()

        # Plot the final complete loss curves (after training) - disabled
        # if len(epoch_train_losses) > 0:
        #     plt.figure(figsize=(15, 5))
        # 
        #     # Plot training loss
        #     plt.subplot(1, 3, 1)
        #     plt.plot(range(1, len(epoch_train_losses) + 1), epoch_train_losses, 'b-', label='Train Loss', linewidth=2, marker='o')
        #     plt.xlabel('Epoch', fontsize=12)
        #     plt.ylabel('Loss', fontsize=12)
        #     plt.title('Training Loss Curve (Final)', fontsize=14)
        #     plt.legend()
        #     plt.grid(True, alpha=0.3)
        # 
        #     # Plot validation loss
        #     plt.subplot(1, 3, 2)
        #     plt.plot(range(1, len(epoch_val_losses) + 1), epoch_val_losses, 'g-', label='Val Loss', linewidth=2, marker='s')
        #     plt.plot(range(1, len(epoch_new_node_val_losses) + 1), epoch_new_node_val_losses, 'r--', label='New Node Val Loss', linewidth=2, marker='^')
        #     plt.xlabel('Epoch', fontsize=12)
        #     plt.ylabel('Loss', fontsize=12)
        #     plt.title('Validation Loss Curve (Final)', fontsize=14)
        #     plt.legend()
        #     plt.grid(True, alpha=0.3)
        # 
        #     # Plot all losses in one figure
        #     plt.subplot(1, 3, 3)
        #     plt.plot(range(1, len(epoch_train_losses) + 1), epoch_train_losses, 'b-', label='Train Loss', linewidth=2, marker='o')
        #     plt.plot(range(1, len(epoch_val_losses) + 1), epoch_val_losses, 'g-', label='Val Loss', linewidth=2, marker='s')
        #     plt.plot(range(1, len(epoch_new_node_val_losses) + 1), epoch_new_node_val_losses, 'r--', label='New Node Val Loss', linewidth=2, marker='^')
        #     plt.xlabel('Epoch', fontsize=12)
        #     plt.ylabel('Loss', fontsize=12)
        #     plt.title('All Loss Curves (Final)', fontsize=14)
        #     plt.legend()
        #     plt.grid(True, alpha=0.3)
        # 
        #     plt.tight_layout()
        # 
        #     # Save the final loss curve figure
        #     loss_curve_folder = f"./loss_curves/{args.model_name}/{args.dataset_name}/"
        #     os.makedirs(loss_curve_folder, exist_ok=True)
        #     plt.savefig(f"{loss_curve_folder}/{args.save_model_name}_final.png", dpi=200, bbox_inches='tight')
        #     plt.close()
        # 
        #     logger.info(f'Final loss curve saved to {loss_curve_folder}/{args.save_model_name}_final.png')
        #     print(f"\n{'='*80}")
        #     print(f"Loss curves saved to: {loss_curve_folder}")
        #     print(f"- Epoch-wise curves: {args.save_model_name}_epoch*.png")
        #     print(f"- Final curve: {args.save_model_name}_final.png")
        #     print(f"- Loss data: {args.save_model_name}_loss_data.json")
        #     print(f"{'='*80}\n")

        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')
        print(f"\n{'='*100}")
        print(f"FINAL EVALUATION RESULTS - {args.model_name} on {args.dataset_name}")
        print(f"{'='*100}")

        # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap)

            new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                       model=model,
                                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                                       evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                       evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                       evaluate_data=new_node_val_data,
                                                                                       loss_func=loss_func,
                                                                                       num_neighbors=args.num_neighbors,
                                                                                       time_gap=args.time_gap)

        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # the memory in the best model has seen the validation edges, we need to backup the memory for new testing nodes
            val_backup_memory_bank = model.module[0].memory_bank.backup_memory_bank()

        test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                   model=model,
                                                                   neighbor_sampler=full_neighbor_sampler,
                                                                   evaluate_idx_data_loader=test_idx_data_loader,
                                                                   evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                   evaluate_data=test_data,
                                                                   loss_func=loss_func,
                                                                   num_neighbors=args.num_neighbors,
                                                                   time_gap=args.time_gap)

        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # reload validation memory bank for new testing nodes
            get_model_submodule(model, 0).memory_bank.reload_memory_bank(val_backup_memory_bank)

        new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                     model=model,
                                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                                     evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                     evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                     evaluate_data=new_node_test_data,
                                                                                     loss_func=loss_func,
                                                                                     num_neighbors=args.num_neighbors,
                                                                                     time_gap=args.time_gap)
        # store the evaluation metrics at the current run
        val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            print(f"\nValidation Results:")
            print(f"Validation Loss: {np.mean(val_losses):.4f}")
            for metric_name in val_metrics[0].keys():
                average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
                logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
                print(f"Validation {metric_name}: {average_val_metric:.4f}")
                val_metric_dict[metric_name] = average_val_metric

            logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            print(f"\nNew Node Validation Results:")
            print(f"New Node Validation Loss: {np.mean(new_node_val_losses):.4f}")
            for metric_name in new_node_val_metrics[0].keys():
                average_new_node_val_metric = np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics])
                logger.info(f'new node validate {metric_name}, {average_new_node_val_metric:.4f}')
                print(f"New Node Validation {metric_name}: {average_new_node_val_metric:.4f}")
                new_node_val_metric_dict[metric_name] = average_new_node_val_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        print(f"\nTest Results:")
        print(f"Test Loss: {np.mean(test_losses):.4f}")
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            print(f"Test {metric_name}: {average_test_metric:.4f}")
            test_metric_dict[metric_name] = average_test_metric

        logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
        print(f"\nNew Node Test Results:")
        print(f"New Node Test Loss: {np.mean(new_node_test_losses):.4f}")
        for metric_name in new_node_test_metrics[0].keys():
            average_new_node_test_metric = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
            logger.info(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
            print(f"New Node Test {metric_name}: {average_new_node_test_metric:.4f}")
            new_node_test_metric_dict[metric_name] = average_new_node_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')
        print(f"\nRun {run + 1} completed in {single_run_time:.2f} seconds.")
        print(f"{'='*100}\n")

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_metric_all_runs.append(val_metric_dict)
            new_node_val_metric_all_runs.append(new_node_val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)
        new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            result_json = {
                "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
                "new node validate metrics": {metric_name: f'{new_node_val_metric_dict[metric_name]:.4f}' for metric_name in new_node_val_metric_dict},
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
                "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
            }
        else:
            result_json = {
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
                "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
            }
        # ----------------------------------------------------------
        # DyGFormer only: record average pairwise L2 and Cos metrics
        # ----------------------------------------------------------
        if args.model_name == 'DyGFormer':
            backbone_module = get_model_submodule(model, 0)
            avg_l2  = getattr(backbone_module, 'latest_avg_pairwise_l2', None)
            avg_cos = getattr(backbone_module, 'latest_avg_pairwise_cos', None)
            val_avg_l2  = getattr(backbone_module, 'latest_val_pairwise_l2', None)
            val_avg_cos = getattr(backbone_module, 'latest_val_pairwise_cos', None)
            if avg_l2 is not None and avg_cos is not None:
                result_json["pairwise metrics"] = {
                    "avg_l2": f"{avg_l2:.4f}",
                    "avg_cos": f"{avg_cos:.4f}"
                }
            if val_avg_l2 is not None and val_avg_cos is not None:
                result_json["pairwise metrics (validation)"] = {
                    "avg_l2": f"{val_avg_l2:.4f}",
                    "avg_cos": f"{val_avg_cos:.4f}"
                }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)

        # Save the model
        if rank == 0:
            torch.save(model.state_dict(), os.path.join(save_model_folder, 'model.pth'))

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')
    
    print(f"\n{'='*120}")
    print(f"FINAL SUMMARY - {args.model_name} on {args.dataset_name} ({args.num_runs} runs)")
    print(f"{'='*120}")

    if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
        print(f"\nValidation Metrics:")
        print(f"{'-'*50}")
        for metric_name in val_metric_all_runs[0].keys():
            values = [val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            logger.info(f'validate {metric_name}, {values}')
            logger.info(f'average validate {metric_name}, {mean_val:.4f} ± {std_val:.4f}')
            print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")

        print(f"\nNew Node Validation Metrics:")
        print(f"{'-'*50}")
        for metric_name in new_node_val_metric_all_runs[0].keys():
            values = [new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            logger.info(f'new node validate {metric_name}, {values}')
            logger.info(f'average new node validate {metric_name}, {mean_val:.4f} ± {std_val:.4f}')
            print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")

    print(f"\nTest Metrics:")
    print(f"{'-'*50}")
    for metric_name in test_metric_all_runs[0].keys():
        values = [test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        logger.info(f'test {metric_name}, {values}')
        logger.info(f'average test {metric_name}, {mean_val:.4f} ± {std_val:.4f}')
        print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")

    print(f"\nNew Node Test Metrics:")
    print(f"{'-'*50}")
    for metric_name in new_node_test_metric_all_runs[0].keys():
        values = [new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        logger.info(f'new node test {metric_name}, {values}')
        logger.info(f'average new node test {metric_name}, {mean_val:.4f} ± {std_val:.4f}')
        print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")
    
    print(f"\n{'='*120}")
    print("Training completed successfully!")
    print(f"{'='*120}")

    sys.exit()

if __name__ == "__main__":
    # Use the correct argument parsing function
    args = get_link_prediction_args()

    # Force single-GPU training to avoid environment variable issues with distributed training
    world_size = 1
    
    # Single-GPU training
    train(rank=0, world_size=1, args=args)