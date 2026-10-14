# -*- coding: utf-8 -*-
import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import shutil
import json
import torch
import torch.nn as nn
from models.NPTDyG import NPTDyG

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.RepeatMixer import RepeatMixer
from models.DyGFormer import DyGFormer
from models.DyGFormerST import DyGFormerST
from models.NPTSTDyG import NPTSTDyG
from models.DyGFormerD import DyGFormerD
from models.DyGFormerP import DyGFormerP
from models.DyGFormerR import DyGFormerR
from models.DyGFormerC import DyGFormerC
from models.DyGFormerF import DyGFormerF
from models.DyGFormerFF import DyGFormerFF
from models.DyGFormerWSDMP import DyGFormerWSDMP
from models.NPTWSDMP import NPTWSDMP
from models.DyGFormerWSDMPF import DyGFormerWSDMPF
from models.modules import MergeLayer, MLPClassifier
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, get_edge_neighbor_sampler
from evaluate_models_utils import evaluate_model_node_classification
from utils.metrics import get_node_classification_metrics
from utils.DataLoader import get_idx_data_loader, get_node_classification_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_node_classification_args, build_run_name_suffix


import argparse
import ast

# Formal argument parsing and main program follow



def get_backbone(model):
    # Recursively unwrap DataParallel/Sequential until the actual model is reached
    while True:
        if hasattr(model, "module"):
            model = model.module
        elif isinstance(model, nn.Sequential):
            model = model[0]
        else:
            break
    return model

# New: submodule getter compatible with both DataParallel and plain models
def get_submodule(model, idx):
    if isinstance(model, torch.nn.DataParallel):
        return model.module[idx]
    else:
        return model[idx]


if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_node_classification_args()

    # Force the main GPU after argument parsing
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data = \
        get_node_classification_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)
    full_edge_neighbor_sampler = get_edge_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                           time_scaling_factor=args.time_scaling_factor, seed=1)


    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False, distributed=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False, distributed=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False, distributed=False)

    # Only declare the lists during initialization, do not print
    val_metric_all_runs, test_metric_all_runs = [], []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        # If --lp_pretrained_model_name is explicitly specified (e.g. DyGFormer_seed1), use that checkpoint; otherwise default to seed0 and fall back based on existence when loading
        lp_suffix = build_run_name_suffix(args, include_batch=True)
        default_lp_name = f'{args.model_name}_seed0'
        if lp_suffix:
            default_lp_name += '_' + lp_suffix
        args.load_model_name = (args.lp_pretrained_model_name if args.lp_pretrained_model_name
                                else default_lp_name)
        args.save_model_name = f'node_classification_{args.model_name}_{args.node_cls_train_mode}_seed{args.seed}'
        nc_suffix = build_run_name_suffix(args, include_batch=True)
        if nc_suffix:
            args.save_model_name += '_' + nc_suffix

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level (INFO instead of WARNING to show important info)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)  # Changed to INFO to avoid showing DEBUG logs
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        # After logger init
        logger.info(f"Initialized metric collection lists: run={run+1}/{args.num_runs}, val_runs={len(val_metric_all_runs)}, test_runs={len(test_metric_all_runs)}")

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # create model
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=device)
        elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                           time_feat_dim=args.time_feat_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=device)
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=device)
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=device)
        elif args.model_name == 'RepeatMixer':
            dynamic_backbone = RepeatMixer(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                edge_neighbor_sampler=full_edge_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                num_tokens=getattr(args, 'num_tokens', args.num_neighbors),
                high_order=getattr(args, 'high_order', False),
                num_layers=args.num_layers,
                token_dim_expansion_factor=getattr(args, 'token_dim_expansion_factor', 0.5),
                channel_dim_expansion_factor=getattr(args, 'channel_dim_expansion_factor', 4.0),
                dropout=args.dropout,
                device=device
            )
            node_classifier = MLPClassifier(input_dim=dynamic_backbone.num_channels, dropout=args.dropout)
        elif args.model_name == 'DyGFormer':
            
            dynamic_backbone = DyGFormer(
                node_raw_features=node_raw_features, 
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim, 
                channel_embedding_dim=args.channel_embedding_dim, 
                patch_size=args.patch_size,
                num_layers=args.num_layers, 
                num_heads=args.num_heads, 
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length, 
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        elif args.model_name == 'DyGFormerST':
            dynamic_backbone = DyGFormerST(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                max_subgraph_size=getattr(args, 'spd_max_subgraph_size', 512),
                spd_precomputed_dir=getattr(args, 'spd_precomputed_dir', None),
                spd_lazy_load=getattr(args, 'spd_lazy_load', False),
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        elif args.model_name == 'NPTSTDyG':
            dynamic_backbone = NPTSTDyG(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        elif args.model_name == 'DyGFormerC':
            dynamic_backbone = DyGFormerC(
                node_raw_features=node_raw_features, 
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim, 
                channel_embedding_dim=args.channel_embedding_dim, 
                patch_size=args.patch_size,
                num_layers=args.num_layers, 
                num_heads=args.num_heads, 
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length, 
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        
        
        elif args.model_name == 'DyGFormerF':
            # Make sure the dimension satisfies the differential attention requirement
            num_channels = 4
            attention_dim = num_channels * args.channel_embedding_dim
            if attention_dim % (2 * args.num_heads) != 0:
                original_dim = args.channel_embedding_dim
                # Adjust to the smallest value that satisfies the condition
                required_divisor = 2 * args.num_heads
                # Find the smallest channel_embedding_dim that satisfies the condition
                while (num_channels * args.channel_embedding_dim) % required_divisor != 0:
                    args.channel_embedding_dim += 1
                logger.info(f"Adjusted channel_embedding_dim from {original_dim} to {args.channel_embedding_dim} to satisfy the differential attention requirement")
            
            dynamic_backbone = DyGFormerF(
                node_raw_features=node_raw_features, 
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim, 
                channel_embedding_dim=args.channel_embedding_dim, 
                patch_size=args.patch_size,
                num_layers=args.num_layers, 
                num_heads=args.num_heads, 
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length, 
                device=device
            )
            logger.info(f"DyGFormerF config: num_channels=4, channel_embedding_dim={args.channel_embedding_dim}, "
                       f"num_heads={args.num_heads}, attention_dim={4*args.channel_embedding_dim}, "
                       f"head_dim={4*args.channel_embedding_dim//args.num_heads//2}")
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        elif args.model_name == 'DyGFormerFF':
            num_channels = 4
            attention_dim = num_channels * args.channel_embedding_dim
            if attention_dim % (2 * args.num_heads) != 0:
                original_dim = args.channel_embedding_dim
                required_divisor = 2 * args.num_heads
                while (num_channels * args.channel_embedding_dim) % required_divisor != 0:
                    args.channel_embedding_dim += 1
                logger.info(f"Adjusted channel_embedding_dim from {original_dim} to {args.channel_embedding_dim} to satisfy attention_dim % (2*num_heads) == 0")
            if args.channel_embedding_dim % args.num_heads != 0:
                original_dim = args.channel_embedding_dim
                args.channel_embedding_dim = (args.channel_embedding_dim // args.num_heads) * args.num_heads
                if args.channel_embedding_dim == 0:
                    args.channel_embedding_dim = args.num_heads
                logger.info(f"Adjusted channel_embedding_dim from {original_dim} to {args.channel_embedding_dim} to satisfy RoPE (channel_embedding_dim % num_heads == 0)")
            dynamic_backbone = DyGFormerFF(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                num_kv_heads=getattr(args, 'num_kv_heads', None),
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=device,
                max_second_hop_length=getattr(args, 'max_second_hop_length', 0),
            )
            logger.info(f"DyGFormerFF config: num_channels=4, channel_embedding_dim={args.channel_embedding_dim}, "
                       f"num_heads={args.num_heads}, attention_dim={4*args.channel_embedding_dim}, "
                       f"head_dim={4*args.channel_embedding_dim//args.num_heads//2}")
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        
        
        
        
        elif args.model_name == 'DyGFormerWSDMP':
            dynamic_backbone = DyGFormerWSDMP(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)

        elif args.model_name == 'NPTWSDMP':
            dynamic_backbone = NPTWSDMP(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)





        elif args.model_name == 'DyGFormerWSDMPF':
            dynamic_backbone = DyGFormerWSDMPF(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        
        
        elif args.model_name == 'DyGFormerD':
            dynamic_backbone = DyGFormerD(
                node_raw_features=node_raw_features, 
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim, 
                channel_embedding_dim=args.channel_embedding_dim, 
                patch_size=args.patch_size,
                num_layers=args.num_layers, 
                num_heads=args.num_heads, 
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                use_lpu=getattr(args, 'use_lpu', True),
                use_rpe=getattr(args, 'use_rpe', True),
                rwpe_dim=getattr(args, 'rwpe_dim', 8),
                rwpe_precomputed_dir=getattr(args, 'rwpe_precomputed_dir', None),
                rwpe_lazy_load=getattr(args, 'rwpe_lazy_load', False),
                max_subgraph_size=getattr(args, 'rwpe_max_subgraph_size', 512),
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        
        elif args.model_name == 'DyGFormerP':
            dynamic_backbone = DyGFormerP(
                node_raw_features=node_raw_features, 
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim, 
                channel_embedding_dim=args.channel_embedding_dim, 
                patch_size=args.patch_size,
                num_layers=args.num_layers, 
                num_heads=args.num_heads, 
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length, 
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        
        elif args.model_name == 'DyGFormerR':
            dynamic_backbone = DyGFormerR(
                node_raw_features=node_raw_features, 
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim, 
                channel_embedding_dim=args.channel_embedding_dim, 
                patch_size=args.patch_size,
                num_layers=args.num_layers, 
                num_heads=args.num_heads, 
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length, 
                device=device
            )
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        
        
        elif args.model_name == 'NPTDyG':
            dynamic_backbone = NPTDyG(
                node_raw_features=node_raw_features,edge_raw_features=edge_raw_features,neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,channel_embedding_dim=args.channel_embedding_dim,num_layers=args.num_layers,
                num_heads=args.num_heads,dropout=args.dropout,max_input_sequence_length=args.max_input_sequence_length,device=device)
            
            # Classifier creation code for NPTDyG
            output_dim = node_raw_features.shape[1]  # Use the original node feature dimension
            logger.info(f"Creating classifier for NPTDyG (input dim: {output_dim})")
            node_classifier = MLPClassifier(input_dim=output_dim, dropout=args.dropout)
            
            # No longer skip the subsequent steps
        elif args.model_name == 'DyGFormerSPSE':
            from models.DyGFormerSPSE import DyGFormerSPSE
            dynamic_backbone = DyGFormerSPSE(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=device,
                spse_dist_len=getattr(args, 'spse_dist_len', 2),
                spse_precomputed_dir=getattr(args, 'spse_precomputed_dir', None),
            )
            node_classifier = MLPClassifier(input_dim=dynamic_backbone.node_feat_dim, dropout=args.dropout)
        else:
            # Other models use the default classifier
            node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], dropout=args.dropout)
        
        if args.node_cls_train_mode == 'random_finetune':
            if args.lp_pretrained_model_name:
                logger.warning(f"random_finetune mode does not load the LP checkpoint, ignoring --lp_pretrained_model_name={args.lp_pretrained_model_name}")
            logger.info("Fine-tuning node classification with a randomly initialized backbone, not loading the LP checkpoint")
            model = nn.Sequential(dynamic_backbone, node_classifier)
        else:
            if hasattr(dynamic_backbone, 'node_feat_dim'):
                lp_feat_dim = dynamic_backbone.node_feat_dim
            elif hasattr(dynamic_backbone, 'num_channels'):
                lp_feat_dim = dynamic_backbone.num_channels
            else:
                lp_feat_dim = node_raw_features.shape[1]
            # Consistent with train_link_prediction: MergeLayer in LP uses a fixed hidden_dim=128, so the structure must match when loading the checkpoint
            link_predictor = MergeLayer(input_dim1=lp_feat_dim, input_dim2=lp_feat_dim,
                                        hidden_dim=128, output_dim=1)
            lp_model = nn.Sequential(dynamic_backbone, link_predictor)

            # load the saved model in the link prediction task
            # If --lp_pretrained_model_name is explicitly specified, only use that checkpoint (use it if it exists, otherwise raise an error); otherwise automatically pick the first existing one in seed order
            base_folder = f"./saved_models/{args.model_name}/{args.dataset_name}"
            load_model_name = args.load_model_name
            if args.lp_pretrained_model_name:
                pkl_path = os.path.join(base_folder, load_model_name, f"{load_model_name}.pkl")
                if os.path.isfile(pkl_path):
                    logger.info(f"Using the specified LP pretrained checkpoint: {pkl_path}")
                else:
                    raise FileNotFoundError(f"Specified LP checkpoint does not exist: {pkl_path} (please check --lp_pretrained_model_name={args.lp_pretrained_model_name})")
            else:
                seed_order = [1] + [s for s in range(args.num_runs) if s != 1]  # [1, 0, 2] when num_runs=3
                for s in seed_order:
                    candidate = f"{args.model_name}_seed{s}"
                    if lp_suffix:
                        candidate += '_' + lp_suffix
                    pkl_path = os.path.join(base_folder, candidate, f"{candidate}.pkl")
                    if os.path.isfile(pkl_path):
                        load_model_name = candidate
                        logger.info(f"Using LP pretrained checkpoint: {pkl_path}")
                        break
                else:
                    logger.warning(f"No .pkl found for {base_folder}/{args.model_name}_seed{{0..{args.num_runs-1}}}[+suffix], still using load_model_name={args.load_model_name}")
            load_model_folder = os.path.join(base_folder, load_model_name)
            lp_loader = EarlyStopping(patience=0, save_model_folder=load_model_folder,
                                      save_model_name=load_model_name, logger=logger, model_name=args.model_name)
            lp_loader.load_checkpoint(lp_model, map_location='cpu')

            # create the model for the node classification task
            model = nn.Sequential(lp_model[0], node_classifier)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024:.4f} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024:.4f} MB.')

        # === Multi-GPU parallelism fix ===
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs! DataParallel enabled.")
            model = torch.nn.DataParallel(model)
        else:
            print('Single GPU or CPU, DataParallel not used.')

        if args.node_cls_train_mode == 'classifier_only':
            for parameter in get_submodule(model, 0).parameters():
                parameter.requires_grad = False
            optimizer = create_optimizer(model=get_submodule(model, 1), optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
        else:
            for parameter in get_submodule(model, 0).parameters():
                parameter.requires_grad = True
            optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        # put the node raw messages of memory-based models on device
        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            for node_id, node_raw_messages in get_submodule(model, 0).memory_bank.node_raw_messages.items():
                new_node_raw_messages = []
                for node_raw_message in node_raw_messages:
                    new_node_raw_messages.append((node_raw_message[0].to(device), node_raw_message[1]))
                get_submodule(model, 0).memory_bank.node_raw_messages[node_id] = new_node_raw_messages

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        # Resume-from-checkpoint related
        start_epoch = 0
        if hasattr(args, 'resume') and args.resume:
            checkpoint_path = os.path.join(save_model_folder, 'checkpoint.pth')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming training: continuing from epoch {start_epoch}")
            else:
                print("No checkpoint found, cannot resume; training will start from scratch.")

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        loss_func = nn.BCELoss()

        epoch_train_losses = []
        epoch_val_losses = []

        if args.node_cls_train_mode == 'classifier_only':
            get_submodule(model, 0).eval()
        else:
            get_submodule(model, 0).train()

        for epoch in range(start_epoch, args.num_epochs):

            if args.node_cls_train_mode == 'classifier_only':
                get_submodule(model, 0).eval()
            else:
                get_submodule(model, 0).train()
            get_submodule(model, 1).train()
            if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'RepeatMixer', 'DyGFormer', 'DyGFormerF', 'DyGFormerFF', 'DyGFormerC', 'DyGFormerD', 'DyGFormerP', 'DyGFormerR', 'DyGFormerST', 'NPTSTDyG', 'NPTDyG', 'DyGFormerWSDMP', 'NPTWSDMP', 'DyGFormerWSDMPF', 'DyGFormerSPSE']:
                # training process, set the neighbor sampler
                get_backbone(model).set_neighbor_sampler(full_neighbor_sampler)
            if args.model_name == 'RepeatMixer':
                get_backbone(model).set_edge_neighbor_sampler(full_edge_neighbor_sampler)
            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # reinitialize memory of memory-based models at the start of each epoch
                get_submodule(model, 0).memory_bank.__init_memory_bank__()

            # store train losses, trues and predicts
            train_total_loss, train_y_trues, train_y_predicts = 0.0, [], []
            batch_peak_allocated_mb_list = []
            batch_peak_reserved_mb_list = []
            epoch_start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=100, 
                                            desc=f'Epoch {epoch + 1}/{args.num_epochs}', 
                                            leave=True, position=0, dynamic_ncols=True)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], train_data.node_interact_times[train_data_indices], \
                    train_data.edge_ids[train_data_indices], train_data.labels[train_data_indices]

                # Before feeding each batch into the model, make sure all input tensors are on the main GPU
                batch_src_node_ids = torch.LongTensor(batch_src_node_ids).to(device)
                batch_dst_node_ids = torch.LongTensor(batch_dst_node_ids).to(device)
                batch_node_interact_times = torch.FloatTensor(batch_node_interact_times).to(device)

                grad_context = torch.no_grad() if args.node_cls_train_mode == 'classifier_only' else torch.enable_grad()
                with grad_context:
                    if args.model_name in ['TGAT', 'CAWN', 'TCL']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            get_backbone(model).compute_src_dst_node_temporal_embeddings(
                                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times
                            )
                    elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch = {
                            'src_node_ids': batch_src_node_ids,
                            'dst_node_ids': batch_dst_node_ids,
                            'node_interact_times': batch_node_interact_times,
                            'edge_ids': batch_edge_ids,
                            'edges_are_positive': True,
                            'num_neighbors': args.num_neighbors
                        }
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            get_backbone(model).compute_src_dst_node_temporal_embeddings(batch)
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
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            get_backbone(model).compute_src_dst_node_temporal_embeddings(batch)
                    elif args.model_name == 'RepeatMixer':
                        # RepeatMixer returns single edge embedding (batch_size, num_channels)
                        batch_src_node_ids_np = batch_src_node_ids.cpu().numpy() if hasattr(batch_src_node_ids, 'cpu') else batch_src_node_ids
                        batch_dst_node_ids_np = batch_dst_node_ids.cpu().numpy() if hasattr(batch_dst_node_ids, 'cpu') else batch_dst_node_ids
                        batch_times_np = batch_node_interact_times.cpu().numpy() if hasattr(batch_node_interact_times, 'cpu') else batch_node_interact_times
                        batch_src_node_embeddings = get_backbone(model).compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_src_node_ids_np,
                            dst_node_ids=batch_dst_node_ids_np,
                            node_interact_times=batch_times_np,
                            num_neighbors=args.num_neighbors,
                            time_gap=args.time_gap
                        )
                    elif args.model_name == 'DyGFormerD':
                        # node_t: look up by (node,t) (edge_ids can be ignored); the old edge_id mode still uses edge_ids
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            get_backbone(model).compute_src_dst_node_temporal_embeddings(
                                src_node_ids=batch_src_node_ids,
                                dst_node_ids=batch_dst_node_ids,
                                node_interact_times=batch_node_interact_times,
                                edge_ids=batch_edge_ids,
                            )
                    elif args.model_name in ['DyGFormer', 'DyGFormerF', 'DyGFormerFF', 'DyGFormerC', 'DyGFormerP', 'DyGFormerR', 'DyGFormerST', 'NPTSTDyG', 'NPTDyG', 'DyGFormerWSDMP', 'NPTWSDMP', 'DyGFormerWSDMPF', 'DyGFormerSPSE']:
                        # get temporal embedding of source and destination nodes
                        # All these models return 2 values: (src_embeddings, dst_embeddings)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            get_backbone(model).compute_src_dst_node_temporal_embeddings(
                                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times
                            )
                    else:
                        raise ValueError(f"Wrong value for model_name {args.model_name}!")

                # get predicted probabilities, shape (batch_size, )
                predicts = get_submodule(model, 1)(x=batch_src_node_embeddings)
                # Make sure the output is one-dimensional
                if len(predicts.shape) > 1:
                    # Squeeze to one dimension regardless of the number of dimensions
                    predicts = predicts.squeeze()
                        
                # If it is a scalar tensor (may happen when batch_size=1), make sure it is one-dimensional
                if predicts.dim() == 0:
                    predicts = predicts.unsqueeze(0)
                    
                predicts = predicts.sigmoid()
                    
                # Added before the loss computation
                labels = torch.from_numpy(batch_labels).float().to(predicts.device)
                predicts = predicts.to(labels.dtype)  # Make sure data types match

                # Final check
                if predicts.shape != labels.shape:
                    logger.warning(f"Adjusting shape: predictions {predicts.shape} -> labels {labels.shape}")
                    predicts = predicts.view(labels.shape)

                loss = loss_func(input=predicts, target=labels)

                train_total_loss += loss.item()

                train_y_trues.append(labels)
                train_y_predicts.append(predicts)

                accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
                if accumulation_steps > 1:
                    if batch_idx % accumulation_steps == 0:
                        optimizer.zero_grad()
                    scaled_loss = loss / accumulation_steps
                    scaled_loss.backward()
                    is_last_batch = (batch_idx + 1) == len(train_idx_data_loader)
                    if (batch_idx + 1) % accumulation_steps == 0 or is_last_batch:
                        optimizer.step()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                    batch_peak_allocated_mb_list.append(
                        torch.cuda.max_memory_allocated(device) / 1000 ** 2)
                    batch_peak_reserved_mb_list.append(
                        torch.cuda.max_memory_reserved(device) / 1000 ** 2)

                # Update the progress bar less frequently
                if batch_idx % 10 == 0 or batch_idx == len(train_idx_data_loader) - 1:
                    train_idx_data_loader_tqdm.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Batch': f'{batch_idx + 1}/{len(train_idx_data_loader)}'
                    })

            # ---- Output GPU memory statistics at the end of each epoch ----
            if torch.cuda.is_available() and batch_peak_allocated_mb_list:
                torch.cuda.synchronize(device)
                avg_alloc = float(np.mean(batch_peak_allocated_mb_list))
                max_alloc = float(np.max(batch_peak_allocated_mb_list))
                avg_rsvd  = float(np.mean(batch_peak_reserved_mb_list))
                max_rsvd  = float(np.max(batch_peak_reserved_mb_list))
                train_time = time.time() - epoch_start_time
                memory_msg = (
                    f'[{args.model_name}][Epoch {epoch + 1}] train CUDA memory: '
                    f'avg_batch_peak_allocated={avg_alloc:.2f} MB, '
                    f'max_batch_peak_allocated={max_alloc:.2f} MB, '
                    f'avg_batch_peak_reserved={avg_rsvd:.2f} MB, '
                    f'max_batch_peak_reserved={max_rsvd:.2f} MB, '
                    f'train_time={train_time:.2f}s'
                )
                logger.info(memory_msg)
                print(memory_msg)
            # ---- End of GPU memory statistics ----

            train_total_loss /= (batch_idx + 1)
            train_y_trues = torch.cat(train_y_trues, dim=0)
            train_y_predicts = torch.cat(train_y_predicts, dim=0)

            # Compute training metrics
            try:
                train_metrics = get_node_classification_metrics(predicts=train_y_predicts, labels=train_y_trues)
                # If nan is returned, log a warning
                if np.isnan(train_metrics.get('roc_auc', 0)):
                    if torch.any(torch.isnan(train_y_predicts)) or torch.any(torch.isinf(train_y_predicts)):
                        logger.warning("Training metric computation: predictions contain nan or inf, cannot compute ROC-AUC")
                    else:
                        unique_labels = torch.unique(train_y_trues)
                        if len(unique_labels) < 2:
                            logger.warning(f"Training metric computation: only one class ({unique_labels[0].item()}), cannot compute ROC-AUC")
                        else:
                            logger.warning("Training metric computation: ROC-AUC result is nan")
            except Exception as e:
                logger.warning(f"Error computing training metrics: {e}")
                train_metrics = {'roc_auc': np.nan}

            # Make sure metrics are collected even if validation has problems
            try:
                val_total_loss, val_metrics = evaluate_model_node_classification(
                    model_name=args.model_name,
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=val_idx_data_loader,
                    evaluate_data=val_data,
                    loss_func=loss_func,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap,
                    edge_neighbor_sampler=full_edge_neighbor_sampler if args.model_name == 'RepeatMixer' else None
                )

                logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {train_total_loss:.4f}')
                for metric_name in train_metrics.keys():
                    logger.info(f'train {metric_name}, {train_metrics[metric_name]:.4f}')
                
                # Print epoch summary
                print(f"\n{'='*80}")
                print(f"Epoch {epoch + 1}/{args.num_epochs} Summary:")
                print(f"{'='*80}")
                print(f"Train Loss: {train_total_loss:.4f}")
                for metric_name in train_metrics.keys():
                    print(f"Train {metric_name}: {train_metrics[metric_name]:.4f}")
                
                logger.info(f'validate loss: {val_total_loss:.4f}')
                print(f"Validation Loss: {val_total_loss:.4f}")
                for metric_name in val_metrics.keys():
                    val_metric_value = val_metrics[metric_name]
                    # Handle nan values
                    if np.isnan(val_metric_value):
                        logger.warning(f'validate {metric_name} is nan (likely because extreme data imbalance makes some batches uncomputable)')
                        print(f"Validation {metric_name}: nan (data issue, cannot compute)")
                    else:
                        logger.info(f'validate {metric_name}, {val_metric_value:.4f}')
                        print(f"Validation {metric_name}: {val_metric_value:.4f}")
                print(f"{'='*80}\n")
            except Exception as e:
                logger.warning(f"Error during validation: {e}")
                val_metrics = {"auc": 0.5}  # Default metrics
                val_total_loss = float('inf')

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # backup memory bank after validating so it can be used for testing nodes (since test edges are strictly later in time than validation edges)
                    val_backup_memory_bank = get_submodule(model, 0).memory_bank.backup_memory_bank()

                try:
                    test_total_loss, test_metrics = evaluate_model_node_classification(model_name=args.model_name,
                                                                                    model=model,
                                                                                    neighbor_sampler=full_neighbor_sampler,
                                                                                    evaluate_idx_data_loader=test_idx_data_loader,
                                                                                    evaluate_data=test_data,
                                                                                    loss_func=loss_func,
                                                                                    num_neighbors=args.num_neighbors,
                                                                                    time_gap=args.time_gap,
                                                                                    edge_neighbor_sampler=full_edge_neighbor_sampler if args.model_name == 'RepeatMixer' else None)

                    logger.info(f'test loss: {test_total_loss:.4f}')
                    for metric_name in test_metrics.keys():
                        logger.info(f'test {metric_name}, {test_metrics[metric_name]:.4f}')
                except Exception as e:
                    logger.info(f"Error during testing, skipping this test")
                    test_metrics = {"auc": 0.5}  # Default metrics
                    test_total_loss = float('inf')

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # reload validation memory bank for saving models
                    # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                    get_submodule(model, 0).memory_bank.reload_memory_bank(val_backup_memory_bank)

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics.keys():
                metric_value = val_metrics[metric_name]
                # If the metric is nan, use 0.0 as the default (this won't break early stopping, but a warning is logged)
                if np.isnan(metric_value):
                    logger.warning(f"Validation metric {metric_name} is nan, using 0.0 for early stopping (may affect model selection)")
                    metric_value = 0.0
                val_metric_indicator.append((metric_name, metric_value, True))
            early_stop = early_stopping.step(val_metric_indicator, model)

            # Persist newly computed (node,t) RWPE from misses
            if args.model_name == 'DyGFormerD':
                backbone = get_backbone(model)
                if hasattr(backbone, 'flush_rwpe_lazy_cache'):
                    backbone.flush_rwpe_lazy_cache()

            # Record epoch-level loss
            #epoch_train_losses.append(train_total_loss)
            #epoch_val_losses.append(val_total_loss if val_total_loss != float('inf') else float('nan'))

            # Save/refresh the loss curve every epoch
            #loss_curve_folder = f"./loss_curve/{args.model_name}/{args.dataset_name}"
            #os.makedirs(loss_curve_folder, exist_ok=True)
           # loss_curve_path = os.path.join(
              #  loss_curve_folder,
              #  f"{args.save_model_name}_loss_curve.png"
           # )
          #  fig, ax = plt.subplots(figsize=(8, 5))
           # epochs_x = list(range(1, len(epoch_train_losses) + 1))
          #  ax.plot(epochs_x, epoch_train_losses, label='Train Loss', marker='o', markersize=3, linewidth=1.5)
           # ax.plot(epochs_x, epoch_val_losses,   label='Val Loss',   marker='s', markersize=3, linewidth=1.5)
          #  ax.set_xlabel('Epoch')
          #  ax.set_ylabel('BCE Loss')
           # ax.set_title(f'{args.model_name} on {args.dataset_name}\nLoss Curve (seed={args.seed})')
           # ax.legend()
            #ax.grid(True, alpha=0.3)
           # plt.tight_layout()
           # plt.savefig(loss_curve_path, dpi=150)
           # plt.close(fig)
           # logger.info(f"Loss curve saved to {loss_curve_path}")

            if early_stop:
                break

            # Save a checkpoint after each epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(save_model_folder, 'checkpoint.pth'))

        if args.model_name == 'DyGFormerD':
            backbone = get_backbone(model)
            if hasattr(backbone, 'flush_rwpe_lazy_cache'):
                backbone.flush_rwpe_lazy_cache()

        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f'Computing final performance evaluation')

        # Validation evaluation
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            try:
                val_total_loss, val_metrics = evaluate_model_node_classification(
                    model_name=args.model_name,
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=val_idx_data_loader,
                    evaluate_data=val_data,
                    loss_func=loss_func,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap,
                    edge_neighbor_sampler=full_edge_neighbor_sampler if args.model_name == 'RepeatMixer' else None
                )
                logger.info(f'Validation metrics: loss={val_total_loss:.4f}')
                val_metric_dict = {}
                for metric_name in val_metrics.keys():
                    val_metric = val_metrics[metric_name]
                    logger.info(f'Validation {metric_name}={val_metric:.4f}')
                    val_metric_dict[metric_name] = val_metric
            except Exception as e:
                logger.info(f"Final validation evaluation failed")
                val_metric_dict = {"auc": 0.5}  # Default metrics

        # Test evaluation
        try:
            test_total_loss, test_metrics = evaluate_model_node_classification(
                model_name=args.model_name,
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=test_idx_data_loader,
                evaluate_data=test_data,
                loss_func=loss_func,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap,
                edge_neighbor_sampler=full_edge_neighbor_sampler if args.model_name == 'RepeatMixer' else None
            )
            logger.info(f'Test metrics: loss={test_total_loss:.4f}')
            test_metric_dict = {}
            for metric_name in test_metrics.keys():
                test_metric = test_metrics[metric_name]
                logger.info(f'Test {metric_name}={test_metric:.4f}')
                test_metric_dict[metric_name] = test_metric
        except Exception as e:
            logger.info(f"Final test evaluation failed")
            test_metric_dict = {"auc": 0.5}  # Default metrics

        # Collect performance metrics
        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        # Make sure metrics are correctly appended to the lists
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_metric_all_runs.append(val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)

        # Save results for the current run inside the loop (consistent with link prediction)
        run_result_dict = {}
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            run_result_dict["validation"] = {metric_name: float(val_metric_dict[metric_name])
                                             for metric_name in val_metric_dict}
        run_result_dict["test"] = {metric_name: float(test_metric_dict[metric_name])
                                   for metric_name in test_metric_dict}
        run_save_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(run_save_folder, exist_ok=True)
        run_save_path = os.path.join(run_save_folder, f"{args.save_model_name}.json")
        with open(run_save_path, 'w') as f:
            f.write(json.dumps(run_result_dict, indent=4))
        logger.info(f"Run {run} results saved to {run_save_path}")

        # Avoid overlapping logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

    # Modified final summary section with simplified output
    # store the average metrics at the log of the last run
    logger.info(f'Model evaluation metrics summary ({args.num_runs} runs):')

    # Check validation metrics
    if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
        if len(val_metric_all_runs) > 0:
            logger.info(f'Validation metrics:')
            for metric_name in val_metric_all_runs[0].keys():
                avg_metric = np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs])
                std_metric = np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1)
                logger.info(f'  {metric_name}: {avg_metric:.4f} ± {std_metric:.4f}')
        else:
            logger.info(f"No validation metrics")

    # Check test metrics
    if len(test_metric_all_runs) > 0:
        logger.info(f'Test metrics:')
        for metric_name in test_metric_all_runs[0].keys():
            avg_metric = np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs])
            std_metric = np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1)
            logger.info(f'  {metric_name}: {avg_metric:.4f} ± {std_metric:.4f}')
    else:
        logger.info(f"No test metrics")

    # Save summary results (mean ± std); the _summary file name distinguishes it from single-run files
    summary_dict = {}
    if args.model_name not in ['JODIE', 'DyRep', 'TGN'] and len(val_metric_all_runs) > 0:
        summary_dict["validation"] = {
            metric_name: {
                "mean": float(np.mean([r[metric_name] for r in val_metric_all_runs])),
                "std":  float(np.std( [r[metric_name] for r in val_metric_all_runs], ddof=1)),
            }
            for metric_name in val_metric_all_runs[0]
        }
    if len(test_metric_all_runs) > 0:
        summary_dict["test"] = {
            metric_name: {
                "mean": float(np.mean([r[metric_name] for r in test_metric_all_runs])),
                "std":  float(np.std( [r[metric_name] for r in test_metric_all_runs], ddof=1)),
            }
            for metric_name in test_metric_all_runs[0]
        }
    save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
    os.makedirs(save_result_folder, exist_ok=True)
    base_name = f"node_classification_{args.model_name}_{args.node_cls_train_mode}"
    summary_suffix = build_run_name_suffix(args, include_batch=True)
    if summary_suffix:
        base_name += '_' + summary_suffix
    save_result_path = os.path.join(save_result_folder, f"{base_name}_summary.json")
    with open(save_result_path, 'w') as file:
        file.write(json.dumps(summary_dict, indent=4))
    logger.info(f"Summary results saved to {save_result_path}")

    sys.exit()
