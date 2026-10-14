import logging
import time
import sys
import os
import numpy as np
import warnings
import json
import torch
import torch.nn as nn
import ast

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

from models.DyGFormerSPSE import DyGFormerSPSE





from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler

from evaluate_models_utils import evaluate_model_link_prediction, get_model_submodule
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args, build_run_name_suffix  # Changed: use link prediction arguments

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # Changed: use link prediction arguments instead of node classification arguments
    args = get_link_prediction_args(is_evaluation=True)
    
    print(f"Running link prediction task...")
    
    # Get link prediction data
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    # Initialize the neighbor sampler (first-order)
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # Get data loaders
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    # Set up negative samplers
    test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
    new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    test_metric_all_runs = []
    new_node_test_metric_all_runs = []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.load_model_name = f'{args.model_name}_seed{args.seed}'
        args.save_result_name = f'evaluate_link_prediction_{args.model_name}_seed{args.seed}'
        suffix = build_run_name_suffix(args, include_batch=True)
        if suffix:
            args.load_model_name += '_' + suffix
            args.save_result_name += '_' + suffix

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/", exist_ok=True)
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # Create the model
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # Compute the time shift, used by the JODIE model
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                           time_feat_dim=args.time_feat_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        elif args.model_name == 'DyGFormerF':
            from models.DyGFormerF import DyGFormerF
            dynamic_backbone = DyGFormerF(
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
                neighbor_sampler=full_neighbor_sampler,
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
                device=args.device
            )





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
                device=args.device
            )
        
        
        elif args.model_name == 'DyGFormerC':
            dynamic_backbone = DyGFormerC(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        
        
        elif args.model_name == 'DyGFormerD':
            from models.DyGFormerD import DyGFormerD
            dynamic_backbone = DyGFormerD(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
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
            dynamic_backbone = DyGFormerST(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length,
                                         max_subgraph_size=getattr(args, 'spd_max_subgraph_size', 512),
                                         spd_precomputed_dir=getattr(args, 'spd_precomputed_dir', None),
                                         spd_lazy_load=getattr(args, 'spd_lazy_load', False),
                                         device=args.device)
        
        elif args.model_name == 'DyGFormerP':
            dynamic_backbone = DyGFormerP(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        
        elif args.model_name == 'DyGFormerR':
            dynamic_backbone = DyGFormerR(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        
        
        
        elif args.model_name == 'NPTDyG':
            dynamic_backbone = NPTDyG(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device
            )
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
                device=args.device
            )
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
                device=args.device
            )
        elif args.model_name == 'DyGFormerSPSE':
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
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")

        # Create link_predictor - use different input dimensions for different models
        if args.model_name in ['DyGFormer', 'DyGFormerF', 'DyGFormerFF', 'DyGFormerC', 'DyGFormerD', 'DyGFormerP', 'DyGFormerR', 'DyGFormerST', 'DyGFormerWSDMP', 'NPTWSDMP', 'DyGFormerWSDMPF', 'DyGFormerSPSE']:
            # DyGFormerF, DyGFormerFF, DyGFormerST, etc. output node_feat_dim, consistent with the rest of the DyGFormer family
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
            # Other models use node_feat_dim by default
            link_predictor = MergeLayer(
                input_dim1=dynamic_backbone.node_feat_dim,
                input_dim2=dynamic_backbone.node_feat_dim,
                hidden_dim=128,
                output_dim=1
             )

        # Combine models
        model = nn.Sequential(dynamic_backbone, link_predictor)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024:.4f} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024:.4f} MB.')

        # Load the trained model
        load_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.load_model_name}/"
        
        # Try to load the original training config
        config_path = os.path.join(load_model_folder, "train_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    train_config = json.load(f)
                    logger.info(f"Loaded training config from {config_path}")
                    
                    # Apply key parameters
                    for key in ['channel_embedding_dim', 'num_heads', 'num_layers', 'time_feat_dim', 'max_input_sequence_length']:
                        if key in train_config and hasattr(args, key):
                            old_val = getattr(args, key)
                            setattr(args, key, train_config[key])
                            logger.info(f"Parameter {key} updated from {old_val} to {train_config[key]}")
            except Exception as e:
                logger.warning(f"Failed to load training config: {str(e)}")
        else:
            # If the config file does not exist, try to record the parameters
            try:
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                config_to_save = {k: v for k, v in vars(args).items() if not k.startswith('_')}
                with open(config_path, 'w') as f:
                    json.dump(config_to_save, f, indent=4)
                logger.info(f"Current parameters saved to {config_path}")
            except Exception as e:
                logger.warning(f"Failed to save current config: {str(e)}")
        
        early_stopping = EarlyStopping(patience=0, save_model_folder=load_model_folder,
                                      save_model_name=args.load_model_name, logger=logger, model_name=args.model_name)
        
        logger.info(f'Trying to load model from {load_model_folder}...')
        try:
            early_stopping.load_checkpoint(model)
            logger.info('Model loaded successfully!')
        except Exception as e:
            logger.error(f'Failed to load model: {str(e)}')
            if run == 0:  # If the first run fails, terminate the program
                sys.exit(1)
            else:
                logger.warning("Skipping this run")
                continue

        model = model.to(args.device)
        model = nn.DataParallel(model)
        print('Using', torch.cuda.device_count(), 'GPUs')

        loss_func = nn.BCEWithLogitsLoss()

        # Evaluate the best model
        logger.info(f'Evaluating model performance on the {args.dataset_name} dataset...')

        # Evaluate test set performance
        eval_sampler = full_neighbor_sampler

        test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                  model=model,
                                                                  neighbor_sampler=eval_sampler,
                                                                  evaluate_idx_data_loader=test_idx_data_loader,
                                                                  evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                  evaluate_data=test_data,
                                                                  loss_func=loss_func,
                                                                  num_neighbors=args.num_neighbors,
                                                                  time_gap=args.time_gap,
                                                                  collect_attention=(args.model_name in ['DyGFormer']))

        # Evaluate new-node test set performance
        new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                    model=model,
                                                                                    neighbor_sampler=eval_sampler,
                                                                                    evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                    evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                    evaluate_data=new_node_test_data,
                                                                                    loss_func=loss_func,
                                                                                    num_neighbors=args.num_neighbors,
                                                                                    time_gap=args.time_gap,
                                                                                    collect_attention=False)  # No need to collect attention for the new-node test

        # Store evaluation metrics for the current run
        test_metric_dict = {}
        new_node_test_metric_dict = {}

        logger.info(f'Test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'Test {metric_name}: {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        logger.info(f'New-node test loss: {np.mean(new_node_test_losses):.4f}')
        for metric_name in new_node_test_metrics[0].keys():
            average_new_node_test_metric = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
            logger.info(f'New-node test {metric_name}: {average_new_node_test_metric:.4f}')
            new_node_test_metric_dict[metric_name] = average_new_node_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        test_metric_all_runs.append(test_metric_dict)
        new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        # Avoid overlapping logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # Save model results
        result_json = {
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
            "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
        }

        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")
        
        with open(save_result_path, 'w') as file:
            file.write(result_json)
        
        logger.info(f'Results saved to {save_result_path}')

    # If there are multiple runs, output the average metrics over all runs
    if args.num_runs > 1:
        logger.info(f'Average metrics over {args.num_runs} runs:')
        
        for metric_name in test_metric_all_runs[0].keys():
            logger.info(f'Test {metric_name}: {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
            logger.info(f'Average test {metric_name}: {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                        f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')
        
        for metric_name in new_node_test_metric_all_runs[0].keys():
            logger.info(f'New-node test {metric_name}: {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}')
            logger.info(f'Average new-node test {metric_name}: {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} '
                        f'± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}')

    sys.exit()
