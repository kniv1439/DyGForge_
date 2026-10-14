import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json
from typing import Callable, Optional
import inspect

# ---------------------- DyGFormer import for pairwise metrics ----------------------
from models.DyGFormer import DyGFormer
# -----------------------------------------------------------------------------------
from utils.metrics import get_link_prediction_metrics, get_node_classification_metrics
from utils.utils import NegativeEdgeSampler, NeighborSampler
from utils.DataLoader import Data
from utils.load_configs import build_run_name_suffix


def get_model_submodule(model, idx):
    """Safely get a submodule of the model, compatible with DataParallel and Sequential"""
    if isinstance(model, torch.nn.DataParallel):
        return model.module[idx]
    else:
        return model[idx]


def call_classifier(classifier, src_emb, dst_emb):
    """Smartly call the classifier, automatically adapting to different forward signatures"""
    # Check the classifier's forward method signature
    try:
        sig = inspect.signature(classifier.forward)
        params = list(sig.parameters.keys())
        
        # Call the appropriate method based on the parameter signature
        if 'input_1' in params and 'input_2' in params:
            return classifier(input_1=src_emb, input_2=dst_emb)
        elif 'x' in params:
            # For classifiers that accept only one argument, pass the difference or concatenation of the two embeddings
            return classifier(x=src_emb - dst_emb)
        elif len(params) >= 2:
            # Pass two arguments directly
            return classifier(src_emb, dst_emb)
        else:
            # Default case: concatenate the two embeddings
            combined = torch.cat([src_emb, dst_emb], dim=-1)
            return classifier(combined)
    except Exception as e:
        # If signature inspection fails, try various calling conventions
        try:
            return classifier(input_1=src_emb, input_2=dst_emb)
        except:
            try:
                return classifier(src_emb, dst_emb)
            except:
                try:
                    return classifier(x=src_emb - dst_emb)
                except:
                    combined = torch.cat([src_emb, dst_emb], dim=-1)
                    return classifier(combined)


def evaluate_model_link_prediction(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module,
                                   num_neighbors: int = 20, time_gap: int = 2000, collect_attention: bool = False):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Get the logger (create it if it does not exist)
    logger = logging.getLogger()
    
    # ------------------------------------------------------------------
    # For DyGFormer: reset pairwise metric accumulators at evaluation start
    # ------------------------------------------------------------------
    pairwise_tracker = None
    if model_name == 'DyGFormer':
        try:
            pairwise_tracker = get_model_submodule(model, 0)
            if isinstance(pairwise_tracker, DyGFormer):
                pairwise_tracker.reset_pairwise_metrics()
        except Exception:
            pairwise_tracker = None

    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'DyGFormerF', 'DyGFormerFF', 'DyGFormerC', 'DyGFormerD', 'DyGFormerP', 'DyGFormerR', 'NPTDyG', 'NPTSTDyG','DyGFormerST', 'DyGFormerWSDMP', 'NPTWSDMP', 'DyGFormerWSDMPF', 'DyGFormerSPSE']:
        # evaluation phase use all the graph information
        get_model_submodule(model, 0).set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

            if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                                  batch_src_node_ids=batch_src_node_ids,
                                                                                                  batch_dst_node_ids=batch_dst_node_ids,
                                                                                                  current_batch_start_time=batch_node_interact_times[0],
                                                                                                  current_batch_end_time=batch_node_interact_times[-1])
            else:
                _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # note that negative nodes do not change the memories while the positive nodes change the memories,
                # we need to first compute the embeddings of negative nodes for memory-based models
                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=None,
                                                                      edges_are_positive=False,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['DyGFormer', 'DyGFormerF', 'DyGFormerFF', 'DyGFormerC', 'DyGFormerD', 'DyGFormerP', 'DyGFormerR', 'NPTDyG', 'NPTSTDyG', 'DyGFormerST', 'DyGFormerWSDMP', 'NPTWSDMP', 'DyGFormerWSDMPF', 'DyGFormerSPSE']:
                if model_name == 'NPTDyG':
                    # Use the standard compute_src_dst_node_temporal_embeddings method
                    batch_src_node_embeddings, batch_dst_node_embeddings = get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times
                    )
                    
                    # Use the same method for negative samples
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_neg_src_node_ids,
                        dst_node_ids=batch_neg_dst_node_ids,
                        node_interact_times=batch_node_interact_times
                    )
                elif model_name == 'DyGFormerSPSE':
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
                elif model_name == 'DyGFormerD':
                    # node_t: both positive and negative samples look up by (node,t); old edge_id: positive samples look up / negative samples computed online
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
                elif model_name == 'DyGFormerST':
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
                else:  # DyGFormer, DyGFormerF and other DyGFormer-family models
                    # Logic for DyGFormer-family models; pass the corresponding arguments if attention weights need to be collected
                    _out_pos = get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times
                    )
                    if len(_out_pos) == 3:
                        batch_src_node_embeddings, batch_dst_node_embeddings, _ = _out_pos
                    else:
                        batch_src_node_embeddings, batch_dst_node_embeddings = _out_pos
                        
                    _out_neg = get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_neg_src_node_ids,
                        dst_node_ids=batch_neg_dst_node_ids,
                        node_interact_times=batch_node_interact_times
                    )
                    if len(_out_neg) == 3:
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, _ = _out_neg
                    else:
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = _out_neg
          
            # Compute logits for positive and negative samples (no sigmoid, let BCEWithLogitsLoss handle it internally)
            positive_probabilities = call_classifier(get_model_submodule(model, 1), batch_src_node_embeddings, batch_dst_node_embeddings).squeeze(dim=-1)
            negative_probabilities = call_classifier(get_model_submodule(model, 1), batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings).squeeze(dim=-1)

            # Handle possible shape mismatches
            if positive_probabilities.dim() == 2 and negative_probabilities.dim() == 2:
                # If both are 2D, make sure the sequence lengths match
                min_seq_len = min(positive_probabilities.size(1), negative_probabilities.size(1))
                positive_probabilities = positive_probabilities[:, :min_seq_len]
                negative_probabilities = negative_probabilities[:, :min_seq_len]
            elif positive_probabilities.dim() != negative_probabilities.dim():
                # If the dimensions differ, try to squeeze to the same dimension
                if positive_probabilities.dim() > 1:
                    positive_probabilities = positive_probabilities.view(-1)
                if negative_probabilities.dim() > 1:
                    negative_probabilities = negative_probabilities.view(-1)
            
            # Make sure both shapes are exactly the same
            if positive_probabilities.shape != negative_probabilities.shape:
                min_len = min(positive_probabilities.numel(), negative_probabilities.numel())
                positive_probabilities = positive_probabilities.view(-1)[:min_len]
                negative_probabilities = negative_probabilities.view(-1)[:min_len]

            # Concatenate predictions and ground-truth labels
            predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

            # Check and handle NaN or Inf (silently)
            if torch.isnan(predicts).any() or torch.isinf(predicts).any():
                predicts = torch.clamp(predicts, min=-50, max=50)

            # Compute loss
            loss = loss_func(input=predicts, target=labels)
            evaluate_losses.append(loss.item())

            # Compute evaluation metrics (using probabilities, converting logits via sigmoid)
            # Try to compute metrics even with NaN (the function filters NaN values internally)
            pred_probs = predicts.sigmoid()
            evaluate_metrics.append(get_link_prediction_metrics(predicts=pred_probs, labels=labels))
            
            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    # ------------------ pairwise metric logging ------------------
    if pairwise_tracker is not None:
        avg_l2, avg_cos = pairwise_tracker.get_avg_pairwise_metrics()
        if avg_l2 is not None:
            logging.getLogger().info(
                f'Pairwise(DyGFormer) L2_mean={avg_l2:.5f}, Cos_mean={avg_cos:.5f}'
            )
            pairwise_tracker.latest_avg_pairwise_l2 = avg_l2
            pairwise_tracker.latest_avg_pairwise_cos = avg_cos

    # ------------------ Attention weight collection and statistics ------------------
    if collect_attention and model_name in ['DyGFormer']:
        print(f"\n{'='*60}")
        print(f"{model_name} attention weight summary statistics for the four node types ({len(evaluate_idx_data_loader)} batches)")
        print(f"{'='*60}")
        
        # Run evaluation again to collect attention weights
        all_batch_summaries = []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120, desc="Collecting attention weights")
        
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

                        
            if model_name == 'DyGFormer':
                _out_pos = get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times
                )
                if len(_out_pos) == 3:
                    _, _, extra = _out_pos
                    if extra and 'attn_weights' in extra:
                        all_batch_summaries.append(extra['attn_weights'])
        
        print(f"Collected attention weight statistics from {len(all_batch_summaries)} batches")
        
        # Aggregate attention weight statistics across all batches
        if all_batch_summaries:
            category_sums = {'self': 0.0, 'key': 0.0, 'other': 0.0, 'pad': 0.0}
            count = 0
            
            for batch_summary in all_batch_summaries:
                if isinstance(batch_summary, dict):
                    # Handle the DyGFormer data structure
                    if model_name == 'DyGFormer':
                        for side in ['src', 'dst']:
                            if side in batch_summary:
                                side_data = batch_summary[side]
                                if isinstance(side_data, dict) and 'A' in side_data and isinstance(side_data['A'], dict) and 'share' in side_data['A']:
                                    share_data = side_data['A']['share']
                                    if isinstance(share_data, dict):
                                        for cat, weight in share_data.items():
                                            if cat in category_sums:
                                                category_sums[cat] += float(weight)
                                        count += 1  # Moved to the outer level, computed once per side
            
            # Output statistics
            if count > 0:
                print(f"\n{'='*60}")
                print(f"{model_name} attention weight summary statistics for the four node types ({len(all_batch_summaries)} batches)")
                print(f"{'='*60}")
                
                # Normalize
                total = sum(category_sums.values())
                if total > 0:
                    print("Average attention weight distribution (weights of the four node types sum to 1):")
                    for cat in ['self', 'key', 'other', 'pad']:
                        weight = category_sums[cat] / total
                        print(f"  {cat}: {weight:.3f} ({weight*100:.1f}%)")
                else:
                    print("No valid attention weight data collected")
                print(f"{'='*60}")
            else:
                print("Failed to aggregate attention weight data")
        else:
            print("No attention weight data collected")
    
    return evaluate_losses, evaluate_metrics


def evaluate_model_node_classification(model_name: str, model: nn.Module, neighbor_sampler, evaluate_idx_data_loader: DataLoader,
                               evaluate_data: Data, loss_func: Callable, num_neighbors: Optional[int] = 20,
                               time_gap: Optional[int] = None, edge_neighbor_sampler=None):
    """
    evaluate models on the node classification task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: Callable, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Get the logger (create it if it does not exist)
    logger = logging.getLogger()
    
    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'RepeatMixer', 'DyGFormer', 'DyGFormerF', 'DyGFormerFF', 'DyGFormerC', 'DyGFormerD', 'DyGFormerP', 'DyGFormerR', 'NPTDyG', 'NPTSTDyG', 'DyGFormerST', 'DyGFormerWSDMP', 'NPTWSDMP', 'DyGFormerWSDMPF', 'DyGFormerSPSE']:
        # evaluation phase use all the graph information
        get_model_submodule(model, 0).set_neighbor_sampler(neighbor_sampler)
    if model_name == 'RepeatMixer' and edge_neighbor_sampler is not None:
        get_model_submodule(model, 0).set_edge_neighbor_sampler(edge_neighbor_sampler)

    if model_name in ['JODIE', 'DyRep', 'TGN']:
        # reinitialize memory of memory-based models at the start of each epoch
        get_model_submodule(model, 0).memory_bank.__init_memory_bank__()

    model.eval()

    with torch.no_grad():
        # store evaluate losses, and collect all predicts/labels for metrics on full dataset
        evaluate_losses, evaluate_y_trues, evaluate_y_predicts = [], [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120, leave=False, desc="Evaluating", 
                                            mininterval=1.0, maxinterval=10.0)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                evaluate_data.src_node_ids[evaluate_data_indices], evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], \
                evaluate_data.labels[evaluate_data_indices]

            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch = {
                    'src_node_ids': batch_src_node_ids,
                    'dst_node_ids': batch_dst_node_ids,
                    'node_interact_times': batch_node_interact_times,
                    'edge_ids': batch_edge_ids,
                    'edges_are_positive': True,
                    'num_neighbors': num_neighbors
                }
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(batch)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch = {
                    'src_node_ids': batch_src_node_ids,
                    'dst_node_ids': batch_dst_node_ids,
                    'node_interact_times': batch_node_interact_times,
                    'num_neighbors': num_neighbors,
                    'time_gap': time_gap
                }
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(batch)
            elif model_name == 'RepeatMixer':
                # RepeatMixer returns single edge embedding (batch_size, num_channels)
                batch_src_node_ids_np = batch_src_node_ids.cpu().numpy() if hasattr(batch_src_node_ids, 'cpu') else batch_src_node_ids
                batch_dst_node_ids_np = batch_dst_node_ids.cpu().numpy() if hasattr(batch_dst_node_ids, 'cpu') else batch_dst_node_ids
                batch_times_np = batch_node_interact_times.cpu().numpy() if hasattr(batch_node_interact_times, 'cpu') else batch_node_interact_times
                batch_src_node_embeddings = get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_src_node_ids_np,
                    dst_node_ids=batch_dst_node_ids_np,
                    node_interact_times=batch_times_np,
                    num_neighbors=num_neighbors,
                    time_gap=time_gap
                )
            elif model_name == 'DyGFormerD':
                # node_t: look up by (node,t); old edge_id: look up by edge_ids
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        edge_ids=batch_edge_ids,
                    )
            elif model_name == 'DyGFormerST':
                # The node classification task has no negative samples, all edges are real, so pass edge_ids directly to look up the precomputed SPD table (misses computed online)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        edge_ids=batch_edge_ids,
                    )
            elif model_name in ['DyGFormer', 'DyGFormerF', 'DyGFormerFF', 'DyGFormerC', 'DyGFormerP', 'DyGFormerR', 'NPTDyG', 'NPTSTDyG', 'DyGFormerWSDMP', 'NPTWSDMP', 'DyGFormerWSDMPF', 'DyGFormerSPSE']:
                # get temporal embedding of source and destination nodes
                # DyGFormer and the other models here all return 2 values (src, dst), each of shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    get_model_submodule(model, 0).compute_src_dst_node_temporal_embeddings(
                        batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times
                    )
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")

            # get predicted probabilities, shape (batch_size, )
            predicts = get_model_submodule(model, 1)(x=batch_src_node_embeddings).squeeze(dim=-1).sigmoid()

            labels = torch.from_numpy(batch_labels).float().to(predicts.device)
            predicts = predicts.to(labels.dtype)  # Make sure data types match

            # Final check
            if predicts.shape != labels.shape:
                predicts = predicts.view(labels.shape)

            loss = loss_func(input=predicts, target=labels)

            evaluate_losses.append(loss.item())
            evaluate_y_trues.append(labels)
            evaluate_y_predicts.append(predicts)

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

        if model_name == 'NPTDyG' and hasattr(get_model_submodule(model, 0), 'eval_mode'):
            get_model_submodule(model, 0).eval_mode = False

    # Collect predictions and labels from all batches first, then compute AUC-ROC on the full dataset (consistent with the provided node classification statistics)
    evaluate_y_trues = torch.cat(evaluate_y_trues, dim=0)
    evaluate_y_predicts = torch.cat(evaluate_y_predicts, dim=0)

    unique_labels = torch.unique(evaluate_y_trues)
    if len(unique_labels) < 2:
        logger.warning(f"The whole evaluation set has only one class ({unique_labels[0].item()}), cannot compute ROC-AUC")
        final_metrics = {'roc_auc': np.nan}
    else:
        final_metrics = get_node_classification_metrics(predicts=evaluate_y_predicts, labels=evaluate_y_trues)

    return np.mean(evaluate_losses), final_metrics



def evaluate_edge_bank_link_prediction(args: argparse.Namespace, train_data: Data, val_data: Data, test_idx_data_loader: DataLoader,
                                       test_neg_edge_sampler: NegativeEdgeSampler, test_data: Data):
    """
    evaluate the EdgeBank model for link prediction
    :param args: argparse.Namespace, configuration
    :param train_data: Data, train data
    :param val_data: Data, validation data
    :param test_idx_data_loader: DataLoader, test index data loader
    :param test_neg_edge_sampler: NegativeEdgeSampler, test negative edge sampler
    :param test_data: Data, test data
    :return:
    """
    # generate the train_validation split of the data: needed for constructing the memory for EdgeBank
    train_val_data = Data(src_node_ids=np.concatenate([train_data.src_node_ids, val_data.src_node_ids]),
                          dst_node_ids=np.concatenate([train_data.dst_node_ids, val_data.dst_node_ids]),
                          node_interact_times=np.concatenate([train_data.node_interact_times, val_data.node_interact_times]),
                          edge_ids=np.concatenate([train_data.edge_ids, val_data.edge_ids]),
                          labels=np.concatenate([train_data.labels, val_data.labels]))

    test_metric_all_runs = []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_result_name = f'{args.negative_sample_strategy}_negative_sampling_{args.model_name}_seed{args.seed}'
        suffix = build_run_name_suffix(args, include_batch=True)
        if suffix:
            args.save_result_name += '_' + suffix

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/{str(time.time())}.log")
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

        loss_func = nn.BCELoss()

        # evaluate EdgeBank
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
        assert test_neg_edge_sampler.seed is not None
        test_neg_edge_sampler.reset_random_state()

        test_losses, test_metrics = [], []
        test_idx_data_loader_tqdm = tqdm(test_idx_data_loader, ncols=120)

        for batch_idx, test_data_indices in enumerate(test_idx_data_loader_tqdm):
            test_data_indices = test_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times = \
                test_data.src_node_ids[test_data_indices], test_data.dst_node_ids[test_data_indices], \
                test_data.node_interact_times[test_data_indices]

            if test_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = test_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                              batch_src_node_ids=batch_src_node_ids,
                                                                                              batch_dst_node_ids=batch_dst_node_ids,
                                                                                              current_batch_start_time=batch_node_interact_times[0],
                                                                                              current_batch_end_time=batch_node_interact_times[-1])
            else:
                _, batch_neg_dst_node_ids = test_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

            positive_edges = (batch_src_node_ids, batch_dst_node_ids)
            negative_edges = (batch_neg_src_node_ids, batch_neg_dst_node_ids)

            # incorporate the testing data before the current batch to history_data, which is similar to memory-based models
            history_data = Data(src_node_ids=np.concatenate([train_val_data.src_node_ids, test_data.src_node_ids[: test_data_indices[0]]]),
                                dst_node_ids=np.concatenate([train_val_data.dst_node_ids, test_data.dst_node_ids[: test_data_indices[0]]]),
                                node_interact_times=np.concatenate([train_val_data.node_interact_times, test_data.node_interact_times[: test_data_indices[0]]]),
                                edge_ids=np.concatenate([train_val_data.edge_ids, test_data.edge_ids[: test_data_indices[0]]]),
                                labels=np.concatenate([train_val_data.labels, test_data.labels[: test_data_indices[0]]]))

            # perform link prediction for EdgeBank
            positive_probabilities, negative_probabilities = edge_bank_link_prediction(history_data=history_data,
                                                                                       positive_edges=positive_edges,
                                                                                       negative_edges=negative_edges,
                                                                                       edge_bank_memory_mode=args.edge_bank_memory_mode,
                                                                                       time_window_mode=args.time_window_mode,
                                                                                       time_window_proportion=args.test_ratio)

            if positive_probabilities.dim() == 2 and negative_probabilities.dim() == 2:
                min_seq_len = min(positive_probabilities.size(1), negative_probabilities.size(1))
                positive_probabilities = positive_probabilities[:, :min_seq_len]
                negative_probabilities = negative_probabilities[:, :min_seq_len]

            predicts = torch.from_numpy(np.concatenate([positive_probabilities, negative_probabilities])).float()
            labels = torch.cat([torch.ones(len(positive_probabilities)), torch.zeros(len(negative_probabilities))], dim=0)

            loss = loss_func(input=predicts, target=labels)

            test_losses.append(loss.item())

            test_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

            test_idx_data_loader_tqdm.set_description(f'test for the {batch_idx + 1}-th batch, test loss: {loss.item()}')

        # store the evaluation metrics at the current run
        test_metric_dict = {}

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        result_json = {
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}'for metric_name in test_metric_dict}
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)
        logger.info(f'save negative sampling results at {save_result_path}')

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')