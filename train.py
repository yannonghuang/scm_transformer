import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Any
import random
import argparse
import os
import pandas as pd
from pathlib import Path
import networkx as nx
import csv
from collections import defaultdict
import logging
from logging.handlers import RotatingFileHandler

from config import config
from transformer import SCMTransformerModel, restore_model, save_model
from config import logger, get_token_type
from data import SCMDataset
from utils import get_max_depth, load_bom_graph

loss_weights = {
    # Core action decisions
    'type': 5.0,                # pick make/move/purchase correctly
    'material': 2.0,            # critical to right output
    'location': 1.5,            # mildly less critical than material

    # Temporal correctness
    'start_time': 1.0,
    'end_time': 1.0,
    'commit_time': 1.0,
    'lead_time': 0.5,           # often derived from others

    # Quantitative accuracy
    'quantity': 1.0,

    # Reference demand matching
    'demand': 1.0,
    'request_time': 0.5,        # often repeated or inferred

    # Sequence structure
    'eod': 2.0,                 # very important for stopping correctly
    'seq_in_demand': 1.5,       # helpful but secondary
    'total_in_demand': 0.5,     # likewise
    'successor': 2.0,            # likewise

    'demand_workorder_sync_loss' : 0.01
    #'demand_workorder_endtime_loss': 0.01
}

def enforce_demand_end_time_constraint(tgt_tokens):
    """
    Overwrite demand fields based on matched workorder end_time.
    """
    for b in range(tgt_tokens['type'].shape[0]):
        # Get all demand token indices
        demand_indices = (tgt_tokens['type'][b] == get_token_type('demand')).nonzero(as_tuple=False).squeeze(-1)
        for idx in demand_indices:
            demand_id = tgt_tokens['demand'][b][idx].item()

            # Look for the workorder with successor==0 and same demand ID
            mask = (
                (tgt_tokens['demand'][b] == demand_id) &
                (tgt_tokens['successor'][b] == 0) &
                (tgt_tokens['type'][b] != get_token_type('demand'))  # make sure it's not the demand token itself
            )
            work_indices = mask.nonzero(as_tuple=False).squeeze(-1)
            if len(work_indices) > 0:
                work_end_time = tgt_tokens['end_time'][b][work_indices[0]].item()
                tgt_tokens['commit_time'][b][idx] = work_end_time
                tgt_tokens['start_time'][b][idx] = work_end_time
                tgt_tokens['end_time'][b][idx] = work_end_time

    return tgt_tokens

def demand_workorder_endtime_loss(tgt_tokens, logits_dict, loss_fn=torch.nn.MSELoss()):
    """
    Supervises commit_time prediction for demand tokens,
    and optionally regularizes alignment between demand.commit_time and workorder.end_time.
    """
    loss = 0.0
    count = 0

    pred_commit = logits_dict['commit_time'].argmax(dim=-1)  # [B, T]
    pred_end    = logits_dict['end_time'].argmax(dim=-1)

    B, T = pred_commit.shape

    for b in range(B):
        demand_mask = (tgt_tokens['type'][b] == get_token_type('demand'))
        demand_indices = demand_mask.nonzero(as_tuple=False).squeeze(-1)

        for idx in demand_indices:
            demand_id = tgt_tokens['demand'][b][idx].item()
            true_commit = tgt_tokens['commit_time'][b][idx].float()
            pred_c = pred_commit[b][idx].float()

            # Primary loss: prediction vs ground truth
            loss += loss_fn(pred_c, true_commit)
            count += 1

            # Optional: Regularization — align with first workorder.end_time
            work_mask = (
                (tgt_tokens['demand'][b] == demand_id) &
                (tgt_tokens['successor'][b] == 0) &
                (tgt_tokens['type'][b] != get_token_type('demand'))
            )
            work_indices = work_mask.nonzero(as_tuple=False).squeeze(-1)

            if len(work_indices) > 0:
                j = work_indices[0].item()
                pred = pred_end[b][j].float()
                true = tgt_tokens['end_time'][b][j].float()
                loss += 0.1 * loss_fn(pred, true)
                count += 1

    return loss / max(count, 1)

def _demand_workorder_endtime_loss(tgt_tokens, logits_dict, loss_fn=torch.nn.MSELoss()):
    loss = 0.0
    count = 0

    pred_end = logits_dict['end_time'].argmax(dim=-1)
    B, T = pred_end.shape

    for b in range(B):
        demand_indices = (tgt_tokens['type'][b] == get_token_type('demand')).nonzero(as_tuple=False).squeeze(-1)
        for idx in demand_indices:
            demand_id = tgt_tokens['demand'][b][idx].item()
            mask = (
                (tgt_tokens['demand'][b] == demand_id) &
                (tgt_tokens['successor'][b] == 0) &
                (tgt_tokens['type'][b] != get_token_type('demand'))
            )
            work_indices = mask.nonzero(as_tuple=False).squeeze(-1)
            if len(work_indices) > 0:
                j = work_indices[0].item()
                pred = pred_end[b][j].float()
                true = tgt_tokens['end_time'][b][j].float()
                loss += 0.1 * loss_fn(pred, true)
                count += 1

    return loss / max(count, 1)


def demand_workorder_sync_loss(tgt_tokens, logits_dict, loss_fn=torch.nn.MSELoss()):
    loss = 0.0
    count = 0

    pred_commit = logits_dict['commit_time'].argmax(dim=-1)  # [B, T]
    pred_start  = logits_dict['start_time'].argmax(dim=-1)
    pred_end    = logits_dict['end_time'].argmax(dim=-1)

    B, T = pred_commit.shape

    for b in range(B):
        demand_indices = (tgt_tokens['type'][b] == get_token_type('demand')).nonzero(as_tuple=False).squeeze(-1)
        for idx in demand_indices:
            demand_id = tgt_tokens['demand'][b][idx].item()
            mask = (
                (tgt_tokens['demand'][b] == demand_id) &
                (tgt_tokens['successor'][b] == 0) &
                (tgt_tokens['type'][b] != get_token_type('demand'))
            )
            work_indices = mask.nonzero(as_tuple=False).squeeze(-1)
            if len(work_indices) > 0:
                j = work_indices[0].item()
                loss += loss_fn(pred_commit[b][idx].float(), pred_end[b][j].float())
                loss += loss_fn(pred_start[b][idx].float(), pred_end[b][j].float())
                loss += loss_fn(pred_end[b][idx].float(),   pred_end[b][j].float())
                count += 1

    return loss / max(count, 1)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SCMTransformerModel(config).to(device)
    
    depth, model = restore_model(model)

    max_depth = None
    if args.depth:
        max_depth = (int)((vars(args))['depth'])
        print(f"CLI gives max_depth = {max_depth}")
    if max_depth is None:
        max_depth = get_max_depth(load_bom_graph())

    for d in range(max_depth + 1):
        next_depth = depth % (max_depth + 1)
        logger.info(f"\n📚 Training on samples with BOM depth <= {next_depth}")
        train_stepwise(model, next_depth)
        depth += 1

def NEW_learn(model, data_loader, device, optimizer=None):
    total_loss = 0.0
    model.train() if optimizer is not None else model.eval()

    for src, tgt, labels in data_loader:
        src = {k: v.to(device) for k, v in src.items()}
        tgt = {k: v.to(device) for k, v in tgt.items()}
        labels = {k: v.to(device) for k, v in labels.items()}

        # Model forward pass with full teacher-forcing target
        pred = model(src, tgt)  # shape: {field: (B, T, C)}

        loss_items = defaultdict(float)

        # 🔧 Supervised fields to compute cross-entropy on
        supervised_fields = [
            'quantity', 'type', 'demand', 'material', 'location',
            'start_time', 'end_time', 'request_time', 'commit_time',
            'lead_time', 'seq_in_demand', 'total_in_demand', 'successor'
        ]

        for k in supervised_fields:
            logits = pred[k]              # (B, T, C)
            target = labels[k]            # (B, T)

            if logits.dim() != 3:
                logger.warning(f"⚠️ Skipping loss[{k}] due to invalid logits shape: {logits.shape}")
                continue

            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = target.view(B * T)

            try:
                if target.max().item() >= C or target.min().item() < 0:
                    logger.warning(f"⚠️ Invalid target for {k}: min={target.min().item()}, max={target.max().item()}, logits.shape={logits.shape}")
                    continue

                field_loss = F.cross_entropy(logits, target, ignore_index=-1)
                if torch.isfinite(field_loss):
                    loss_items[k] = field_loss
                    logger.info(f"🔍 Loss[{k}]: {field_loss.item():.4f}")
            except Exception as e:
                logger.error(f"❌ Skipping loss[{k}] due to {str(e)}")

        # 🧩 EOD binary loss
        if "eod" in pred:
            eod_logits = pred["eod"]  # (B, T)
            if torch.isfinite(eod_logits.max()):
                eod_labels = ((labels["seq_in_demand"] + 1) == labels["total_in_demand"]).float()  # (B, T)
                eod_loss = F.binary_cross_entropy_with_logits(eod_logits, eod_labels, reduction="mean")
                if torch.isfinite(eod_loss):
                    loss_items["eod"] = eod_loss
                    logger.info(f"🔍 Loss[eod]: {eod_loss.item():.4f}")

        '''
        # ⚖️ Demand-Workorder time sync constraint (custom soft constraint)
        sync_loss = demand_workorder_endtime_loss(tgt, pred)
        if torch.isfinite(sync_loss):
            loss_items["demand_workorder_endtime_loss"] = sync_loss
            logger.info(f"🔍 Loss[demand_workorder_endtime_loss]: {sync_loss.item():.4f}")
        '''

        # 🎯 Final total loss
        loss = sum(loss_weights[k] * v for k, v in loss_items.items())
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(float(loss), device=device, requires_grad=True)
        total_loss += loss.item()


        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    return total_loss

def NEW_train_stepwise(model=None, depth=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 🌱 Initialize model if not provided
    if model is None:
        model = SCMTransformerModel(config).to(device)

    # 🔧 Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    # 📂 Load dataset
    sample_path = os.path.join('data', 'samples', f'depth_{depth}')
    if not os.path.exists(sample_path):
        logger.warning(f"⚠️ Sample data {sample_path} does not exist, exiting.")
        return

    dataset = SCMDataset(sample_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    logger.info(f"📚 Training set: {len(train_set)} samples, Validation set: {len(val_set)} samples")

    check_count = 0
    for epoch in range(1, config['epochs'] + 1):
        # 🏋️ Train
        logger.info(f"🚀 Epoch {epoch}/{config['epochs']}")
        model.train()
        train_loss = learn(model, train_loader, device, optimizer)
        logger.info(f"✅ Train Loss: {train_loss:.4f}")

        # 🧪 Validate
        model.eval()
        with torch.no_grad():
            val_loss = learn(model, val_loader, device)
        logger.info(f"🔍 Validation Loss: {val_loss:.4f}")

        # 📉 Learning rate schedule
        scheduler.step(val_loss)
        for param_group in optimizer.param_groups:
            logger.info(f"📉 Learning rate: {param_group['lr']:.6f}")

        if scheduler.num_bad_epochs == 0:
            logger.info(f"🎯 Learning rate reduced to {optimizer.param_groups[0]['lr']:.6f}")

        # 💾 Save model checkpoint
        check_count += 1
        if check_count >= min(config['checkpoint_frequency'], config['epochs']):
            save_model(model, depth)
            logger.info(f"💾 Model checkpoint saved (depth={depth}, epoch={epoch})")
            check_count = 0

def learn(model, data_loader, device, optimizer=None):
    total_loss = 0.0
    for src, tgt, labels in data_loader:
        src = {k: v.to(device) for k, v in src.items()}
        tgt = {k: v.to(device) for k, v in tgt.items()}
        labels = {k: v.to(device) for k, v in labels.items()}
        tgt_len = tgt['material'].shape[1]

        tgt_tokens = {key: tgt[key][:, :1].clone() for key in tgt}
        #tgt_tokens = {k: torch.zeros((1, 1), dtype=v.dtype, device=device) for k, v in tgt.items()}

        loss_accum = 0.0
        for t in range(1, tgt_len):
        #for t in range(tgt_len):

            # Before model forward pass (to correct hard targets)
            tgt_tokens = enforce_demand_end_time_constraint(tgt_tokens)

            pred = model(src, tgt_tokens)
            last_pred = {k: v[:, -1] for k, v in pred.items()}

            loss_items = defaultdict(float)

            for k in ['quantity', 'type', 'demand', 'material', 'location', 'start_time', 'end_time', 'request_time', 'commit_time', 'lead_time', 'seq_in_demand', 'total_in_demand', 'successor']:
            #for k in ['quantity', 'type', 'demand', 'material', 'location', 'start_time', 'end_time', 'request_time', 'commit_time', 'lead_time']:
                logits = last_pred[k]
                target = labels[k][:, t]
                max_val = target.max().item()
                min_val = target.min().item()

                logger.debug(f"🧪 {k} logits.shape={logits.shape}, target={target.tolist()}, min_val={min_val}, max_val={max_val}")

                try:
                    if max_val >= logits.shape[-1] or min_val < 0:
                        logger.warning(f"⚠️ Invalid target for {k}: {target.tolist()} (logits.shape={logits.shape}, min={min_val}, max={max_val})")
                        raise ValueError("Target out of bounds")

                    if torch.isfinite(logits.max()).item():
                        field_loss = F.cross_entropy(logits, target)
                        #if field_loss is not None and torch.isfinite(field_loss).item():
                        if torch.isfinite(field_loss).item():
                            loss_items[k] = field_loss

                    #if not torch.isfinite(loss_items[k]):
                    #    raise ValueError("Non-finite loss")
                except Exception as e:
                    logger.error(f"❌ Skipping loss[{k}] due to {str(e)}")
                    logger.debug(f"  logits = {logits}")
                    logger.debug(f"  target = {target}")
                    #loss_items[k] = torch.tensor(1e9, device=device)

            eod_logits = last_pred['eod']
            if torch.isfinite(eod_logits.max()).item():
                #eod_labels = (labels['type'][:, t] == get_token_type('eod')).float()
                eod_labels = ((labels["seq_in_demand"][:, t] + 1) == labels["total_in_demand"][:, t]).float()
                eod_loss = F.binary_cross_entropy_with_logits(eod_logits, eod_labels, reduction="mean")
                if torch.isfinite(eod_loss.max()).item():
                    loss_items['eod'] = eod_loss

            # Add soft demand time constraint penalty
            #loss_items['demand_workorder_sync_loss'] = demand_workorder_sync_loss(tgt_tokens, pred)

            for k, v in loss_items.items():
                if k == 'demand_workorder_sync_loss':
                    logger.info(f"🔍 Loss[{k}]: {v:.4f}")
                else:
                    logger.info(f"🔍 Loss[{k}]: {v.item():.4f}")

            loss = sum(loss_weights[k] * loss_items[k] for k in loss_items)
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(float(loss), device=device, requires_grad=True)

            loss_accum += loss
            for key in tgt_tokens:
                val = tgt[key][:, t].unsqueeze(1)
                tgt_tokens[key] = torch.cat([tgt_tokens[key], val], dim=1)

        if optimizer is not None:
            optimizer.zero_grad()
            loss_accum.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss_accum.item()

    return total_loss

def train_stepwise(model=None, depth=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = SCMTransformerModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    sample_path = os.path.join('data', 'samples', f'depth_{depth}')
    if not os.path.exists(sample_path):
        logger.info(f'sample data {sample_path} does not exist, exit ...')
        return

    dataset = SCMDataset(sample_path)
    train_set, val_set = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    check_count = 0
    model.train()
    for epoch in range(config['epochs']):
        total_loss = learn(model, train_loader, device, optimizer)

        scheduler.step(total_loss)
        if scheduler.num_bad_epochs == 0:
            logger.info(f"📉 Learning rate reduced to {optimizer.param_groups[0]['lr']:.6f}")

        for param_group in optimizer.param_groups:
            logger.info(f"📉 Learning rate: {param_group['lr']:.6f}")

        logger.info(f"Epoch {epoch+1}/{config['epochs']} - Stepwise Loss: {total_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_loss = learn(model, val_loader, device)
        
        model.train()
        logger.info(f"🔍 Validation Loss: {val_loss:.4f}")

        check_count += 1
        if (check_count >= min(config['checkpoint_frequency'], config['epochs'])):  
            save_model(model, depth)
            check_count = 0
    

