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
from config import logger
from data import SCMDataset
from utils import get_max_depth, load_bom_graph

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
        depth += 1
        next_depth = depth % (max_depth + 1)
        logger.info(f"\n📚 Training on samples with BOM depth <= {next_depth}")
        train_stepwise(model, next_depth)


loss_weights = {'type': 1.0, 
    'demand': 1.0, 'material': 1.0, 'location': 1.0,
    'start_time': 1.0, 'end_time': 1.0, 'request_time': 1.0,
    'commit_time': 1.0, 'quantity': 1.0
}


def compute_loss(logits, targets):
    if logits.shape[:-1] != targets.shape:
        return None  # shape mismatch, cannot compute loss

    IGNORE_INDEX = -100  # this tells PyTorch to ignore these targets in loss
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=IGNORE_INDEX
    )

def _compute_loss(logits, targets):
    MASK_THRESH = -1e5

    B, T, V = logits.shape
    logits = logits.view(B * T, V)
    targets = targets.view(B * T)

    logit_max = logits.max(dim=-1).values
    valid = logit_max > MASK_THRESH

    if valid.sum() == 0:
        return logits.sum() * 0.0  # safely returns zero loss with gradient path

    return F.cross_entropy(logits[valid], targets[valid], reduction='mean')



def train_stepwise(model=None, depth=None):
    MASK_VAL = -1e9

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

    model.train()
    for epoch in range(config['epochs']):
        total_loss = 0.0
        for src, tgt, labels in train_loader:
            src = {k: v.to(device) for k, v in src.items()}
            tgt = {k: v.to(device) for k, v in tgt.items()}
            labels = {k: v.to(device) for k, v in labels.items()}
            tgt_len = tgt['material'].shape[1]

            tgt_tokens = {
                key: tgt[key][:, :1].clone()
                for key in tgt
            }

            loss_accum = 0.0
            for t in range(1, tgt_len):
                pred = model(src, tgt_tokens)
                last_pred = {k: v[:, -1] for k, v in pred.items()}

                loss_items = defaultdict(float)

                for k in ['quantity', 'type', 'demand', 'material', 'location', 'start_time', 'end_time', 'request_time', 'commit_time']:
                    logits = last_pred[k]
                    target = labels[k][:, t]
                    max_val = target.max().item()
                    min_val = target.min().item()

                    logger.debug(f"🧪 {k} logits.shape={logits.shape}, target={target.tolist()}, min_val={min_val}, max_val={max_val}")

                    try:
                        if max_val >= logits.shape[-1] or min_val < 0:
                            logger.warning(f"⚠️ Invalid target for {k}: {target.tolist()} (logits.shape={logits.shape}, min={min_val}, max={max_val})")
                            raise ValueError("Target out of bounds")
                        '''
                        # Replace MASK_VAL with -100 for ignoring
                        if isinstance(target, torch.Tensor) and (target == MASK_VAL).any():
                            target = target.masked_fill(target == MASK_VAL, -100)
                        
                        field_loss = None
                        if (logits.max(dim=-1).values > -1e6).any():                        
                            field_loss = compute_loss(logits, target)
    
                        if (
                            field_loss is not None 
                            and isinstance(field_loss, torch.Tensor) 
                            and field_loss.requires_grad 
                            and field_loss.numel() == 1 
                            and field_loss.item() < -MASK_VAL
                        ):
                            loss_items[k] = field_loss
                        '''

                        #if logits.max().item() > -1e4:
                        if torch.isfinite(logits.max()).item():
                            field_loss = F.cross_entropy(logits, target)
                        if field_loss is not None and torch.isfinite(field_loss).item():
                            loss_items[k] = field_loss

                        '''
                        B, T, V = logits.shape
                        logits_flat = logits.view(B * T, V)
                        logit_max = logits_flat.max(dim=-1).values
                        valid = logit_max > -1e5
                        if valid.any():
                            #loss_items[k] = compute_loss(logits, target)
                        '''
                        

                        #if not torch.isfinite(loss_items[k]):
                        #    raise ValueError("Non-finite loss")
                    except Exception as e:
                        logger.error(f"❌ Skipping loss[{k}] due to {str(e)}")
                        logger.debug(f"  logits = {logits}")
                        logger.debug(f"  target = {target}")
                        loss_items[k] = torch.tensor(1e9, device=device)

                '''
                pred_q_class = last_pred['quantity']
                target_q_class = quantize_quantity(labels['quantity'][:, t])
                loss_items['quantity'] = F.cross_entropy(pred_q_class, target_q_class)
                '''

                for k, v in loss_items.items():
                    logger.info(f"🔍 Loss[{k}]: {v.item():.4f}")

                loss = sum(loss_weights[k] * loss_items[k] for k in loss_items)

                loss_accum += loss
                for key in tgt_tokens:
                    val = tgt[key][:, t].unsqueeze(1)
                    tgt_tokens[key] = torch.cat([tgt_tokens[key], val], dim=1)

            optimizer.zero_grad()
            loss_accum.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss_accum.item()

        scheduler.step(total_loss)
        if scheduler.num_bad_epochs == 0:
            logger.info(f"📉 Learning rate reduced to {optimizer.param_groups[0]['lr']:.6f}")

        for param_group in optimizer.param_groups:
            logger.info(f"📉 Learning rate: {param_group['lr']:.6f}")

        logger.info(f"Epoch {epoch+1}/{config['epochs']} - Stepwise Loss: {total_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt, labels in val_loader:
                src = {k: v.to(device) for k, v in src.items()}
                tgt = {k: v.to(device) for k, v in tgt.items()}
                labels = {k: v.to(device) for k, v in labels.items()}
                tgt_len = tgt['material'].shape[1]

                tgt_tokens = {k: torch.zeros((1, 1), dtype=v.dtype, device=device) for k, v in tgt.items()}

                loss_accum = 0.0
                for t in range(tgt_len):
                    pred = model(src, tgt_tokens)
                    last_pred = {k: v[:, -1] for k, v in pred.items()}

                    loss_items = defaultdict(float)
                    for k in ['quantity', 'demand', 'material', 'location', 'start_time', 'end_time', 'request_time', 'commit_time']:
                        logits = last_pred[k]
                        target = labels[k][:, t]
                        max_val = target.max().item()
                        min_val = target.min().item()

                        try:
                            if max_val >= logits.shape[-1] or min_val < 0:
                                raise ValueError("Target out of bounds")
                            '''
                            # Replace MASK_VAL with -100 for ignoring
                            if isinstance(target, torch.Tensor) and (target == MASK_VAL).any():
                                target = target.masked_fill(target == MASK_VAL, -100)
                            
                            field_loss = None
                            if (logits.max(dim=-1).values > -1e6).any():                        
                                field_loss = compute_loss(logits, target)
                            if (
                                    field_loss is not None 
                                    and isinstance(field_loss, torch.Tensor) 
                                    and field_loss.requires_grad 
                                    and field_loss.numel() == 1 
                                    and field_loss.item() < -MASK_VAL
                                ):
                                loss_items[k] = field_loss
                            '''

                            #if logits.max().item() > -1e4:
                            if torch.isfinite(logits.max()).item():
                                field_loss = F.cross_entropy(logits, target)
                            if field_loss is not None and torch.isfinite(field_loss).item():
                                loss_items[k] = field_loss
                                
                            '''
                            B, T, V = logits.shape
                            logits_flat = logits.view(B * T, V)
                            logit_max = logits_flat.max(dim=-1).values
                            valid = logit_max > -1e5
                            if valid.any():                            
                                #loss_items[k] = compute_loss(logits, target)
                                #loss_items[k] = F.cross_entropy(logits[valid], target[valid], reduction='mean')
                                #loss_items[k] = F.cross_entropy(logits, target, reduction='mean')
                                loss_items[k] = F.cross_entropy(logits, target)
                            '''

                            #if not torch.isfinite(loss_items[k]):
                            #    raise ValueError("Non-finite loss")
                        except Exception as e:
                            logger.warning(f"⚠️ Validation loss[{k}] skipped: {str(e)}")
                            loss_items[k] = torch.tensor(1e9, device=device)
                    '''
                    try:
                        pred_q_class = last_pred['quantity']
                        target_q_class = quantize_quantity(labels['quantity'][:, t])
                        loss_items['quantity'] = F.cross_entropy(pred_q_class, target_q_class)
                    except Exception as e:
                        logger.warning(f"⚠️ Validation loss[quantity] skipped: {str(e)}")
                        loss_items['quantity'] = torch.tensor(1e9, device=device)
                    '''

                    val_loss_step = sum(loss_weights.get(k, 1.0) * loss_items[k] for k in loss_items)
                    loss_accum += val_loss_step

                    for key in tgt_tokens:
                        val = tgt[key][:, t].unsqueeze(1)
                        tgt_tokens[key] = torch.cat([tgt_tokens[key], val], dim=1)
                try:
                    val_loss += loss_accum.item()
                except Exception as e:
                    val_loss += loss_accum
        model.train()
        logger.info(f"🔍 Validation Loss: {val_loss:.4f}")

    save_model(model, depth)

