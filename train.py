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

loss_weights = {
    # Core action decisions
    'type': 2.0,                # pick make/move/purchase correctly
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
    'successor': 2.0            # likewise
}

def learn(model, data_loader, device, optimizer=None):
    total_loss = 0.0
    for src, tgt, labels in data_loader:
        src = {k: v.to(device) for k, v in src.items()}
        tgt = {k: v.to(device) for k, v in tgt.items()}
        labels = {k: v.to(device) for k, v in labels.items()}
        tgt_len = tgt['material'].shape[1]

        #tgt_tokens = {key: tgt[key][:, :1].clone() for key in tgt}
        tgt_tokens = {k: torch.zeros((1, 1), dtype=v.dtype, device=device) for k, v in tgt.items()}

        loss_accum = 0.0
        #for t in range(1, tgt_len):
        for t in range(tgt_len):
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

                    #if logits.max().item() > -1e4:
                    if torch.isfinite(logits.max()).item():
                        field_loss = F.cross_entropy(logits, target)
                    if field_loss is not None and torch.isfinite(field_loss).item():
                        loss_items[k] = field_loss

                    #if not torch.isfinite(loss_items[k]):
                    #    raise ValueError("Non-finite loss")
                except Exception as e:
                    logger.error(f"❌ Skipping loss[{k}] due to {str(e)}")
                    logger.debug(f"  logits = {logits}")
                    logger.debug(f"  target = {target}")
                    loss_items[k] = torch.tensor(1e9, device=device)

            eod_logits = last_pred['eod']
            #eod_labels = (labels['type'][:, t] == get_token_type('eod')).float()
            eod_labels = ((labels["seq_in_demand"][:, t] + 1) == labels["total_in_demand"][:, t]).float()
            eod_loss = F.binary_cross_entropy_with_logits(eod_logits, eod_labels, reduction="mean")
            loss_items['eod'] = eod_loss
                            
            for k, v in loss_items.items():
                logger.info(f"🔍 Loss[{k}]: {v.item():.4f}")

            loss = sum(loss_weights[k] * loss_items[k] for k in loss_items)

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

    save_model(model, depth)

