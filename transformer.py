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

from config import logger, config, get_token_type
from utils import load_bom

# --- Embedding Module (Updated) ---
class SCMEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        self.type_emb = nn.Embedding(config['num_token_types'], d_model)
        self.loc_emb = nn.Embedding(config['num_locations'], d_model)
        self.time_emb = nn.Embedding(config['num_time_steps'], d_model)
        self.demand_emb = nn.Embedding(config['num_demands'], d_model)
        self.mat_emb = nn.Embedding(config['num_materials'], d_model)
        self.method_emb = nn.Embedding(config['num_methods'], d_model)

        #num_quantity_bins = int(1e6 // config['quantity_scale'])
        #self.quantity_emb = nn.Embedding(num_quantity_bins, d_model)
        self.quantity_emb = nn.Embedding(config['max_quantity'], d_model)

        #self.quantity_proj = nn.Sequential(
        #    nn.Linear(1, d_model),
        #    nn.ReLU(),
        #    nn.LayerNorm(d_model)
        #)

        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, tokens):
        # Helper: ensure tokens are (B, T)
        def ensure_2d(x):
            if x.dim() == 1:
                return x.unsqueeze(0)
            elif x.dim() == 3:
                return x.squeeze(1)
            return x

        for k in tokens:
            tokens[k] = ensure_2d(tokens[k])

        # Standard token embeddings
        e_type = self.type_emb(tokens['type'])
        e_loc = self.loc_emb(tokens['location'])
        e_src_loc = self.loc_emb(tokens['source_location'])
        e_time = self.time_emb(tokens['time'])
        e_start = self.time_emb(tokens['start_time'])
        e_end = self.time_emb(tokens['end_time'])
        e_req = self.time_emb(tokens['request_time'])
        e_commit = self.time_emb(tokens['commit_time'])

        e_demand = self.demand_emb(tokens['demand'])
        e_mat = self.mat_emb(tokens['material'])
        e_method = self.method_emb(tokens['method'])


        # Convert float quantity to bin index
        #quantity_bins = (tokens['quantity'].float() / config['quantity_scale']).long().clamp(min=0, max=self.quantity_emb.num_embeddings - 1)
        e_qty = self.quantity_emb(tokens['quantity'])
        #e_qty = self.quantity_proj(tokens['quantity'].unsqueeze(-1).float())

        # BOM-specific
        e_parent = self.mat_emb(tokens['parent'])
        e_child = self.mat_emb(tokens['child'])

        # Mask: 1 if type is 'bom', 0 otherwise
        is_bom = (tokens['type'] == get_token_type('bom')).unsqueeze(-1).float()

        e_combined = (
            e_type + e_loc + e_src_loc + e_time + e_start + e_end +
            e_req + e_commit + e_demand + e_mat + e_method + e_qty
        )
        e_bom = e_parent + e_child

        embeddings = (1 - is_bom) * e_combined + is_bom * e_bom
        return self.dropout(embeddings)


# --- Transformer Model ---
DEBUG_FORWARD = True
class SCMTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = SCMEmbedding(config)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['n_heads'],
            dim_feedforward=config['d_ff'],
            dropout=config['dropout'],
            batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config['d_model'],
            nhead=config['n_heads'],
            dim_feedforward=config['d_ff'],
            dropout=config['dropout'],
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['n_layers'])
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config['n_layers'])

        d_model = config['d_model']
        self.demand_out = nn.Linear(d_model, config['num_demands'])
        self.type_out = nn.Linear(d_model, config['num_token_types'])
        self.material_out = nn.Linear(d_model, config['num_materials'])
        
        self.location_out = nn.Linear(d_model, config['num_locations'])
        self.source_location_out = nn.Linear(d_model, config['num_locations'])

        self.time_out = nn.Linear(d_model, config['num_time_steps'])
        self.start_time_out = nn.Linear(d_model, config['num_time_steps'])
        self.end_time_out = nn.Linear(d_model, config['num_time_steps'])
        self.request_time_out = nn.Linear(d_model, config['num_time_steps'])
        self.commit_time_out = nn.Linear(d_model, config['num_time_steps'])

        #self.quantity_out = nn.Linear(d_model, 1)
        self.method_out = nn.Linear(d_model, config['num_methods'])

        #self.quantity_out = nn.Linear(d_model, int(1e6 // config['quantity_scale']))
        self.quantity_out = nn.Linear(d_model, config['max_quantity'])

        #self.ref_id_out = nn.Linear(d_model, 64)  # Assume 64 is max number of ref_ids
        #self.depends_on_out = nn.Linear(d_model, 64)  # Same assumption

    def apply_bom_mask(self, logits, src_tokens, tgt_tokens):
        from collections import defaultdict

        #child_map = defaultdict(set)
        #for parent, child in bom_edges:
        #    child_map[parent].add(child)
        child_map = load_bom()
        batch_size = logits.size(0)
        allowed_mask = torch.full_like(logits, float('-inf'))

        for b in range(batch_size):
            allowed = set()
            for i, token_type in enumerate(src_tokens['type'][b]):
                if token_type == get_token_type('demand'):  # Only consider demand tokens
                    allowed.add(src_tokens['material'][b][i].item())
            if tgt_tokens is not None:
                for i, token_type in enumerate(tgt_tokens['type'][b]):
                    mat = tgt_tokens['material'][b][i].item()
                    if token_type == get_token_type('make'):  # make
                        allowed.update(child_map.get(mat, []))
                    if token_type == get_token_type('purchase') or token_type == get_token_type('move') or token_type == get_token_type('demand'):
                        allowed.add(mat)

            if not allowed:
                allowed = {0}  # fallback

            for mat_id in allowed:
                if mat_id < logits.shape[-1]:
                    allowed_mask[b, :, mat_id] = 0.0

        return logits + allowed_mask
 
    def apply_field_constraints(self, logits_dict, prev_tokens):
        MASK_VAL = -1e9  # safer alternative to float('-inf')

        batch_size = prev_tokens['type'].size(0)

        for b in range(batch_size):
            last = {k: prev_tokens[k][b][-1].item() for k in prev_tokens}

            if last['type'] == get_token_type('demand'): #1:  demand
                logits_dict['end_time'][b, :, last['commit_time'] + 1:] = float('-inf')

            if last['type'] == get_token_type('move'):  # move
                logits_dict['location'][b, :, :] = float('-inf')
                logits_dict['location'][b, :, last['source_location']] = 0.0
            else:
                logits_dict['location'][b, :, :] = float('-inf')
                logits_dict['location'][b, :, last['location']] = 0.0

            logits_dict['demand'][b, :, :] = float('-inf')
            logits_dict['demand'][b, :, last['demand']] = 0.0

            logits_dict['material'][b, :, :] = float('-inf')
            logits_dict['material'][b, :, last['material']] = 0.0

            logits_dict['end_time'][b, :, last['start_time'] + 1:] = float('-inf')

            # Enforce quantity equality
            if 'quantity' in logits_dict:
                #logits_dict['quantity'][b, :] = float('-inf')
                #logits_dict['quantity'][b, -1] = last['quantity']
                logits_dict['quantity'][b, :, :] = float('-inf')
                logits_dict['quantity'][b, :, last['quantity']] = 0.0

        return logits_dict

    def _apply_field_constraints(self, logits_dict, prev_tokens):
        MASK_VAL = -1e9  # safer alternative to float('-inf')

        batch_size = prev_tokens['type'].size(0)

        for b in range(batch_size):
            #seq_size = prev_tokens['type'][b].size(0)
            seq_size = prev_tokens['type'].size(1)
            for seq in range(seq_size):

                last = {k: prev_tokens[k][b][seq].item() for k in prev_tokens}

                if last['type'] == get_token_type('demand'): #1:  demand
                    logits_dict['end_time'][b, seq, last['commit_time'] + 1:] = float('-inf')
                else:
                    logits_dict['end_time'][b, seq, last['start_time'] + 1:] = float('-inf')

                if last['type'] == get_token_type('move'):  # move
                    logits_dict['location'][b, seq, :] = float('-inf')
                    logits_dict['location'][b, seq, last['source_location']] = 0.0
                else:
                    logits_dict['location'][b, seq, :] = float('-inf')
                    logits_dict['location'][b, seq, last['location']] = 0.0

                logits_dict['demand'][b, seq, :] = float('-inf')
                logits_dict['demand'][b, seq, last['demand']] = 0.0

                logits_dict['material'][b, seq, :] = float('-inf')
                logits_dict['material'][b, seq, last['material']] = 0.0

                

                # Enforce quantity equality
                #if 'quantity' in logits_dict:
                #    logits_dict['quantity'][b, seq] = float('-inf')
                #    logits_dict['quantity'][b, seq] = last['quantity']

                logits_dict['quantity'][b, seq, :] = float('-inf')
                logits_dict['quantity'][b, seq, last['quantity']] = 0.0

        # additional constraints
        
        next_type = logits_dict['type'].argmax(dim=-1)
        demand_type = get_token_type('demand')
        make_type = get_token_type('make')
        purchase_type = get_token_type('purchase')
        move_type = get_token_type('move')

        for b in range(next_type.size(0)):
                for t in range(next_type.size(1)):
                    t_type = next_type[b, t].item()
                    if t_type == demand_type:
                        shared_time = logits_dict['commit_time'][b, t]
                        logits_dict['start_time'][b, t] = shared_time
                        logits_dict['end_time'][b, t] = shared_time
                    elif t_type in (make_type, purchase_type, move_type):
                        shared_time = logits_dict['end_time'][b, t]
                        logits_dict['request_time'][b, t] = shared_time
                        logits_dict['commit_time'][b, t] = shared_time
        
        return logits_dict
       
    def forward(self, src_tokens, tgt_tokens, bom_edges=None):
        assert (src_tokens['demand'] < config['num_demands']).all(), "demand_id out of range"
        assert (src_tokens['material'] < config['num_materials']).all(), "material_id out of range"

        src = self.embed(src_tokens)
        tgt = self.embed(tgt_tokens)

        memory = self.encoder(src)
        decoded = self.decoder(tgt, memory)

        material_logits = self.material_out(decoded)
        #if bom_edges is not None:
        material_logits = self.apply_bom_mask(material_logits, src_tokens, tgt_tokens)

        quantity_pred = self.quantity_out(decoded).squeeze(-1)
        quantity_pred = torch.clamp(quantity_pred, min=0.0, max=1e6)  # Clamp prediction for stability

        output_logits = {
            'demand': self.demand_out(decoded),
            'type': self.type_out(decoded),
            'material': material_logits,

            'location': self.location_out(decoded),
            'source_location': self.source_location_out(decoded),

            'start_time': self.start_time_out(decoded),
            'end_time': self.end_time_out(decoded),
            'request_time': self.request_time_out(decoded),
            'commit_time': self.commit_time_out(decoded),            

            'quantity': self.quantity_out(decoded),
            #'quantity': quantity_pred, #self.quantity_out(decoded).squeeze(-1),

            #'method': self.method_out(decoded),
            #'parent': self.material_out(decoded),
            #'child': self.material_out(decoded),
            #'parent': material_logits,
            #'child': material_logits
        }
        if tgt_tokens is not None:
            output_logits = self.apply_field_constraints(output_logits, tgt_tokens)

        def decode_val(key, out, use_argmax=True):
            val = out[key][0, -1]
            if key == 'type':
                val = val[..., :4]
            if use_argmax:
                return val.argmax(-1).item()
            if val.numel() == 1:
                return val.item()
            return val.tolist()
    
        if DEBUG_FORWARD and self.training:
            logger.info("\n🔎 Debug Forward Pass:")
            #print("src_tokens['type']:", src_tokens['type'][0].tolist())
            logger.info(f"tgt_tokens['type']: {tgt_tokens['type'][0].tolist()}")
            #logger.info(f"Output logits['quantity']: {output_logits['quantity'][0, -1].detach().cpu().numpy()}")
            #logger.info(f"Ground truth quantity: {tgt_tokens['quantity'][0, :10]}")
            logger.info(f"Output logits['quantity'] (bin idx): {output_logits['quantity'][0, -1].argmax(-1).item()}")
            #logger.info(f"Predicted quantity: {dequantize_quantity(output_logits['quantity'][0, -1].argmax(-1))}")
            logger.info(f"Predicted quantity: {output_logits['quantity'][0, -1].argmax(-1)}")
            logger.info(f"Ground truth quantity: {tgt_tokens['quantity'][0, :10]}")

            if isinstance(output_logits, dict):
                decoded = {}
                for key in output_logits:
                    use_argmax = True
                    #use_argmax = key in ['type', 'material', 'location', 'source_location', 'start_time', 'end_time', 'request_time', 'commit_time', 'demand']
                    pred = decode_val(key, output_logits, use_argmax=use_argmax)
                    #pred = decode_val(key, output_logits)
                    #if key == 'quantity':
                    #    pred = dequantize_quantity(pred)
                    decoded[key] = pred

                logger.info("🔢 Predicted next token:")
                for k, v in decoded.items():
                    logger.info(f"  {k}: {v}")

            if tgt_tokens['type'].size(1) >= 2:  # Ensure we have a "previous" token
                prev_idx = -2
                prev = {k: tgt_tokens[k][0, prev_idx].item() for k in ['type', 'end_time', 'location', 'source_location', 'demand', 'material', 'quantity']}
                logger.debug("🔁 Previous token:")
                for k, v in prev.items():
                    logger.debug(f"  {k}: {v}")

                logger.debug("\n🔍 Constraint Check:")
                if prev['type'] == get_token_type('demand'):
                    logger.debug("✔ Constraint (end_time <= commit_time)? Skipped for now")
                if prev['type'] == get_token_type('move'):
                    logger.debug(f"✔ Constraint: location == source_location? {decoded['location'] == prev['source_location']}")
                else:
                    logger.debug(f"✔ Constraint: location == location? {decoded['location'] == prev['location']}")
                logger.debug(f"✔ Constraint: demand match? {decoded['demand'] == prev['demand']}")
                logger.debug(f"✔ Constraint: material match? {decoded['material'] == prev['material']}")
                logger.debug(f"✔ Constraint: quantity match? {decoded['quantity'] == prev['quantity']}")
                logger.debug(f"✔ Constraint: start_time >= end_time? {decoded['start_time'] >= prev['end_time']}")

        return output_logits


def restore_model(model=None):
    if model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SCMTransformerModel(config).to(device)

    depth = -1
    model_path = "models"
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
    else:
        model_files = sorted(Path(model_path).glob("*")) 
        length = len(model_files)
        if length > 0: 
            depth = length - 1
            logger.info(f"Restore model from {model_files[-1]}")          
            model.load_state_dict(torch.load(model_files[-1]))

    return depth, model

def save_model(model, depth):
    model_file_name = f"models/{config['checkpoint_name']}_depth_{depth}.pt"
    torch.save(model.state_dict(), model_file_name)
    logger.info(f"✅ Model saved to {model_file_name}")

