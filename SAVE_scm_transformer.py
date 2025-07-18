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

# Ensure the logs/ directory exists
os.makedirs("logs", exist_ok=True)

# Create a rotating file handler: 5 MB per file, keep up to 3 files
rotating_handler = RotatingFileHandler(
    "logs/training.log", maxBytes=50 * 1024 * 1024, backupCount=5
)

# Console handler to also log to stdout
console_handler = logging.StreamHandler()

# Set formatter for both handlers
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
rotating_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Get the root logger and set the level
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Set global log level here

# Add both handlers to the logger
logger.addHandler(rotating_handler)
logger.addHandler(console_handler)


# --- Config ---
token_types = [
    'demand',
    'make',
    'purchase',
    'move',
    'material',
    'location',
    'method',
    'bom'
]

def get_token_type(t):
    return token_types.index(t)

config = {
    'num_token_types': len(token_types), #9,
    'num_demands': 50,
    'num_locations': 4,
    'num_time_steps': 70,
    'num_materials': 100,
    'num_methods': 600,
    'd_model': 128,
    'n_heads': 4,
    'd_ff': 256,
    'n_layers': 4,
    'dropout': 0.1,
    'batch_size': 8,
    'lr': 1e-4,
    'epochs': 5,
    'checkpoint_name': 'scm_transformer',
    "max_train_samples": 1000,
    'quantity_scale': 1,  # updated to allow integer binning
    'max_quantity': 100 #1e5
}

def dequantize_quantity(q_class):
    return q_class * config['quantity_scale']

#def dequantize_quantity(bin_idx, scale):
#    return (bin_idx + 0.5) * scale

bom_map = defaultdict(set)
def load_bom(path="data/bom.csv") -> dict[int, set[int]]:
    global bom_map
    if bom_map is None or len(bom_map) == 0 :
        bom = defaultdict(set)
        with open(path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                parent = int(row["parent"])
                child = int(row["child"])
                bom[parent].add(child)
        #return dict(bom)
        bom_map = bom
    return bom_map

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


# --- Candidate Token Generator ---
def generate_candidate_tokens(input_dict):
    """
    Generate encoder input tokens from demand dictionary.
    Each token must include numeric fields: type, location, material, time, method_id, quantity.
    """
    candidates = []
    for d in input_dict.get("demand", []):
        candidates.append({
            "demand": d["demand_id"],
            "type": 0,  # fixed type ID for demand
            
            "location": d["location_id"],
            "source_location": d["location_id"],

            "material": d["material_id"],
            #"time": d["time"],

            "start_time": 0,
            "end_time": 0,
            "request_time": d["request_time"],
            "commit_time": 0,

            "method": 0,
            "quantity": d["quantity"],

            "parent": 0,
            "child": 0,
        })
    return candidates


# --- Updated encode_tokens with token_type_id ---
def _encode_tokens(token_list, token_type_id=1):
    # Collect all keys used in the token list
    all_keys = set()
    for token in token_list:
        all_keys.update(token.keys())

    # Determine the appropriate dtype for each key
    float_keys = {'quantity'}
    tensor_dict = {}

    for key in all_keys:
        # Use float for specific keys, otherwise long
        dtype = torch.float if key in float_keys else torch.long
        tensor_dict[key] = torch.tensor(
            [t.get(key, 0.0 if dtype == torch.float else 0) for t in token_list],
            dtype=dtype
        )

    # Optional: add token_type_id if you want to track input segments
    # tensor_dict['token_type_id'] = torch.tensor(
    #     [token_type_id] * len(token_list), dtype=torch.long
    # )

    return tensor_dict

def encode_tokens(token_list, token_type_id=1):
    def to_tensor(key, dtype=torch.long):
        return torch.tensor([t.get(key, 0) for t in token_list], dtype=dtype)
    return {
        'type': to_tensor("type"),
        'location': to_tensor("location"),
        'source_location': to_tensor("location"),

        'material': to_tensor("material"),
        'time': to_tensor("time"),

        'start_time': to_tensor("start_time"),
        'end_time': to_tensor("end_time"),
        'request_time': to_tensor("request_time"),
        'commit_time': to_tensor("commit_time"),
        'demand': to_tensor("demand"),
        'method': to_tensor("method"),

        'quantity': to_tensor("quantity"),
        #'quantity': to_tensor("quantity", dtype=torch.float),

        'parent': to_tensor("parent"),
        'child': to_tensor("child"),
        #'token_type_id': torch.tensor([token_type_id] * len(token_list), dtype=torch.long)
    }

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

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SCMTransformerModel(config).to(device)
    
    min_depth, model = restore_model(model)
    min_depth = min_depth + 1

    max_depth = None
    if args.depth:
        max_depth = (int)((vars(args))['depth'])
        print(f"CLI gives max_depth = {max_depth}")
    if max_depth is None:
        max_depth = get_max_depth(load_bom_graph())

    for depth in range(min_depth + 1, max_depth + 1):
        logger.info(f"\n📚 Training on samples with BOM depth <= {depth}")
        train_stepwise(model, depth - 1)

class SCMDataset(Dataset):
    def __init__(self, root_dir):
        self.sample_dirs = sorted(Path(root_dir).glob("*"))

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        #src = self.load_demand(sample_dir / "demands.csv")
        demand_dicts, _ = self.load_demand(sample_dir / "demands.csv")
        src = generate_encoder_input({"demand": demand_dicts})

        #tgt, labels = self.load_plan(sample_dir / "workorders.csv")
        #return src, tgt, labels
        tokens = self.load_combined(sample_dir / "combined_output.csv")
        return src, tokens, tokens

    def load_demand(self, path):
        df = pd.read_csv(path)
        demand_dicts = df.to_dict(orient="records")  # 👈 for generate_encoder_input()

        tokens = {
            "demand": torch.tensor(df["demand_id"].values, dtype=torch.long), 
            "type": torch.zeros((len(df)), dtype=torch.long),
            "location": torch.tensor(df["location_id"].values, dtype=torch.long),
            "material": torch.tensor(df["material_id"].values, dtype=torch.long),
            "time": torch.tensor(df["request_time"].values, dtype=torch.long),

            #"quantity": torch.tensor(df["quantity"].values, dtype=torch.float),
            "quantity": torch.tensor(df["quantity"].values, dtype=torch.long),

            "start_time": torch.zeros((len(df)), dtype=torch.long),
            "end_time": torch.zeros((len(df)), dtype=torch.long),
            "request_time": torch.tensor(df["request_time"].values, dtype=torch.long),
            "commit_time": torch.zeros((len(df)), dtype=torch.long),

            "parent": torch.zeros((len(df)), dtype=torch.long),
            "child": torch.zeros((len(df)), dtype=torch.long),

            "source_location": torch.zeros((len(df)), dtype=torch.long),
        }
        return demand_dicts, tokens

    def load_combined(self, path):
        df = pd.read_csv(path)
        num_bins = config['num_time_steps']

        # Clip time values to avoid IndexError
        #df["start_time"] = df["start_time"].clip(lower=0, upper=num_bins - 1).astype(int)
        #df["end_time"] = df["end_time"].clip(lower=0, upper=num_bins - 1).astype(int)

        tokens = {
            "demand": torch.tensor(df["demand_id"].values, dtype=torch.long),
            #"type": torch.zeros((1, len(df)), dtype=torch.long),
            #"type": torch.zeros((len(df)), dtype=torch.long),
            "type": torch.tensor(df["type"].values, dtype=torch.long),

            "location": torch.tensor(df["location_id"].values, dtype=torch.long), #.unsqueeze(0),
            "source_location": torch.tensor(df["source_location_id"].values, dtype=torch.long), #.unsqueeze(0),

            "material": torch.tensor(df["material_id"].values, dtype=torch.long), #.unsqueeze(0),
            "time": torch.tensor(df["start_time"].values, dtype=torch.long), #.unsqueeze(0),

            "request_time": torch.tensor(df["request_time"].values, dtype=torch.long),
            "commit_time": torch.tensor(df["commit_time"].values, dtype=torch.long),

            "start_time": torch.tensor(df["start_time"].values, dtype=torch.long), #.unsqueeze(0),
            "end_time": torch.tensor(df["end_time"].values, dtype=torch.long), #.unsqueeze(0),

            #"method_id": torch.zeros((1, len(df)), dtype=torch.long),
            "method": torch.zeros((len(df)), dtype=torch.long),

            #"quantity": torch.tensor(df["quantity"].values, dtype=torch.float), #.unsqueeze(0),
            "quantity": torch.tensor(df["quantity"].values, dtype=torch.long),

            "parent": torch.zeros((len(df)), dtype=torch.long),
            "child": torch.zeros((len(df)), dtype=torch.long),

            #"token_type_id": torch.full((1, len(df)), 2, dtype=torch.long),
            #"token_type_id": torch.zeros((len(df)), dtype=torch.long),
        }
 
        return tokens
    
def update_config_from_static_data(config, logs_root="data/logs"):
    max_material = -1
    max_location = -1

    #for sample_dir in Path(logs_root).glob("sample_*"):
    for sample_dir in Path(logs_root).glob("depth_*/sample_*"):
        for file_name in ["demands.csv", "combined_output.csv"]:
            fpath = sample_dir / file_name
            if fpath.exists():
                df = pd.read_csv(fpath)
                if "material_id" in df.columns:
                    max_material = max(max_material, df["material_id"].max())
                if "location_id" in df.columns:
                    max_location = max(max_location, df["location_id"].max())

    config['num_materials'] = max_material + 1
    config['num_locations'] = max_location + 1
    logger.info(f"🔧 Config updated: {config['num_materials']} materials, {config['num_locations']} locations")

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

    sample_path = os.path.join('data', 'logs', f'depth_{depth}')
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

# --- Predict Plan ---


def load_bom_graph(path="data/bom.csv"):
    df = pd.read_csv(path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(int(row['parent']), int(row['child']))
    return G

def compute_material_depths(G):
    depths = {}
    for node in nx.topological_sort(G):
        if G.out_degree(node) == 0:
            depths[node] = 0
        else:
            depths[node] = 1 + max(depths[child] for child in G.successors(node))
    return depths

def get_max_depth(G):
    """Safely compute max depth of a DAG from leaves upwards."""
    memo = {}

    def dfs(node):
        node = int(node)  # Normalize node ID
        if node in memo:
            return memo[node]
        children = list(G.successors(node))
        if not children:
            memo[node] = 0
        else:
            memo[node] = 1 + max(dfs(child) for child in children)
        return memo[node]

    max_depth = 0
    for node in G.nodes:
        max_depth = max(max_depth, dfs(node))
    return max_depth



# Constants for normalization (update as appropriate for your data)
MAX_QUANTITY = 1000.0
MAX_TIME = 50

# Utility functions for normalization/denormalization
def normalize_quantity(q):
    return q / MAX_QUANTITY

def denormalize_quantity(q):
    return q * MAX_QUANTITY

def normalize_time(t):
    return t / MAX_TIME

def denormalize_time(t):
    return int(round(t * MAX_TIME))

def decode_predictions(model, src_tokens, max_steps=50, threshold=0.5, beam_width=3, sampling=None, topk=5, topp=0.9, inventory=None, bom=None, time_window=None):
    model.eval()

    def initialize_tgt():
        return {
            'demand': torch.tensor([[0]], dtype=torch.long),
            'type': torch.tensor([[0]], dtype=torch.long),
            'location': torch.tensor([[0]], dtype=torch.long),
            'material': torch.tensor([[0]], dtype=torch.long),
            'time': torch.tensor([[0]], dtype=torch.long),
            'start_time': torch.tensor([[0]], dtype=torch.long),
            'end_time': torch.tensor([[0]], dtype=torch.long),
            'request_time': torch.tensor([[0]], dtype=torch.long),
            'commit_time': torch.tensor([[0]], dtype=torch.long),
            'method': torch.tensor([[0]], dtype=torch.long),

            'quantity': torch.tensor([[0]], dtype=torch.long),

            'id': torch.tensor([[0]], dtype=torch.long),

            'parent': torch.tensor([[0]], dtype=torch.long),
            'child': torch.tensor([[0]], dtype=torch.long),

            'source_location': torch.tensor([[0]], dtype=torch.long),
        }

    def _expand_beam(tgt_tokens, new_val):
        tgt_copy = {k: v.clone() for k, v in tgt_tokens.items()}
        for key, val in new_val.items():
            if key == 'quantity':
                val = normalize_quantity(val)
            elif 'time' in key:
                val = normalize_time(val)
            val_tensor = torch.tensor([[val]], dtype=torch.float if key == 'quantity' else torch.long)
            tgt_copy[key] = torch.cat([tgt_copy[key], val_tensor], dim=1)
        return tgt_copy

    def expand_beam(tgt_tokens, new_val):
        tgt_copy = {k: v.clone() for k, v in tgt_tokens.items()}
        for key, val in new_val.items():
            if 'time' in key:
                val = normalize_time(val)
            val_tensor = torch.tensor([[val]], dtype=torch.long)
            tgt_copy[key] = torch.cat([tgt_copy[key], val_tensor], dim=1)
        return tgt_copy
    
    def _decode_val(key, out, use_argmax=True):
        val = out[key][0, -1]
        if key == 'type':
            val = val[..., :4]
        if use_argmax:
            return val.argmax(-1).item()
        return val.item() if val.numel() == 1 else val.tolist()

    def decode_val(key, out, use_argmax=True):
        val = out[key][0, -1]
        if key == 'type':
            val = val[..., :4]
        if use_argmax:
            return val.argmax(-1).item()
        if val.numel() == 1:
            return val.item()
        return val.tolist()
    
    def violates_constraints(plan, work_order):
        if inventory:
            inv_key = (work_order['material_id'], work_order['location_id'])
            if inventory.get(inv_key, 1e9) < work_order['quantity']:
                return True

        if time_window:
            if not (time_window[0] <= work_order['start_time'] <= time_window[1]):
                return True

        return False

    def is_valid_successor(plan, next_token):
        if next_token['type'] == 0:
            return True

        if bom is None:
            return True

        existing_materials = {t['material_id'] for t in plan if t['type'] != 0}
        if not existing_materials:
            demand_tokens = [t for t in plan if t['type'] == 0]
            if not demand_tokens:
                return False
            last_demand = demand_tokens[-1]
            existing_materials.add(last_demand['material_id'])

        all_allowed = set()
        for mat in existing_materials:
            all_allowed.update(bom.get(mat, []))

        return next_token['material_id'] in all_allowed

    beams = [(0.0, initialize_tgt(), [], 1)]

    for step in range(max_steps):
        new_beams = []

        for score, tgt_tokens, plan, next_id in beams:
            logger.info(f"beam search iteration {next_id}")

            with torch.no_grad():
                out = model(src_tokens, tgt_tokens)
            

            #logger.info(f"out = model(src_tokens, tgt_tokens); out['material'][0].size(0) = {out['material'][0].size(0)}, ")
            #decoded = {}
            #for key in out:
            #    use_argmax = key in ['type', 'material', 'location', 'source_location', 'start_time', 'end_time', 'request_time', 'commit_time', 'demand']
            #    pred = decode_val(key, out, use_argmax=use_argmax)
            #    #pred = decode_val(key, output_logits)
            #    decoded[key] = pred
            #logger.info("🔢 Predicted next token:")
            #for k, v in decoded.items():
            #    logger.info(f"  {k}: {v}")


            material_logits = out['material'][0, -1]

            if sampling == "topk":
                probs = F.softmax(material_logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, topk)
                idx = torch.multinomial(topk_probs, 1).item()
                m = topk_indices[idx].item()
                log_prob = torch.log(topk_probs[idx]).item()
                materials = [(log_prob, m)]

            elif sampling == "topp":
                probs = F.softmax(material_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                cutoff = (cumulative_probs > topp).nonzero(as_tuple=True)[0][0].item() + 1
                filtered_probs = sorted_probs[:cutoff]
                filtered_indices = sorted_indices[:cutoff]
                idx = torch.multinomial(filtered_probs, 1).item()
                m = filtered_indices[idx].item()
                log_prob = torch.log(filtered_probs[idx]).item()
                materials = [(log_prob, m)]

            else:
                material_probs = F.log_softmax(material_logits, dim=-1)
                topk_vals, topk_idxs = torch.topk(material_probs, beam_width)
                materials = list(zip(topk_vals.tolist(), topk_idxs.tolist()))

            for log_prob, m in materials:
                l = decode_val("location", out)
                s = denormalize_time(decode_val("start_time", out))
                e = denormalize_time(decode_val("end_time", out))
                r = denormalize_time(decode_val("request_time", out))
                c = denormalize_time(decode_val("commit_time", out))
                t = decode_val("type", out)

                #q_raw = decode_val("quantity", out, use_argmax=False)
                #q = denormalize_quantity(float(q_raw[0]) if isinstance(q_raw, list) else float(q_raw))
                q = decode_val("quantity", out)

                d = decode_val("demand", out)

                #if t >= 3:
                #    continue
                #if q < threshold:
                #    continue

                work_order = {
                    "id": next_id,
                    "demand_id": d,
                    "material_id": m,
                    "location_id": l,
                    "start_time": s,
                    "end_time": e,
                    "request_time": r,
                    "commit_time": c,
                    "quantity": q, #round(q, 2),
                    "type": t
                }

                #if violates_constraints(plan, work_order):
                #    continue
                #if not is_valid_successor(plan, work_order):
                #    continue

                new_val = {
                    'demand': d, 'type': t, 'location': l, 'material': m, 'time': s,
                    'quantity': q, 'id': next_id,
                    'start_time': s, 'end_time': e,
                    'request_time': r, 'commit_time': c
                }

                new_tgt = expand_beam(tgt_tokens, new_val)
                new_plan = plan + [work_order]
                new_beams.append((score + log_prob, new_tgt, new_plan, next_id + 1))

        if not new_beams:
            break
        beams = sorted(new_beams, key=lambda x: -x[0])[:beam_width]
    
    if not beams:
        return []
    
    best_score, _, best_plan, _ = beams[0]
    return best_plan


# (New) --- Static Token Generator ---
def generate_static_tokens():
    material_df = pd.read_csv("data/material.csv")
    location_df = pd.read_csv("data/location.csv")
    method_df = pd.read_csv("data/method.csv")
    bom_df = pd.read_csv("data/bom.csv")

    tokens = []

    for _, row in material_df.iterrows():
        tokens.append({
            'type': get_token_type('material'),
            'material': int(row['id']),
            #'is_leaf': bool(row.get('is_leaf', False))
        })

    for _, row in location_df.iterrows():
        tokens.append({
            'type': get_token_type('location'),
            'location': int(row['id'])
        })

    for _, row in method_df.iterrows():
        tokens.append({
            'type': int(row['type']),
            'method': int(row['id']),
            'material': int(row['material_id']),
            'location': int(row['location_id']),
            #'source_location': int(row['source_location_id']),
            'source_location': int(row['source_location_id']) if pd.notna(row['source_location_id']) else 0

            #'method_type': int(row['method_type'])
        })

    for _, row in bom_df.iterrows():
        tokens.append({
            'type': get_token_type('bom'),
            'parent': int(row['parent']),
            'child': int(row['child']),
            #'quantity': float(row['quantity'])
        })

    return tokens


# --- Updated Combined Input Token Generator ---
def generate_encoder_input(input_dict):
    static_tokens = generate_static_tokens()
    dynamic_tokens = generate_candidate_tokens(input_dict)
    encoded_static = encode_tokens(static_tokens, token_type_id=0)
    encoded_dynamic = encode_tokens(dynamic_tokens, token_type_id=1)

    combined = {}
    all_keys = set(encoded_static.keys()) | set(encoded_dynamic.keys())
    for k in all_keys:
        static_val = encoded_static.get(k)
        dynamic_val = encoded_dynamic.get(k)
        if static_val is None and dynamic_val is not None:
            combined[k] = torch.cat([
                torch.zeros_like(dynamic_val), dynamic_val
            ], dim=0)
        elif dynamic_val is None and static_val is not None:
            combined[k] = torch.cat([
                static_val, torch.zeros_like(static_val)
            ], dim=0)
        elif static_val is not None and dynamic_val is not None:
            combined[k] = torch.cat([static_val, dynamic_val], dim=0)
    return combined


# Patch --- Update predict_plan to use generate_encoder_input()
def predict_plan(model, input_example, threshold=0.5):
    logger.info(f"📦 Input to generate_candidate_tokens: {input_example["input"]}")
    src_tokens = generate_encoder_input(input_example["input"])

    for k in src_tokens:
        if src_tokens[k].dim() == 1:
            src_tokens[k] = src_tokens[k].unsqueeze(0)
    
    #logger.info(f"src_tokens['quantity']: {src_tokens['quantity'][0].tolist()}")

    bom = load_bom("data/bom.csv")
    plan = decode_predictions(
        model, src_tokens,
        beam_width=5,
        sampling="topk",
        topk=5,
        #inventory=inventory,  # define this if you want to enforce availability
        bom=bom,
        time_window=(0, 100)
    )
    #plan = decode_predictions(model, src_tokens, threshold=threshold)
    return plan


# --- Visualize Demand vs Plan ---
def summarize_by_key(steps):
    from collections import defaultdict
    summary = defaultdict(float)
    for s in steps:
        k = (s['material_id'], s['location_id'], s['start_time'])
        summary[k] += s['quantity']
    return summary

def print_plan_vs_demand(predicted, ground_truth):
    pred_summary = summarize_by_key(predicted)
    gt_summary = summarize_by_key(ground_truth)
    all_keys = set(pred_summary) | set(gt_summary)

    logger.info("\nPlan vs Demand Comparison:")
    for k in sorted(all_keys):
        p = pred_summary.get(k, 0)
        g = gt_summary.get(k, 0)
        logger.info(f"  {k}: predicted={p:.2f}, ground_truth={g:.2f}, diff={abs(p-g):.2f}")


# --- Evaluate Plan ---
def evaluate_plan(predicted: List[Dict], ground_truth: List[Dict]) -> None:
    def to_dict(plan):
        return {(p['material_id'], p['location_id'], p['start_time']): p['quantity'] for p in plan}
    
    pred_dict = to_dict(predicted)
    gt_dict = to_dict(ground_truth)
    keys = set(pred_dict) | set(gt_dict)

    total_diff = 0.0
    for key in keys:
        diff = abs(pred_dict.get(key, 0) - gt_dict.get(key, 0))
        total_diff += diff
        logger.info(f"Diff at {key}: {diff:.2f}")

    logger.info(f"\nTotal quantity diff: {total_diff:.2f}")


# --- CLI Entrypoint ---
def main():
    update_config_from_static_data(config)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train_stepwise", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("-d", "--depth", help='Starting from 0')

    args = parser.parse_args()

    if args.train_stepwise:
        train_stepwise()
    elif args.train:
        train(args)
    elif args.predict:
        _, model = restore_model()
        input_example = {
            "input": {
                "demand": [
                    {"type": 0, "demand_id": 0, "location_id": 1, "material_id": 89, "request_time": 8, "commit_time": 8, "start_time": 8, "end_time": 8, "quantity": 339},
                ]
            },
            "tgt": []
        }

        aps_example = [
            {"type": 2, "demand_id": 0, "location_id": 1, "material_id": 89, "request_time": 8, "commit_time": 8, "start_time": 5, "end_time": 8, "quantity": 339},
        ]

        plan = predict_plan(model, input_example)
        logger.info("\nPredicted Plan:")
        for step in plan:
            logger.info(step)

        logger.info("\nEvaluation vs APS:")
        evaluate_plan(plan, aps_example)

        print_plan_vs_demand(plan, aps_example)

if __name__ == "__main__":
    main()

