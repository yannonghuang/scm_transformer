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

from config import config, get_token_type

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
            "lead_time": 0,

            "method": 0,
            "quantity": d["quantity"],

            "parent": 0,
            "child": 0,
        })
    return candidates


class SCMDataset(Dataset):
    def __init__(self, root_dir):
        self.sample_dirs = sorted(Path(root_dir).glob("*"))

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        #src = self.load_demand(sample_dir / "demands.csv")
        demand_dicts, tokens = self.load_demand(sample_dir / "demands.csv")
        src = generate_encoder_input(tokens, istensor=True)
        #src = generate_encoder_input({"demand": demand_dicts})

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
            #"time": torch.tensor(df["request_time"].values, dtype=torch.long),

            #"quantity": torch.tensor(df["quantity"].values, dtype=torch.float),
            "quantity": torch.tensor(df["quantity"].values, dtype=torch.long),

            "start_time": torch.zeros((len(df)), dtype=torch.long),
            "end_time": torch.zeros((len(df)), dtype=torch.long),
            "request_time": torch.tensor(df["request_time"].values, dtype=torch.long),
            "commit_time": torch.tensor(df["commit_time"].values, dtype=torch.long),
            "lead_time": torch.zeros((len(df)), dtype=torch.long),

            "parent": torch.zeros((len(df)), dtype=torch.long),
            "child": torch.zeros((len(df)), dtype=torch.long),

            "source_location": torch.tensor(df["location_id"].values, dtype=torch.long),
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
            #"time": torch.tensor(df["start_time"].values, dtype=torch.long), #.unsqueeze(0),

            "request_time": torch.tensor(df["request_time"].values, dtype=torch.long),
            "commit_time": torch.tensor(df["commit_time"].values, dtype=torch.long),

            "start_time": torch.tensor(df["start_time"].values, dtype=torch.long), #.unsqueeze(0),
            "end_time": torch.tensor(df["end_time"].values, dtype=torch.long), #.unsqueeze(0),
            "lead_time": torch.tensor(df["lead_time"].values, dtype=torch.long), #.unsqueeze(0),

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
            'source_location': int(row['source_location_id']) if pd.notna(row['source_location_id']) else 0,
            'lead_time': int(row['lead_time']),
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
        #'time': to_tensor("time"),

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

        'lead_time': to_tensor("lead_time"),
    }

# --- Updated Combined Input Token Generator ---
def generate_encoder_input(input_dict, istensor=False):
    static_tokens = generate_static_tokens()
    encoded_static = encode_tokens(static_tokens, token_type_id=0)

    if not istensor:
        dynamic_tokens = generate_candidate_tokens(input_dict)
        encoded_dynamic = encode_tokens(dynamic_tokens, token_type_id=1)
    else:
        encoded_dynamic = input_dict

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

