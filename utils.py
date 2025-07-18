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

from config import config, logger

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
