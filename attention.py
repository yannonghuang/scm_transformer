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
from utils import load_bom, load_bom_parent, get_method, get_method_lead_time


def compute_simple_bom_mask(src_tokens, tgt_tokens, from_make_to_bom=True):
    B, S = src_tokens['type'].shape
    _, T = tgt_tokens['type'].shape

    TYPE_BOM = get_token_type('bom')
    TYPE_MAKE = get_token_type('make')

    # Mask only applies when tgt is 'make' and src is 'bom'
    is_make = tgt_tokens['type'] == TYPE_MAKE     # [B, T]
    is_bom = src_tokens['type'] == TYPE_BOM       # [B, S]

    # Material in make should match parent in BOM
    tgt_material = tgt_tokens['material']         # [B, T]
    if from_make_to_bom:
        src_bom = src_tokens['parent']             # [B, S]
    else:
        src_bom = src_tokens['child']             # [B, S]

    # Broadcast and compare [B, T, 1] vs [B, 1, S] -> [B, T, S]
    material_match = tgt_material.unsqueeze(2) == src_bom.unsqueeze(1)  # [B, T, S]
    type_equal = is_make.unsqueeze(2) & is_bom.unsqueeze(1)                # [B, T, S]
    if from_make_to_bom:
        type_match = type_equal
    else:
        type_match = torch.ones_like(is_make).unsqueeze(2) & is_bom.unsqueeze(1) 

    #mask = type_match & material_match
    # Final boolean mask: True = allowed, False = disallowed
    mask_bool = type_match & material_match        # [B, T, S]

    # Convert to float mask for additive attention
    # True → 0.0, False → -inf
    mask = torch.where(mask_bool, torch.tensor(0.0, device=mask_bool.device),
                                  torch.tensor(float('-inf'), device=mask_bool.device))

    return mask_bool

def compute_bom_mask(src_tokens, tgt_tokens):
    M = compute_simple_bom_mask(src_tokens, tgt_tokens, from_make_to_bom=True)
    B = compute_simple_bom_mask(src_tokens, tgt_tokens, from_make_to_bom=False)

    # Transpose B so it becomes [B, S, T]
    B_transposed = B.transpose(1, 2)

    # Perform batched matrix multiplication (bool): [B, T, S] @ [B, S, T] -> [B, T, T]
    # Each [i,j] position will be True if there's any `k` such that M[b,i,k] & B[b,j,k]
    T_primitive = torch.matmul(M.float(), B_transposed.float()) > 0  # boolean result

    # additional filters
    locations  = tgt_tokens['location']   # [B, T]
    same_location = locations.unsqueeze(2) == locations.unsqueeze(1)  # [B, T, T]

    T = T_primitive & same_location # [B, T, T]

    #return make_to_bom_mask, bom_to_make_mask, make_to_make_mask
    return M, B, T

def compute_simple_method_mask(src_tokens, tgt_tokens):
    B, S = src_tokens['type'].shape
    _, T = tgt_tokens['type'].shape

    TYPE_DEMAND = get_token_type('demand')
    TYPE_MAKE = get_token_type('make')
    TYPE_MOVE = get_token_type('move')
    TYPE_PURCHASE = get_token_type('purchase')

    is_demand = tgt_tokens['type'] == TYPE_DEMAND     # [B, T]
    is_make = tgt_tokens['type'] == TYPE_MAKE           # [B, T]
    is_purchase = tgt_tokens['type'] == TYPE_PURCHASE   # [B, T]
    is_move = tgt_tokens['type'] == TYPE_MOVE           # [B, T]
    t_workorder = (is_make | is_purchase | is_move)       # [B, S]

    s_make = src_tokens['type'] == TYPE_MAKE           # [B, S]
    s_purchase = src_tokens['type'] == TYPE_PURCHASE   # [B, S]
    s_move = src_tokens['type'] == TYPE_MOVE           # [B, S]
    
    s_method = (s_make | s_purchase | s_move)       # [B, S]
    tgt_ones = torch.ones_like(is_demand)

    ############# Type match
    tgt_type = tgt_tokens['type']         # [B, T]
    src_type = src_tokens['type']         # [B, S]
    # Broadcast and compare [B, T, 1] vs [B, 1, S] -> [B, T, S]
    type_match = tgt_type.unsqueeze(2) == src_type.unsqueeze(1)  # [B, T, S]
    ############# Material match
    tgt_material = tgt_tokens['material']         # [B, T]
    src_material = src_tokens['material']         # [B, S]
    # Broadcast and compare [B, T, 1] vs [B, 1, S] -> [B, T, S]
    material_match = tgt_material.unsqueeze(2) == src_material.unsqueeze(1)  # [B, T, S]

    ############# Location match
    tgt_location = tgt_tokens['location']         # [B, T]
    src_location_ms = src_tokens['source_location']         # [B, S]
    src_location = src_tokens['location']         # [B, S]
    # Broadcast and compare [B, T, 1] vs [B, 1, S] -> [B, T, S]
    location_match = tgt_location.unsqueeze(2) == src_location.unsqueeze(1)  # [B, T, S]
    location_match_ms = tgt_location.unsqueeze(2) == src_location_ms.unsqueeze(1)  # [B, T, S]
    
    ############# demand match
    tgt_demand = tgt_tokens['demand']         # [B, T]
    src_demand = src_tokens['demand']         # [B, S]
    # Broadcast and compare [B, T, 1] vs [B, 1, S] -> [B, T, S]
    demand_match = tgt_demand.unsqueeze(2) == src_demand.unsqueeze(1)  # [B, T, S]

    ############# lead_time match
    tgt_lead_time = tgt_tokens['lead_time']    # [B, T]
    src_lead_time = src_tokens['lead_time']                 # [B, S]
    # Broadcast for comparison
    lead_time_equal = tgt_lead_time.unsqueeze(2) == src_lead_time.unsqueeze(1)  # [B, T, S]
    # Broadcast is_demand to [B, T, S]
    is_demand_expanded = is_demand.unsqueeze(2).expand(-1, -1, src_lead_time.size(1))  # [B, T, S]
    # If is_demand, override with True; otherwise, enforce lead_time match
    lead_time_match = torch.where(is_demand_expanded, torch.ones_like(lead_time_equal), lead_time_equal)

    ############################################################################################################
    type_match_demand_to_method = (is_demand).unsqueeze(2) & s_method.unsqueeze(1)
    mask_bool_demand_to_method = type_match_demand_to_method & material_match & location_match     # [B, T, S]
    ##################
    type_match_method_to_workorder = (t_workorder).unsqueeze(2) & s_method.unsqueeze(1)
    mask_bool_method_to_workorder = type_match_method_to_workorder & type_match & material_match & location_match & lead_time_match     # [B, T, S]
    ##################
    type_match_move_to_method = (is_move).unsqueeze(2) & s_move.unsqueeze(1)
    mask_bool_move_to_method = type_match_move_to_method & material_match & location_match     # [B, T, S]
    #########
    type_match_method_to_workorder_ms = (t_workorder).unsqueeze(2) & s_move.unsqueeze(1)                # [B, T, S]
    mask_bool_method_to_workorder_ms = type_match_method_to_workorder_ms & material_match & location_match_ms     # [B, T, S]
    ##################    
    
    return mask_bool_demand_to_method, mask_bool_method_to_workorder, mask_bool_method_to_workorder_ms, mask_bool_move_to_method

def build_temporal_mask(tgt_tokens):
    start_time = tgt_tokens['start_time']     # [B, T]
    end_time = tgt_tokens['end_time']     # [B, T]
    good_ordering = start_time.unsqueeze(2) > end_time.unsqueeze(1)  # [B, T, T]
    workorder_paced = (tgt_tokens['start_time'] + tgt_tokens['lead_time']) == tgt_tokens['end_time']  # [B, T]    
    good_internals = workorder_paced.unsqueeze(2) & workorder_paced.unsqueeze(1)  # [B, T, T]
    temporal_check = good_ordering & good_internals
    return temporal_check


def build_demand_mask(tgt_tokens: dict) -> torch.Tensor:
    """
    Builds a decoder self-attention mask that enforces:
    - causal (left-to-right) masking
    - monotonicity by demand group (tokens from demand i cannot attend to those from demand j > i)

    Args:
        tgt_tokens: dict with key "demand" -> tensor of shape [B, T] (batch of demand ids)

    Returns:
        attn_mask: bool tensor of shape [B, T, T]
    """
    demand_ids = tgt_tokens['demand']  # [B, T]
    B, T = demand_ids.shape

    # Causal mask: [T, T]
    causal_mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=demand_ids.device))  # [T, T]

    # Compare demand_ids: [B, T, 1] <= [B, 1, T] -> [B, T, T]
    demand_mask = demand_ids.unsqueeze(2) >= demand_ids.unsqueeze(1)  # [B, T, T]
    #demand_mask = demand_ids.unsqueeze(2) <= demand_ids.unsqueeze(1)  # [B, T, T]

    # Combine masks: broadcast causal_mask over batch
    attn_mask = demand_mask & causal_mask.unsqueeze(0)  # [B, T, T]
    return attn_mask

def compute_method_mask(src_tokens, tgt_tokens):
    demand_to_method, method_to_workorder, method_to_workorder_ms, move_to_method = compute_simple_method_mask(src_tokens, tgt_tokens)
     # Transpose B so it becomes [B, S, T]
    method_to_workorder_transposed = method_to_workorder.transpose(1, 2)
    method_to_workorder_ms_transposed = method_to_workorder_ms.transpose(1, 2)
    
    # Perform batched matrix multiplication (bool): [B, T, S] @ [B, S, T] -> [B, T, T]
    # Each [i,j] position will be True if there's any `k` such that M[b,i,k] & B[b,j,k]
    demand_to_workorder = torch.matmul(demand_to_method.float(), method_to_workorder_transposed.float()) > 0  # boolean result [B, T, T]
    move_to_workorder = torch.matmul(move_to_method.float(), method_to_workorder_ms_transposed.float()) > 0  # boolean result [B, T, T]

    #return demand_to_method_mask, method_to_workorder_mask, workorder_to_workorder_mask
    return demand_to_method, method_to_workorder, demand_to_workorder, move_to_method, method_to_workorder_ms, move_to_workorder

def compute_attention_mask(src_tokens, tgt_tokens):
    make_to_bom, bom_to_make, make_to_make = compute_bom_mask(src_tokens, tgt_tokens)        
    demand_to_method, method_to_workorder, demand_to_workorder, move_to_method, method_to_workorder_ms, move_to_workorder = compute_method_mask(src_tokens, tgt_tokens)

    cross_attention = demand_to_method | method_to_workorder | move_to_method | method_to_workorder_ms | make_to_bom | bom_to_make
    self_attention = demand_to_workorder | move_to_workorder | make_to_make

    # additional filters
    demand_ids = tgt_tokens['demand']     # [B, T]
    same_demand = demand_ids.unsqueeze(2) == demand_ids.unsqueeze(1)  # [B, T, T]
    quantity = tgt_tokens['quantity']     # [B, T]
    same_quantity = quantity.unsqueeze(2) == quantity.unsqueeze(1)  # [B, T, T]

    temporal_check = build_temporal_mask(tgt_tokens)
    demand_check = build_demand_mask(tgt_tokens)

    self_attention = self_attention & same_demand & same_quantity & temporal_check & demand_check

    self_attention = ~self_attention

    cross_attention_mask = torch.where(
        cross_attention,
        torch.tensor(0.0, device=cross_attention.device),
        torch.tensor(float('-inf'), device=cross_attention.device)
    )

    self_attention_mask = torch.where(
        self_attention,
        torch.tensor(0.0, device=self_attention.device),
        torch.tensor(float('-inf'), device=self_attention.device)
    )

    return cross_attention_mask, self_attention_mask

def TODELETE_get_method_attention_bias(src_tokens, tgt_tokens):
    """
    Returns a bias tensor of shape (B, T_tgt, T_src) where invalid attentions are masked with -inf.
    src_tokens: dict of tensors for static method tokens, each of shape (B, T_src)
    tgt_tokens: dict of tensors for predicted tokens, each of shape (B, T_tgt)
    """
    B, T_tgt = tgt_tokens['material'].shape
    T_src = src_tokens['material'].shape[1]
    
    device = tgt_tokens['material'].device
    attn_bias = torch.zeros(B, T_tgt, T_src, device=device)

    for b in range(B):
        for t in range(T_tgt):
            tgt_material = tgt_tokens['material'][b, t].item()
            tgt_type = tgt_tokens['type'][b, t].item()
            tgt_location = tgt_tokens['location'][b, t].item()

            for s in range(T_src):
                src_material = src_tokens['material'][b, s].item()
                src_type = src_tokens['type'][b, s].item()
                src_location = src_tokens['location'][b, s].item()
                src_lead_time = src_tokens['lead_time'][b, s].item()

                if (
                    tgt_material != src_material or
                    tgt_type != src_type or
                    tgt_location != src_location
                ):
                    attn_bias[b, t, s] = float('-inf')
                    continue

                # Optional: compare lead_time (based on start_time, end_time if available)
                tgt_start = tgt_tokens['start_time'][b, t].item()
                tgt_end = tgt_tokens['end_time'][b, t].item()
                tgt_lead_time = tgt_start - tgt_end
                if tgt_lead_time != src_lead_time:
                    attn_bias[b, t, s] = float('-inf')

    return attn_bias

