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

def is_demand(tgt_tokens):
    # [B, T]
    return tgt_tokens['type'] == get_token_type('demand')

def is_workorder(tgt_tokens):
    # [B, T]
    make = tgt_tokens['type'] == get_token_type('make')
    purchase = tgt_tokens['type'] == get_token_type('purchase')
    move = tgt_tokens['type'] == get_token_type('move')

    return make | purchase | move

def is_target_eod(tgt_tokens):
    TYPE_EOD = get_token_type('eod')
    return tgt_tokens['type'] == TYPE_EOD           # [B, T]

def is_not_target_eod(tgt_tokens):
    TYPE_EOD = get_token_type('eod')
    # [B, T]
    demand = tgt_tokens['type'] == get_token_type('demand')
    make = tgt_tokens['type'] == get_token_type('make')
    purchase = tgt_tokens['type'] == get_token_type('purchase')
    move = tgt_tokens['type'] == get_token_type('move')

    return demand | make | purchase | move

def is_target_successor(tgt_tokens):
    # successor mask:
    seq = tgt_tokens['seq_in_demand'].unsqueeze(2)        # [B, T, 1]
    succ = tgt_tokens['successor'].unsqueeze(1)           # [B, 1, T]

    # Mask[i, j] = True if j is the successor of i
    return (succ == seq)                        # [B, T, T]

def compute_next_sequence_mask(tgt_tokens):
    current = tgt_tokens['seq_in_demand'].unsqueeze(2)     # [B, T, 1]
    next = tgt_tokens['seq_in_demand'].unsqueeze(1)     # [B, 1, T]

    return (current + 1 == next)                        # [B, T, T]

def compute_purchase_eod_mask(tgt_tokens):
    purchase = tgt_tokens['type'] == get_token_type('purchase')     # [B, T]
    eod = tgt_tokens['type'] == get_token_type('eod')     # [B, T]

    purchase = purchase.unsqueeze(2)        # [B, T, 1]
    eod = eod.unsqueeze(1)           # [B, 1, T]

    return (purchase & eod)                        # [B, T, T]

def compute_eod_purchase_mask(tgt_tokens):
    purchase = tgt_tokens['type'] == get_token_type('purchase')     # [B, T]
    eod = tgt_tokens['type'] == get_token_type('eod')     # [B, T]

    purchase = purchase.unsqueeze(1)        # [B, T, 1]
    eod = eod.unsqueeze(2)           # [B, 1, T]

    return (eod & purchase)                        # [B, T, T]

LARGE_POSITIVE_BIAS = 100000
def compute_last_purchase_eod_bias(tgt_tokens):
    # Assume:
    # - tgt_tokens['type']: [B, T] where 0=purchase, 1=eod, etc.
    # - tgt_tokens['seq_in_demand']: [B, T]
    # - tgt_tokens['total_in_demand']: [B, T]

    is_eod = tgt_tokens['type'] == get_token_type('eod')
    is_purchase = tgt_tokens['type'] == get_token_type('purchase')
    seq = tgt_tokens['seq_in_demand']
    total = tgt_tokens['total_in_demand']

    # [B, T] → [B, T, 1] and [B, 1, T]
    eod_mask = is_eod & (seq == total - 1)      # Only eod tokens
    purchase_mask = is_purchase & (seq == total - 2)  # Last purchase


    # [B, T, T]: eod attends to final purchase in same demand group
    attention_bias = (
        eod_mask.unsqueeze(2) & purchase_mask.unsqueeze(1)
    )  # bias where i (query) is eod, j (key) is final purchase
    '''
    # [B, T, T]: eod attends to final purchase in same demand group
    attention_bias = (
        purchase_mask.unsqueeze(2) & eod_mask.unsqueeze(1)
    )  # bias where j (key) is eod, i (query) is final purchase
    '''

    # Scale and add to attention scores
    attn_scores = attention_bias.float() * LARGE_POSITIVE_BIAS
    return attn_scores

def compute_demand_eod_bias(tgt_tokens):
    # Assume:
    # - tgt_tokens['type']: [B, T] where 0=purchase, 1=eod, etc.
    # - tgt_tokens['seq_in_demand']: [B, T]
    # - tgt_tokens['total_in_demand']: [B, T]

    is_eod = tgt_tokens['type'] == get_token_type('eod')
    seq = tgt_tokens['seq_in_demand']
    total = tgt_tokens['total_in_demand']
    is_demand = tgt_tokens['type'] == get_token_type('demand')

    # [B, T] → [B, T, 1] and [B, 1, T]
    eod_mask = is_eod & (seq == total - 1)      # Only eod tokens
    demand_mask = is_demand & (seq == 0) # Last purchase
    '''
    # [B, T, T]: eod attends to final purchase in same demand group
    attention_bias = (
        eod_mask.unsqueeze(2) & demand_mask.unsqueeze(1)
    )  # bias where i (query) is eod, j (key) is final purchase
    '''
    # [B, T, T]: eod attends to final purchase in same demand group
    attention_bias = (
        demand_mask.unsqueeze(2) & eod_mask.unsqueeze(1)
    )  # bias where j (key) is eod, i (query) is final purchase


    # Scale and add to attention scores
    attn_scores = attention_bias.float() * LARGE_POSITIVE_BIAS
    return attn_scores

def compute_demand_eod_mask(tgt_tokens):
    # Assume:
    # - tgt_tokens['type']: [B, T] where 0=purchase, 1=eod, etc.
    # - tgt_tokens['seq_in_demand']: [B, T]
    # - tgt_tokens['total_in_demand']: [B, T]

    is_eod = tgt_tokens['type'] == get_token_type('eod')
    seq = tgt_tokens['seq_in_demand']
    total = tgt_tokens['total_in_demand']
    is_demand = tgt_tokens['type'] == get_token_type('demand')

    # [B, T] → [B, T, 1] and [B, 1, T]
    eod_mask = is_eod & (seq == total - 1)      # Only eod tokens
    demand_mask = is_demand & (seq == 0) # Last purchase
    '''
    # [B, T, T]: eod attends to final purchase in same demand group
    attention_bias = (
        eod_mask.unsqueeze(2) & demand_mask.unsqueeze(1)
    )  # bias where i (query) is eod, j (key) is final purchase
    '''
    # [B, T, T]: eod attends to final purchase in same demand group
    return (
        demand_mask.unsqueeze(2) & eod_mask.unsqueeze(1)
    )  # bias where j (key) is eod, i (query) is final purchase


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

def is_target_successor(tgt_tokens): # this one works "locally"
    # successor mask:
    seq = tgt_tokens['seq_in_demand'].unsqueeze(2)        # [B, T, 1]
    succ = tgt_tokens['successor'].unsqueeze(1)           # [B, 1, T]

    # Mask[i, j] = True if j is the successor of i
    succ_mask = succ == seq
    
    #print("Successor pairs (i, j):", torch.nonzero(succ_mask[0]))
    return succ_mask                        # [B, T, T]

def build_temporal_mask(tgt_tokens): # this one works "locally"
    start_time = tgt_tokens['start_time']     # [B, T]
    end_time = tgt_tokens['end_time']     # [B, T]

    # [B, T, T]
    good_ordering = (is_target_successor(tgt_tokens) & end_time.unsqueeze(2) <= start_time.unsqueeze(1))  
    #good_ordering = is_target_successor(tgt_tokens) & start_time.unsqueeze(2) >= end_time.unsqueeze(1)

    '''
    # [B, T]
    token_paced = (
            (tgt_tokens['start_time'] + tgt_tokens['lead_time']) == tgt_tokens['end_time']
        ) & (
            tgt_tokens['seq_in_demand'] > tgt_tokens['successor']
        ) & (
            (is_target_eod(tgt_tokens) & ((tgt_tokens['seq_in_demand'] + 1) == tgt_tokens['total_in_demand']))
            | is_not_target_eod(tgt_tokens)
        )        

    good_internals = token_paced.unsqueeze(2) & token_paced.unsqueeze(1)  # [B, T, T]

    temporal_check = good_ordering & good_internals
    '''
    return good_ordering

def build_pace_mask(tgt_tokens): # this one works "locally"
    start_time = tgt_tokens['start_time']     # [B, T]
    end_time = tgt_tokens['end_time']     # [B, T]

    # [B, T]
    token_paced = (
            (tgt_tokens['start_time'] + tgt_tokens['lead_time']) == tgt_tokens['end_time']
        ) & (
            (is_target_eod(tgt_tokens) & ((tgt_tokens['seq_in_demand'] + 1) == tgt_tokens['total_in_demand']))
            | (is_demand(tgt_tokens) & (tgt_tokens['seq_in_demand'] == 0))
            | is_workorder(tgt_tokens)
        ) 
    '''
    & (
            tgt_tokens['seq_in_demand'] > tgt_tokens['successor']
        )      
    '''
    good_internals = token_paced.unsqueeze(2) & token_paced.unsqueeze(1)  # [B, T, T]

    return good_internals


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

    ################ Compare demand_ids: [B, T, 1] <= [B, 1, T] -> [B, T, T]
    demand_mask = demand_ids.unsqueeze(2) >= demand_ids.unsqueeze(1)  # [B, T, T]
    #demand_mask = demand_ids.unsqueeze(2) <= demand_ids.unsqueeze(1)  # [B, T, T]

    ################ Combine masks: broadcast causal_mask over batch
    attn_mask = (demand_mask & causal_mask.unsqueeze(0))  # [B, T, T]
    return attn_mask

def build_sequence_mask(tgt_tokens: dict) -> torch.Tensor:
    ################ Builds a self-attention mask such that each token can only attend to tokens with
    # the same demand ID and an equal or earlier seq_in_demand value.

    seq = tgt_tokens['seq_in_demand']         # [B, T]
    seq_i = seq.unsqueeze(2)                  # [B, T, 1]
    seq_j = seq.unsqueeze(1)                  # [B, 1, T]

    # Only allow attention if demand matches and j.seq ≤ i.seq
    seq_mask = (seq_j >= seq_i)  # [B, T, T]
    #seq_mask = (demand_i == demand_j) & (seq_j >= seq_i)  # [B, T, T]

    return seq_mask  # [B, T, T]


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
    purchase_eod = compute_purchase_eod_mask(tgt_tokens)
    demand_eod = compute_demand_eod_mask(tgt_tokens)

    cross_attention = demand_to_method | method_to_workorder | move_to_method | method_to_workorder_ms | make_to_bom | bom_to_make
    self_attention = demand_to_workorder | move_to_workorder | make_to_make | demand_eod

    # additional filters
    demand_ids = tgt_tokens['demand']     # [B, T]
    same_demand = demand_ids.unsqueeze(2) == demand_ids.unsqueeze(1)  # [B, T, T]
    quantity = tgt_tokens['quantity']     # [B, T]
    same_quantity = quantity.unsqueeze(2) == quantity.unsqueeze(1)  # [B, T, T]
    total_in_demand = tgt_tokens['total_in_demand']     # [B, T]
    same_total_in_demand = total_in_demand.unsqueeze(2) == total_in_demand.unsqueeze(1)  # [B, T, T]
    next_sequence = compute_next_sequence_mask(tgt_tokens)

    temporal_check = build_temporal_mask(tgt_tokens)
    #inter_demand_check = build_demand_mask(tgt_tokens)
    sequence_check = build_sequence_mask(tgt_tokens)

    pace_check = build_pace_mask(tgt_tokens)

    #self_attention = self_attention & same_demand & same_quantity & sequence_check & pace_check & temporal_check 
    self_attention_target = same_demand & same_quantity & same_total_in_demand & sequence_check & pace_check & temporal_check
    #print(f"self_attention_target true count: {(self_attention_target).bool().sum().item()}")
    #print(f"self_attention true count: {(self_attention & is_target_successor(tgt_tokens)).bool().sum().item()}")
    self_attention_scores = (self_attention & is_target_successor(tgt_tokens) & same_demand).float() * LARGE_POSITIVE_BIAS
    #print("self_attention & is_target_successor(tgt_tokens) & same_demand pairs (i, j):", torch.nonzero((self_attention & is_target_successor(tgt_tokens) & same_demand)[0]))
    #print("compute_demand_eod_mask(tgt_tokens) pairs (i, j):", torch.nonzero(compute_demand_eod_mask(tgt_tokens)[0]))
 

    #self_attention = torch.ones_like(self_attention)
    #cross_attention = torch.ones_like(cross_attention)

    if config['use_attention'] == 0:
        self_attention_target = torch.ones_like(self_attention_target)

    cross_attention_mask = torch.where(
        cross_attention,
        torch.tensor(0.0, device=cross_attention.device),
        torch.tensor(float('-inf'), device=cross_attention.device)
    )

    self_attention_mask = torch.where(
        self_attention_target,
        torch.tensor(0.0, device=self_attention.device),
        torch.tensor(float('-inf'), device=self_attention.device)
    )

    return cross_attention_mask, compute_last_purchase_eod_bias(tgt_tokens) + self_attention_scores + compute_demand_eod_bias(tgt_tokens) #+ self_attention_mask
 