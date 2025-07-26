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


def apply_bom_mask(logits, src_tokens, tgt_tokens):
    #child_map = defaultdict(set)
    #for parent, child in bom_edges:
    #    child_map[parent].add(child)
    #child_map = load_bom()
    child_map = extract_bom_from_tokens(src_tokens)

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


def apply_field_constraints(logits_dict, src_tokens, prev_tokens):
    B, T = prev_tokens['type'].shape[:2]
    workorder = {get_token_type('make'), get_token_type('purchase'), get_token_type('move')}
    bom = extract_bom_parent_from_tokens(src_tokens) # load_bom_parent()
    method_maps = extract_method_from_tokens(src_tokens)
    for b in range(B):
        for t in range(T):
            token = {k: prev_tokens[k][b][t].item() for k in prev_tokens}
            op_type = token['type']

            if op_type not in workorder:
                continue

            material = token["material"]
            demand = token["demand"]
            location = token["location"]

            method_key = (material, location, op_type)
            #lead_time = get_method_lead_time(method_key)
            #method = get_method(method_key)
            method = method_maps.get(method_key, [])

            #if lead_time is None:
            if not method:
                # Invalid method: mask entire token
                for field in logits_dict:
                    if field != 'eod':
                        logits_dict[field][b, t, :] = float('-inf')
                continue

            lead_time = method[0]['lead_time']
            type = method[0]['type']

            parents = bom.get(material, [])
            parent_start_times = []
            for t_prev in range(t):
                prev = {k: prev_tokens[k][b][t_prev].item() for k in prev_tokens}
                if prev["demand"] == demand and (
                    (prev["type"] == get_token_type('make') and prev["material"] in parents) or 
                    (prev["type"] in {get_token_type('demand'), get_token_type('move')} and prev["material"] == material)) :

                    parent_start_times.append(prev["start_time"])

                    # material inherited
                    logits_dict['material'][b, t, :] = float('-inf')
                    logits_dict['material'][b, t, prev['material']] = 0.0

                    # quantity inherited
                    logits_dict['quantity'][b, t, :] = float('-inf')
                    logits_dict['quantity'][b, t, prev['quantity']] = 0.0

                    # demand inherited
                    logits_dict['demand'][b, t, :] = float('-inf')
                    logits_dict['demand'][b, t, prev['demand']] = 0.0

                    # lead_time
                    logits_dict['lead_time'][b, t, :] = float('-inf')
                    logits_dict['lead_time'][b, t, int(lead_time)] = 0.0

                    # type
                    logits_dict['type'][b, t, :] = float('-inf')
                    logits_dict['type'][b, t, int(type)] = 0.0

                    # location
                    if prev['type'] == get_token_type('move'):  # move
                        logits_dict['location'][b, t, :] = float('-inf')
                        logits_dict['location'][b, t, prev['source_location']] = 0.0
                    else:
                        logits_dict['location'][b, t, :] = float('-inf')
                        logits_dict['location'][b, t, prev['location']] = 0.0

            if not parent_start_times:
                for field in logits_dict:
                    logits_dict[field][b, t, :] = float('-inf')                    
                continue  # or mask if this should never happen

            max_end_time = max(min(parent_start_times), 0) #torch.clamp(min(parent_start_times), min=0)
            max_start_time = max(max_end_time - lead_time, 0) #torch.clamp(max_end_time - lead_time, min=0)

            L_end = logits_dict["end_time"].shape[-1]
            L_start = logits_dict["start_time"].shape[-1]

            mask_end = torch.arange(L_end, device=logits_dict["end_time"].device) > max_end_time
            mask_start = torch.arange(L_start, device=logits_dict["start_time"].device) > max_start_time

            logits_dict["end_time"][b, t][mask_end] = float('-inf')
            logits_dict["start_time"][b, t][mask_start] = float('-inf')

    return logits_dict


def apply_demand_constraints(logits_dict, src_tokens, prev_tokens):
    B, T = prev_tokens['type'].shape[:2]
    demand_type = get_token_type('demand')
    eod_type = get_token_type('eod')  # End-of-demand token

    for b in range(B):
        # Extract ordered demand_ids from src
        src_demands = [src_tokens['demand'][b][i].item()
                       for i in range(src_tokens['type'].shape[1])
                       if src_tokens['type'][b][i].item() == demand_type]
        if not src_demands:
            continue

        # Get demand_ids already generated
        prev_demand_ids = [
            prev_tokens['demand'][b][t].item()
            for t in range(T)
            if prev_tokens['type'][b][t].item() == demand_type
        ]

        unmet_demand_id = next(
            (d for d in src_demands if d not in prev_demand_ids),
            None
        )

        for t in range(T):
            if unmet_demand_id is None:
                # No more demands should be predicted
                logits_dict['type'][b, t, demand_type] = float('-inf')
                continue

            is_first = (t == 0)
            last_token_type = prev_tokens['type'][b][t - 1].item() if t > 0 else None
            is_after_eod = (last_token_type == eod_type)

            if is_first or is_after_eod:
                # Allow next demand only at the start or after EOD
                logits_dict['type'][b, t, :] = float('-inf')
                logits_dict['type'][b, t, demand_type] = 0
                logits_dict['demand'][b, t, :] = float('-inf')
                logits_dict['demand'][b, t, unmet_demand_id] = 0
            else:
                # Forbid demand mid-plan
                logits_dict['type'][b, t, demand_type] = float('-inf')

    return logits_dict


def extract_method_from_tokens(src_tokens):
    B, L = src_tokens['type'].shape
    method_type = get_token_type('method')
    method_maps = defaultdict(list)

    for b in range(B):
        for i in range(L):
            if src_tokens['type'][b, i].item() == method_type:
                mat = src_tokens['material'][b, i].item()
                loc = src_tokens['location'][b, i].item()
                op = src_tokens['op_type'][b, i].item()  # or 'type' if reused
                lead_time = src_tokens['lead_time'][b, i].item()
                method_maps[(mat, loc, op)].append({
                    'lead_time': lead_time,
                    'type': op
                })

    return method_maps

def extract_bom_from_tokens(src_tokens):
    B, L = src_tokens['type'].shape
    child_maps = defaultdict(list)
    bom_type = get_token_type('bom')

    for b in range(B):
        for i in range(L):
            if src_tokens['type'][b, i].item() == bom_type:
                parent = src_tokens['parent'][b, i].item()
                child = src_tokens['child'][b, i].item()
                child_maps[parent].append(child)
    return child_maps

def extract_bom_parent_from_tokens(src_tokens):
    B, L = src_tokens['type'].shape
    parent_maps = defaultdict(list)
    bom_type = get_token_type('bom')
    
    for b in range(B):
        for i in range(L):
            if src_tokens['type'][b, i].item() == bom_type:
                parent = src_tokens['parent'][b, i].item()
                child = src_tokens['child'][b, i].item()
                parent_maps[child].append(parent)
    return parent_maps

def _apply_field_constraints(self, logits_dict, prev_tokens): # this one works
    B, T = prev_tokens['type'].shape[:2]

    #batch_size = prev_tokens['type'].size(0)
    
    workorder = {get_token_type('make'), get_token_type('purchase'), get_token_type('move')}
    bom = load_bom_parent()

    #for b in range(batch_size):
    for b in range(B):
        for t in range(T):
            
            token = {k: prev_tokens[k][b][t].item() for k in prev_tokens}

            if token['type'] not in workorder:
                continue

            material = token["material"]
            demand = token["demand"]
            type = token["type"]
            location = token["location"]

            method_key = (
                material,
                location,
                type,
            )
            lead_time = get_method_lead_time(method_key)
            if lead_time is None:
                for field in logits_dict:
                    logits_dict[field][b, t, :] = float('-inf')
                continue
            
            parents = bom.get(material, [])

            parent_start_times = []
            for t_prev in range(t):
                prev = {k: prev_tokens[k][b][t_prev].item() for k in prev_tokens}
                if (prev["material"] in parents or prev["type"] == get_token_type('demand')) and prev["demand"] == demand:
                    parent_start_times.append(prev["start_time"])

            if not parent_start_times:
                continue

            max_allowed_for_end_time = max(min(parent_start_times), 0)
            L_for_end_time = logits_dict["end_time"].shape[-1]
            mask_for_end_time = torch.arange(L_for_end_time, device=logits_dict["end_time"].device) > max_allowed_for_end_time
            logits_dict["end_time"][b, t][mask_for_end_time] = float('-inf')

            max_allowed_for_start_time = max(max_allowed_for_end_time - lead_time, 0)
            L_for_start_time = logits_dict["start_time"].shape[-1]
            mask_for_start_time = torch.arange(L_for_start_time, device=logits_dict["start_time"].device) > max_allowed_for_start_time
            logits_dict["start_time"][b, t][mask_for_start_time] = float('-inf')

    return logits_dict

def _apply_field_constraints(self, logits_dict, prev_tokens): # this one works
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
