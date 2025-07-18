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

from data import generate_encoder_input
from config import logger
from utils import load_bom

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


