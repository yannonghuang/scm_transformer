import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
import random
import argparse
import os
import pandas as pd
from pathlib import Path


# --- Config ---
config = {
    'num_token_types': 9,
    'num_locations': 10,
    'num_time_steps': 70,
    'num_materials': 30,
    'num_methods': 30,
    'd_model': 128,
    'n_heads': 4,
    'd_ff': 256,
    'n_layers': 4,
    'dropout': 0.1,
    'batch_size': 8,
    'lr': 1e-4,
    'epochs': 10,
    'checkpoint_path': 'scm_transformer.pt',
    "max_train_samples": 1000
}

# --- Embedding Module (Updated) ---
class SCMEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.type_emb = nn.Embedding(config['num_token_types'], config['d_model'])
        self.loc_emb = nn.Embedding(config['num_locations'], config['d_model'])
        self.time_emb = nn.Embedding(config['num_time_steps'], config['d_model'])

        self.start_time_emb = nn.Embedding(config['num_time_steps'], config['d_model'])
        self.end_time_emb = nn.Embedding(config['num_time_steps'], config['d_model'])

        self.mat_emb = nn.Embedding(config['num_materials'], config['d_model'])
        self.method_emb = nn.Embedding(config['num_methods'], config['d_model'])

        self.token_type_emb = nn.Embedding(3, config['d_model'])  # 0=static, 1=demand, 2=plan

        self.quantity_proj = nn.Sequential(
            nn.Linear(1, config['d_model']),
            nn.ReLU(),
            nn.LayerNorm(config['d_model'])
        )
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, tokens):
        # Ensure shape: [batch_size, seq_len]
        #for k in ['type', 'location', 'material', 'time', 'method_id', 'token_type_id']:
        for k in ['type', 'location', 'material', 'time', 'start_time', 'end_time']:
            if tokens[k].dim() == 1:
                tokens[k] = tokens[k].unsqueeze(0)
            elif tokens[k].dim() == 3:
                tokens[k] = tokens[k].squeeze(1)

        if tokens['quantity'].dim() == 1:
            tokens['quantity'] = tokens['quantity'].unsqueeze(0)
        elif tokens['quantity'].dim() == 3:
            tokens['quantity'] = tokens['quantity'].squeeze(1)

        e_type = self.type_emb(tokens['type'])
        e_loc = self.loc_emb(tokens['location'])
        e_time = self.time_emb(tokens['time'])
        e_mat = self.mat_emb(tokens['material'])
        #e_method = self.method_emb(tokens['method_id'])
        e_qty = self.quantity_proj(tokens['quantity'].unsqueeze(-1).float())
        #e_toktype = self.token_type_emb(tokens['token_type_id'])

        #return self.dropout(e_type + e_loc + e_time + e_mat + e_method + e_qty + e_toktype)
        return self.dropout(e_type + e_loc + e_time + e_mat + e_qty)
    

# --- Transformer Model ---
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
        self.type_out = nn.Linear(d_model, config['num_token_types'])
        self.material_out = nn.Linear(d_model, config['num_materials'])
        self.location_out = nn.Linear(d_model, config['num_locations'])
        self.start_time_out = nn.Linear(d_model, config['num_time_steps'])
        self.end_time_out = nn.Linear(d_model, config['num_time_steps'])
        self.quantity_out = nn.Linear(d_model, 1)
        #self.method_out = nn.Linear(d_model, config['num_methods'])
        #self.ref_id_out = nn.Linear(d_model, 64)  # Assume 64 is max number of ref_ids
        #self.depends_on_out = nn.Linear(d_model, 64)  # Same assumption

    def forward(self, src_tokens, tgt_tokens):
        src = self.embed(src_tokens)
        tgt = self.embed(tgt_tokens)

        memory = self.encoder(src)
        decoded = self.decoder(tgt, memory)

        return {
            'type': self.type_out(decoded),
            'material': self.material_out(decoded),
            'location': self.location_out(decoded),
            'start_time': self.start_time_out(decoded),
            'end_time': self.end_time_out(decoded),
            'quantity': self.quantity_out(decoded).squeeze(-1),
            #'method_id': self.method_out(decoded),
            #'ref_id': self.ref_id_out(decoded),
            #'depends_on': self.depends_on_out(decoded)
        }


# --- Candidate Token Generator ---
def generate_candidate_tokens(input_dict):
    """
    Generate encoder input tokens from demand dictionary.
    Each token must include numeric fields: type, location, material, time, method_id, quantity.
    """
    candidates = []
    for d in input_dict.get("demand", []):
        candidates.append({
            "type": 0,  # fixed type ID for demand
            "location": d["location"],
            "material": d["material"],
            "time": d["time"],

            "start_time": d["start_time"],
            "end_time": d["end_time"],

            #"method_id": 0,
            "quantity": d["quantity"]
        })
    return candidates


# --- Updated encode_tokens with token_type_id ---
def encode_tokens(token_list, token_type_id=1):
    def to_tensor(key, dtype=torch.long):
        return torch.tensor([t.get(key, 0) for t in token_list], dtype=dtype)
    return {
        'type': to_tensor("type"),
        'location': to_tensor("location"),
        'material': to_tensor("material"),
        'time': to_tensor("time"),

        'start_time': to_tensor("start_time"),
        'end_time': to_tensor("end_time"),

        #'method_id': to_tensor("method_id"),
        'quantity': to_tensor("quantity", dtype=torch.float),
        #'token_type_id': torch.tensor([token_type_id] * len(token_list), dtype=torch.long)
    }


# --- Training ---
def train():
    dataset = SCMDataset("data/logs")
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_batch)
    model = SCMTransformerModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    model.train()
    for epoch in range(config['epochs']):
        total_loss = 0
        for src, tgt, labels in dataloader:
            pred = model(src, tgt)
            loss = (
                F.cross_entropy(pred['material'].view(-1, config['num_materials']), labels['material'].view(-1)) +
                F.cross_entropy(pred['location'].view(-1, config['num_locations']), labels['location'].view(-1)) +
                F.cross_entropy(pred['start_time'].view(-1, config['num_time_steps']), labels['start_time'].view(-1)) +
                F.cross_entropy(pred['end_time'].view(-1, config['num_time_steps']), labels['end_time'].view(-1)) +
                F.mse_loss(pred['quantity'].view(-1), labels['quantity'].view(-1))
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), config['checkpoint_path'])
    print(f"✅ Model saved to {config['checkpoint_path']}")

# --- csv sample datasets

class SCMDataset(Dataset):
    def __init__(self, root_dir):
        self.sample_dirs = sorted(Path(root_dir).glob("*"))

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        src = self.load_demand(sample_dir / "demands.csv")

        #tgt, labels = self.load_plan(sample_dir / "workorders.csv")
        #return src, tgt, labels
        tokens = self.load_combined(sample_dir / "combined_output.csv")
        return src, tokens, tokens

    def load_demand(self, path):
        df = pd.read_csv(path)
        tokens = {
            #"type": torch.zeros((1, len(df)), dtype=torch.long),  # dummy or inferred
            "type": torch.zeros((len(df)), dtype=torch.long),  # dummy or inferred

            "location": torch.tensor(df["location_id"].values, dtype=torch.long), #.unsqueeze(0),
            "material": torch.tensor(df["material_id"].values, dtype=torch.long), #.unsqueeze(0),
            "time": torch.tensor(df["request_time"].values, dtype=torch.long), #.unsqueeze(0),
            "quantity": torch.tensor(df["quantity"].values, dtype=torch.float), #.unsqueeze(0),

            #"method_id": torch.zeros((1, len(df)), dtype=torch.long),  # dummy for input
            #"method_id": torch.zeros((len(df)), dtype=torch.long),  # dummy for input

            #"token_type_id": torch.full((1, len(df)), 0, dtype=torch.long),  # input = 0
            #"token_type_id": torch.zeros((len(df)), dtype=torch.long),  # input = 0

            "start_time": torch.zeros((len(df)), dtype=torch.long),
            "end_time": torch.zeros((len(df)), dtype=torch.long),
        }
        return tokens

    def load_plan(self, path):
        df = pd.read_csv(path)
        num_bins = config['num_time_steps']

        # Clip time values to avoid IndexError
        df["start_time"] = df["start_time"].clip(lower=0, upper=num_bins - 1).astype(int)
        df["end_time"] = df["end_time"].clip(lower=0, upper=num_bins - 1).astype(int)

        tokens = {
            #"type": torch.zeros((1, len(df)), dtype=torch.long),
            "type": torch.zeros((len(df)), dtype=torch.long),

            "location": torch.tensor(df["location_id"].values, dtype=torch.long), #.unsqueeze(0),
            "material": torch.tensor(df["material_id"].values, dtype=torch.long), #.unsqueeze(0),
            "time": torch.tensor(df["start_time"].values, dtype=torch.long), #.unsqueeze(0),

            #"method_id": torch.zeros((1, len(df)), dtype=torch.long),
            #"method_id": torch.zeros((len(df)), dtype=torch.long),

            "quantity": torch.tensor(df["quantity"].values, dtype=torch.float), #.unsqueeze(0),

            #"token_type_id": torch.full((1, len(df)), 2, dtype=torch.long),
            "token_type_id": torch.zeros((len(df)), dtype=torch.long),
        }
        labels = {
            "material": tokens["material"].clone(),
            "location": tokens["location"].clone(),
            "start_time": torch.tensor(df["start_time"].values, dtype=torch.long), #.unsqueeze(0),
            "end_time": torch.tensor(df["end_time"].values, dtype=torch.long), #.unsqueeze(0),
            "quantity": tokens["quantity"].clone(),
        }
        return tokens, labels
    
    def load_combined(self, path):
        df = pd.read_csv(path)
        num_bins = config['num_time_steps']

        # Clip time values to avoid IndexError
        #df["start_time"] = df["start_time"].clip(lower=0, upper=num_bins - 1).astype(int)
        #df["end_time"] = df["end_time"].clip(lower=0, upper=num_bins - 1).astype(int)

        tokens = {
            #"type": torch.zeros((1, len(df)), dtype=torch.long),
            "type": torch.zeros((len(df)), dtype=torch.long),

            "location": torch.tensor(df["location_id"].values, dtype=torch.long), #.unsqueeze(0),
            "material": torch.tensor(df["material_id"].values, dtype=torch.long), #.unsqueeze(0),
            "time": torch.tensor(df["start_time"].values, dtype=torch.long), #.unsqueeze(0),

            "request_time": torch.tensor(df["request_time"].values, dtype=torch.long),
            "commit_time": torch.tensor(df["commit_time"].values, dtype=torch.long),

            "start_time": torch.tensor(df["start_time"].values, dtype=torch.long), #.unsqueeze(0),
            "end_time": torch.tensor(df["end_time"].values, dtype=torch.long), #.unsqueeze(0),

            #"method_id": torch.zeros((1, len(df)), dtype=torch.long),
            #"method_id": torch.zeros((len(df)), dtype=torch.long),

            "quantity": torch.tensor(df["quantity"].values, dtype=torch.float), #.unsqueeze(0),

            #"token_type_id": torch.full((1, len(df)), 2, dtype=torch.long),
            #"token_type_id": torch.zeros((len(df)), dtype=torch.long),
        }
 
        return tokens
    
def update_config_from_static_data(config, logs_root="data/logs"):
    max_material = -1
    max_location = -1

    for sample_dir in Path(logs_root).glob("sample_*"):
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
    print(f"🔧 Config updated: {config['num_materials']} materials, {config['num_locations']} locations")

def train_stepwise():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SCMTransformerModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    dataset = SCMDataset("data/logs")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model.train()
    for epoch in range(config['epochs']):
        total_loss = 0.0
        for src, tgt, labels in dataloader:
            # Move everything to device
            src = {k: v.to(device) for k, v in src.items()}
            tgt = {k: v.to(device) for k, v in tgt.items()}
            labels = {k: v.to(device) for k, v in labels.items()}

            tgt_len = tgt['material'].shape[1]

            # Initialize tgt_tokens with a dummy BOS token (all zeros)
            tgt_tokens = {
                'type': torch.zeros((1, 1), dtype=torch.long, device=device),
                'location': torch.zeros((1, 1), dtype=torch.long, device=device),
                'material': torch.zeros((1, 1), dtype=torch.long, device=device),
                'time': torch.zeros((1, 1), dtype=torch.long, device=device),
                
                'start_time': torch.zeros((1, 1), dtype=torch.long, device=device),
                'end_time': torch.zeros((1, 1), dtype=torch.long, device=device),

                #'method_id': torch.zeros((1, 1), dtype=torch.long, device=device),
                'quantity': torch.zeros((1, 1), dtype=torch.float, device=device)
                #'token_type_id': torch.full((1, 1), 2, dtype=torch.long, device=device),  # Plan tokens
            }

            loss_accum = 0.0
            for t in range(tgt_len):
                pred = model(src, tgt_tokens)
                last_pred = {k: v[:, -1] for k, v in pred.items()}

                loss = (
                    3.0 * F.cross_entropy(last_pred['material'], labels['material'][:, t]) +
                    3.0 * F.cross_entropy(last_pred['location'], labels['location'][:, t]) +
                    1.0 * F.cross_entropy(last_pred['start_time'], labels['start_time'][:, t]) +
                    1.0 * F.cross_entropy(last_pred['end_time'], labels['end_time'][:, t]) +
                    1.0 * F.mse_loss(last_pred['quantity'], labels['quantity'][:, t])
                )

                loss_accum += loss

                # Append ground truth target token for next step (teacher forcing)
                #for key in ['type', 'location', 'material', 'time', 'method_id', 'quantity']:
                for key in ['type', 'location', 'material', 'time', 'quantity']:
                    val = tgt[key][:, t].unsqueeze(1)
                    tgt_tokens[key] = torch.cat([tgt_tokens[key], val], dim=1)

                # Ensure token_type_id is correctly appended
                #tgt_tokens['token_type_id'] = torch.cat([
                #    tgt_tokens['token_type_id'], torch.full((1, 1), 2, dtype=torch.long, device=device)
                #], dim=1)

            optimizer.zero_grad()
            loss_accum.backward()
            optimizer.step()
            total_loss += loss_accum.item()

        print(f"Epoch {epoch+1}/{config['epochs']} - Stepwise Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), config['checkpoint_path'])
    print(f"✅ Model saved to {config['checkpoint_path']}")

  
# --- Collate function (for batch padding if needed) ---
def collate_batch(batch):
    from torch.nn.utils.rnn import pad_sequence

    def stack_dicts(dicts, pad=False):
        from torch.nn.utils.rnn import pad_sequence

        result = {}
        keys = dicts[0].keys()
        for k in keys:
            items = [d[k] for d in dicts]
            if pad:
                # Determine correct dtype padding value
                dtype = items[0].dtype
                padding_value = 0.0 if dtype == torch.float else 0
                result[k] = pad_sequence(items, batch_first=True, padding_value=padding_value)
            else:
                result[k] = torch.stack(items)
        return result

    def _stack_dicts(dicts, pad=False):
        result = {}
        keys = dicts[0].keys()
        for k in keys:
            items = [d[k] for d in dicts]
            if pad:
                if items[0].dim() == 1:
                    result[k] = pad_sequence(items, batch_first=True)
                else:
                    result[k] = pad_sequence(items, batch_first=True, padding_value=0.0)
            else:
                result[k] = torch.stack(items)
        return result

    src_batch, tgt_batch, label_batch = zip(*batch)
    #return stack_dicts(src_batch), stack_dicts(tgt_batch, pad=True), stack_dicts(label_batch, pad=True)
    return stack_dicts(src_batch, pad=True), stack_dicts(tgt_batch, pad=True), stack_dicts(label_batch, pad=True)


# --- Main ---


# --- Predict Plan ---
def is_demand_balanced(demands, plan, tolerance=1e-2):
    from collections import defaultdict

    fulfilled = defaultdict(float)
    for step in plan:
        key = (step["material_id"], step["location_id"])
        fulfilled[key] += step["quantity"]

    for d in demands:
        key = (d["material_id"], d["location_id"])
        if fulfilled[key] + tolerance < d["quantity"]:
            return False

    return True

def decode_predictions(model, src_tokens, max_steps=50, threshold=0.5):
    model.eval()
    tgt_tokens = {
        'type': torch.tensor([[0]], dtype=torch.long),
        'location': torch.tensor([[0]], dtype=torch.long),
        'material': torch.tensor([[0]], dtype=torch.long),
        'time': torch.tensor([[0]], dtype=torch.long),

        'start_time': torch.tensor([[0]], dtype=torch.long),
        'end_time': torch.tensor([[0]], dtype=torch.long),

        #'method_id': torch.tensor([[0]], dtype=torch.long),
        'quantity': torch.tensor([[0.0]], dtype=torch.float),
        'id': torch.tensor([[0]], dtype=torch.long),
        #'ref_id': torch.tensor([[0]], dtype=torch.long),
        #'depends_on': torch.tensor([[0]], dtype=torch.long),

        #'token_type_id': torch.tensor([[0]], dtype=torch.long),
    }

    plan = []
    next_id = 1
    for step in range(max_steps):
        with torch.no_grad():
            out = model(src_tokens, tgt_tokens)

        try:
            def decode_val(key, use_argmax=True):
                val = out[key]
                if val.dim() == 3:
                    val = val[0, -1]
                elif val.dim() == 2:
                    val = val[-1]
                if use_argmax:
                    return val.argmax(-1).item()
                val = val.squeeze()
                return val.item() if val.numel() == 1 else val.tolist()


            m = decode_val("material")
            l = decode_val("location")
            s = decode_val("start_time")
            e = decode_val("end_time")
            
            #q = decode_val("quantity", use_argmax=False)
            q_raw = decode_val("quantity", use_argmax=False)
            q = float(q_raw[0]) if isinstance(q_raw, list) else float(q_raw)

            t = decode_val("type")
            #method = decode_val("method_id")

            #ref_val = out.get("ref_id")
            #ref = decode_val("ref_id") if ref_val is not None else -1

            #dep_val = out.get("depends_on")
            #dep = decode_val("depends_on") if dep_val is not None else -1

            #if isinstance(ref, list):
            #    ref = ref[0]
            #if isinstance(dep, list):
            #    dep = dep[0]

        except Exception as err:
            print(f"❌ Error during decoding step {step}: {err}")
            break

        if q < threshold:
            break

        work_order = {
            "id": next_id,
            "material_id": m,
            "location_id": l,
            "start_time": s,
            "end_time": e,
            "quantity": round(q, 2),
            "type": t,
            #"method_id": method,
            #"ref_id": ref,
            #"depends_on": dep if isinstance(dep, int) and dep >= 0 else None
        }
        plan.append(work_order)

        for key, val in zip(
            #['type', 'location', 'material', 'time', 'method_id', 'quantity', 'id', 'ref_id', 'depends_on'],
            ['type', 'location', 'material', 'time', 'quantity', 'id'],
            [t, l, m, s, q, next_id]
            #[t, l, m, s, method, q, next_id, ref, dep]
        ):
            val_tensor = torch.tensor([[val]], dtype=torch.float if key == 'quantity' else torch.long)
            tgt_tokens[key] = torch.cat([tgt_tokens[key], val_tensor], dim=1)

        next_id += 1

    return plan



# (New) --- Static Token Generator ---
def generate_static_tokens():
    """
    Generate tokens for static supply chain structure:
    materials, boms, locations, methods, etc.
    You may load this from a static ontology or config file.
    """
    static_tokens = []

    # Example: material master tokens
    for mat in range(config['num_materials']):
        static_tokens.append({
            "type": 3,  # material definition
            "location": 0,
            "material": mat,
            "time": 0,
            #"method_id": 0,
            "quantity": 0.0
        })

    # Example: method (operation) tokens
    for method in range(config['num_methods']):
        static_tokens.append({
            "type": 4,  # method definition
            "location": 0,
            "material": 0,
            "time": 0,
            #"method_id": method,
            "quantity": 0.0
        })

    return static_tokens



# --- Updated Combined Input Token Generator ---
def generate_encoder_input(input_dict):
    static_tokens = generate_static_tokens()
    dynamic_tokens = generate_candidate_tokens(input_dict)
    encoded_static = encode_tokens(static_tokens, token_type_id=0)
    encoded_dynamic = encode_tokens(dynamic_tokens, token_type_id=1)

    combined = {}
    for k in encoded_static:
        combined[k] = torch.cat([encoded_static[k], encoded_dynamic[k]], dim=0)
    return combined


# Patch --- Update predict_plan to use generate_encoder_input()
def predict_plan(model, input_example, threshold=0.5):
    print("📦 Input to generate_candidate_tokens:", input_example["input"])
    src_tokens = generate_encoder_input(input_example["input"])

    for k in src_tokens:
        if src_tokens[k].dim() == 1:
            src_tokens[k] = src_tokens[k].unsqueeze(0)

    plan = decode_predictions(model, src_tokens, threshold=threshold)
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

    print("\nPlan vs Demand Comparison:")
    for k in sorted(all_keys):
        p = pred_summary.get(k, 0)
        g = gt_summary.get(k, 0)
        print(f"  {k}: predicted={p:.2f}, ground_truth={g:.2f}, diff={abs(p-g):.2f}")


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
        print(f"Diff at {key}: {diff:.2f}")

    print(f"\nTotal quantity diff: {total_diff:.2f}")


# --- CLI Entrypoint ---
def main():
    update_config_from_static_data(config)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train_stepwise", action="store_true")
    parser.add_argument("--predict", action="store_true")

    args = parser.parse_args()


    if args.train_stepwise:
        train_stepwise()
    elif args.train:
        train()
    elif args.predict:
        model = SCMTransformerModel(config)
        model.load_state_dict(torch.load(config["checkpoint_path"]))

        input_example = {
            "input": {
                "demand": [
                    {"location": 1, "material": 6, "request_time": 6,  "time": 6, "start_time": 0, "end_time": 1, "quantity": 796},
                ]
            },
            "tgt": []
        }

        aps_example = [
            {"material_id": 1, "location_id": 0, "start_time": 0, "end_time": 1, "quantity": 14.0},
            {"material_id": 5, "location_id": 3, "start_time": 1, "end_time": 3, "quantity": 10.0}
        ]

        plan = predict_plan(model, input_example)
        print("\nPredicted Plan:")
        for step in plan:
            print(step)

        print("\nEvaluation vs APS:")
        evaluate_plan(plan, aps_example)

        print_plan_vs_demand(plan, aps_example)

if __name__ == "__main__":
    main()

