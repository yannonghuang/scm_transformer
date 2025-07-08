import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
import random

# --- Config ---
config = {
    'num_token_types': 9,
    'num_locations': 10,
    'num_time_steps': 20,
    'num_materials': 30,
    'num_methods': 30,
    'd_model': 128,
    'n_heads': 4,
    'd_ff': 256,
    'n_layers': 4,
    'dropout': 0.1,
    'batch_size': 8,
    'lr': 1e-4,
    'epochs': 30,
    'checkpoint_path': 'scm_transformer.pt'
}

# --- Embedding Module ---
class SCMEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.type_emb = nn.Embedding(config['num_token_types'], config['d_model'])
        self.loc_emb = nn.Embedding(config['num_locations'], config['d_model'])
        self.time_emb = nn.Embedding(config['num_time_steps'], config['d_model'])
        self.mat_emb = nn.Embedding(config['num_materials'], config['d_model'])
        self.method_emb = nn.Embedding(config['num_methods'], config['d_model'])

        self.quantity_proj = nn.Sequential(
            nn.Linear(1, config['d_model']),
            nn.ReLU(),
            nn.LayerNorm(config['d_model'])
        )
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, tokens):
        e_type = self.type_emb(tokens['type'])
        e_loc = self.loc_emb(tokens['location'])
        e_time = self.time_emb(tokens['time'])
        e_mat = self.mat_emb(tokens['material'])
        e_method = self.method_emb(tokens['method_id'])
        e_qty = self.quantity_proj(tokens['quantity'].unsqueeze(-1).float())

        return self.dropout(e_type + e_loc + e_time + e_mat + e_method + e_qty)

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
        self.material_out = nn.Linear(d_model, config['num_materials'])
        self.location_out = nn.Linear(d_model, config['num_locations'])
        self.start_time_out = nn.Linear(d_model, config['num_time_steps'])
        self.end_time_out = nn.Linear(d_model, config['num_time_steps'])
        self.quantity_out = nn.Linear(d_model, 1)

    def forward(self, src_tokens, tgt_tokens):
        src = self.embed(src_tokens)
        tgt = self.embed(tgt_tokens)

        memory = self.encoder(src)
        decoded = self.decoder(tgt, memory)

        return {
            'material': self.material_out(decoded),
            'location': self.location_out(decoded),
            'start_time': self.start_time_out(decoded),
            'end_time': self.end_time_out(decoded),
            'quantity': self.quantity_out(decoded).squeeze(-1)
        }

# --- Toy Data Generator ---
def _generate_candidate_tokens(raw_data):
    tokens = []
    for sample in raw_data:
        if not isinstance(sample, dict):
            continue  # skip malformed entries

        # Try to get a list of demands (training case)
        demand_items = sample.get("demand") if "demand" in sample else [sample]

        if not isinstance(demand_items, list):
            continue  # skip malformed demand entries

        for d in demand_items:
            tokens.append({
                "type": d.get("type", 0),
                "material": d.get("material", 0),
                "location": d.get("location", 0),
                "quantity": d.get("quantity", 0),
                "start_time": d.get("start_time", 0),
                "end_time": d.get("end_time", 0),
                "method_id": d.get("method_id", 0),
                "route_id": d.get("route_id", 0),
                "op_id": d.get("op_id", 0),
                "resource_id": d.get("resource_id", 0),
            })

    return tokens

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
            "method_id": 0,
            "quantity": d["quantity"]
        })
    return candidates


def encode_tokens(token_list):
    def to_tensor(key, dtype=torch.long):
        return torch.tensor([t.get(key, 0) for t in token_list], dtype=dtype)
    return {
        'type': to_tensor("type"),
        'location': to_tensor("location"),
        'material': to_tensor("material"),
        'time': to_tensor("time"),
        'method_id': to_tensor("method_id"),
        'quantity': to_tensor("quantity", dtype=torch.float)
    }

# --- Dataset ---
class SCMDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        for file in os.listdir(data_dir):
            if file.endswith(".json"):
                with open(os.path.join(data_dir, file), 'r') as f:
                    self.samples.append(json.load(f))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        src = encode_tokens(generate_candidate_tokens(entry["input"]))

        # Shifted decoder input (tgt) and labels for teacher forcing
        tgt_tokens = entry["aps_plan"]
        tgt_input_tokens = tgt_tokens[:-1]
        tgt_label_tokens = tgt_tokens[1:] if len(tgt_tokens) > 1 else tgt_tokens

        tgt = encode_tokens(tgt_input_tokens)
        labels = {
            'material': torch.tensor([t["material"] for t in tgt_label_tokens]),
            'location': torch.tensor([t["location"] for t in tgt_label_tokens]),
            'start_time': torch.tensor([t["start_time"] for t in tgt_label_tokens]),
            'end_time': torch.tensor([t["end_time"] for t in tgt_label_tokens]),
            'quantity': torch.tensor([t["quantity"] for t in tgt_label_tokens], dtype=torch.float)
        }
        return src, tgt, labels


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
import argparse

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

def predict_plan(model, input_example, threshold=0.5):
    model.eval()

    print("📦 Input to generate_candidate_tokens:", input_example["input"])

    src_candidates = generate_candidate_tokens(input_example["input"])
    src_tokens = encode_tokens(src_candidates)

    print("🔍 Source input tokens (first few):")

    for k in src_tokens:
        if src_tokens[k].dim() == 1:
            src_tokens[k] = src_tokens[k].unsqueeze(0)

    # Initialize with a dummy first token to avoid empty tgt
    tgt_tokens = {
        'type': torch.tensor([[0]], dtype=torch.long),
        'location': torch.tensor([[0]], dtype=torch.long),
        'material': torch.tensor([[0]], dtype=torch.long),
        'time': torch.tensor([[0]], dtype=torch.long),
        'method_id': torch.tensor([[0]], dtype=torch.long),
        'quantity': torch.tensor([[0.0]], dtype=torch.float)
    }

    plan = []
    for _ in range(10):  # max 10 planning steps
        with torch.no_grad():
            out = model(src_tokens, tgt_tokens)

        m = out["material"][:, -1].argmax(-1).item()
        l = out["location"][:, -1].argmax(-1).item()
        s = out["start_time"][:, -1].argmax(-1).item()
        e = out["end_time"][:, -1].argmax(-1).item()
        q = out["quantity"][:, -1].item()

        if q < threshold:
            break

        plan.append({
            "material_id": m,
            "location_id": l,
            "start_time": s,
            "end_time": e,
            "quantity": round(q, 2)
        })

        for key, val in zip(['type', 'location', 'material', 'time', 'method_id', 'quantity'],
                            [0, l, m, s, 0, q]):
            val_tensor = torch.tensor([[val]], dtype=torch.float if key == 'quantity' else torch.long)
            tgt_tokens[key] = torch.cat([tgt_tokens[key], val_tensor], dim=1)

    return plan

def _predict_plan(model, input_example, threshold=0.5):
    model.eval()
    src_candidates = generate_candidate_tokens(input_example["input"])
    src_tokens = encode_tokens(src_candidates)

    for k in src_tokens:
        if src_tokens[k].dim() == 1:
            src_tokens[k] = src_tokens[k].unsqueeze(0)

    # Initialize with a dummy first token to avoid empty tgt
    tgt_tokens = {
        'type': torch.tensor([[0]], dtype=torch.long),
        'location': torch.tensor([[0]], dtype=torch.long),
        'material': torch.tensor([[0]], dtype=torch.long),
        'time': torch.tensor([[0]], dtype=torch.long),
        'method_id': torch.tensor([[0]], dtype=torch.long),
        'quantity': torch.tensor([[0.0]], dtype=torch.float)
    }

    plan = []
    for _ in range(10):  # max 10 planning steps
        with torch.no_grad():
            out = model(src_tokens, tgt_tokens)

        m = out["material"][:, -1].argmax(-1).item()
        l = out["location"][:, -1].argmax(-1).item()
        s = out["start_time"][:, -1].argmax(-1).item()
        e = out["end_time"][:, -1].argmax(-1).item()
        q = out["quantity"][:, -1].item()

        if q < threshold:
            break

        plan.append({
            "material_id": m,
            "location_id": l,
            "start_time": s,
            "end_time": e,
            "quantity": round(q, 2)
        })

        for key, val in zip(['type', 'location', 'material', 'time', 'method_id', 'quantity'],
                            [0, l, m, s, 0, q]):
            val_tensor = torch.tensor([[val]], dtype=torch.float if key == 'quantity' else torch.long)
            tgt_tokens[key] = torch.cat([tgt_tokens[key], val_tensor], dim=1)

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

def _predict_plan(model, input_example, threshold=0.5):
    model.eval()
    src_candidates = generate_candidate_tokens(input_example["input"])
    src_tokens = encode_tokens(src_candidates)

    for k in src_tokens:
        if src_tokens[k].dim() == 1:
            src_tokens[k] = src_tokens[k].unsqueeze(0)

    # Initialize with a dummy first token to avoid empty tgt
    tgt_tokens = {
        'type': torch.tensor([[0]], dtype=torch.long),
        'location': torch.tensor([[0]], dtype=torch.long),
        'material': torch.tensor([[0]], dtype=torch.long),
        'time': torch.tensor([[0]], dtype=torch.long),
        'method_id': torch.tensor([[0]], dtype=torch.long),
        'quantity': torch.tensor([[0.0]], dtype=torch.float)
    }

    plan = []
    for _ in range(10):  # max 10 planning steps
        with torch.no_grad():
            out = model(src_tokens, tgt_tokens)

        m = out["material"][:, -1].argmax(-1).item()
        l = out["location"][:, -1].argmax(-1).item()
        s = out["start_time"][:, -1].argmax(-1).item()
        e = out["end_time"][:, -1].argmax(-1).item()
        q = out["quantity"][:, -1].item()

        if q < threshold:
            break

        plan.append({
            "material_id": m,
            "location_id": l,
            "start_time": s,
            "end_time": e,
            "quantity": round(q, 2)
        })

        for key, val in zip(['type', 'location', 'material', 'time', 'method_id', 'quantity'],
                            [0, l, m, s, 0, q]):
            val_tensor = torch.tensor([[val]], dtype=torch.float if key == 'quantity' else torch.long)
            tgt_tokens[key] = torch.cat([tgt_tokens[key], val_tensor], dim=1)

    return plan


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

# --- Generate Synthetic Data ---
def generate_synthetic_data(n=1000, out_dir="data/logs"):
    import random, os, json
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        input_demand = []
        plan = []
        for j in range(random.randint(1, 3)):
            loc = random.randint(0, 9)
            mat = random.randint(0, 29)
            time = random.randint(0, 19)
            qty = round(random.uniform(5, 20), 2)
            input_demand.append({"location": loc, "material": mat, "time": time, "quantity": qty})
            plan.append({
                "material": mat,
                "location": loc,
                "start_time": time,
                "end_time": min(time + random.randint(1, 5), 19),
                "quantity": qty
                #"quantity": qty - random.uniform(0, 1.0)
            })
        with open(os.path.join(out_dir, f"sample_{i:04}.json"), "w") as f:
            json.dump({"input": {"demand": input_demand}, "aps_plan": plan}, f, indent=2)

def _generate_synthetic_data(output_dir, num_samples=100, num_demands=5):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_samples):
        input_demands = []
        aps_plan = []

        for _ in range(num_demands):
            material = random.randint(0, 29)
            location = random.randint(0, 9)
            time = random.randint(0, 15)
            quantity = random.randint(10, 50)

            # Demand
            input_demands.append({
                "material": material,
                "location": location,
                "time": time,
                "quantity": quantity
            })

            # A plausible (but imperfect) plan to fulfill the demand
            plan_start = max(0, time - random.randint(0, 2))
            plan_end = plan_start + random.randint(1, 3)
            delivered_quantity = quantity - random.uniform(0.0, 2.5)

            aps_plan.append({
                "material": material,
                "location": location,
                "start_time": plan_start,
                "end_time": plan_end,
                "quantity": round(delivered_quantity, 1)
            })

        sample = {
            "input": {
                "demand": input_demands
            },
            "aps_plan": aps_plan
        }

        with open(os.path.join(output_dir, f"sample_{i:03d}.json"), "w") as f:
            json.dump(sample, f, indent=2)

    print(f"✅ Generated {num_samples} synthetic samples in {output_dir}")

# --- CLI Entrypoint ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--data", action="store_true")
    args = parser.parse_args()

    if args.data:
        generate_synthetic_data()
    elif args.train:
        train()
    elif args.predict:
        model = SCMTransformerModel(config)
        model.load_state_dict(torch.load(config["checkpoint_path"]))

        input_example = {
            "input": {
                "demand": [
                    {"location": 0, "material": 1, "time": 0, "quantity": 15},
                    {"location": 3, "material": 5, "time": 1, "quantity": 10}
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

