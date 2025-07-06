# scm_transformer_predict.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
    'batch_size': 16,
    'lr': 1e-4,
    'epochs': 10,
    'checkpoint_path': 'scm_model.pt'
}

# --- SCM Transformer Model ---
class SCMEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.type_emb = nn.Embedding(config['num_token_types'], config['d_model'])
        self.loc_emb = nn.Embedding(config['num_locations'], config['d_model'])
        self.time_emb = nn.Embedding(config['num_time_steps'], config['d_model'])
        self.mat_emb = nn.Embedding(config['num_materials'], config['d_model'])
        self.method_emb = nn.Embedding(config['num_methods'], config['d_model'])
        self.route_emb = nn.Embedding(10, config['d_model'])
        self.op_emb = nn.Embedding(20, config['d_model'])
        self.resource_emb = nn.Embedding(20, config['d_model'])

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

        e_route = self.route_emb(tokens['route_id'])
        e_op = self.op_emb(tokens['op_id'])
        e_res = self.resource_emb(tokens['resource_id'])

        emb = e_type + e_loc + e_time + e_mat + e_method + e_qty + e_route + e_op + e_res

        return self.dropout(emb)

class SCMTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = SCMEmbedding(config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['n_heads'],
            dim_feedforward=config['d_ff'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['n_layers'])
        self.output_head = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
            nn.Linear(config['d_model'], 1)
        )

    def forward(self, tokens, attention_mask=None):
        x = self.embedding(tokens)
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
        x = self.encoder(x, src_key_padding_mask=~attention_mask if attention_mask is not None else None)
        out = self.output_head(x)
        return out.squeeze(-1)

# --- SCM Dataset ---
class SCMDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        for file in os.listdir(data_dir):
            if file.endswith(".json"):
                with open(os.path.join(data_dir, file), 'r') as f:
                    self.data.append(json.load(f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        candidate_tokens = generate_candidate_tokens(entry["input"])
        input_tokens = encode_tokens(candidate_tokens)
        label_plan = { (a['type'], a['location'], a['material'], a['time'], a['method_id']): a['quantity'] for a in entry['aps_plan'] }

        targets = []
        for i in range(len(input_tokens['type'])):
            key = (
                input_tokens['type'][i].item(),
                input_tokens['location'][i].item(),
                input_tokens['material'][i].item(),
                input_tokens['time'][i].item(),
                input_tokens['method_id'][i].item(),
            )
            targets.append(label_plan.get(key, 0.0))

        input_tokens['target'] = torch.tensor(targets, dtype=torch.float)
        return input_tokens

# --- Save/Load Model Checkpoint ---
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model_class, path, config):
    model = model_class(config)
    model.load_state_dict(torch.load(path))
    return model

# --- Plan Simulator ---
def simulate_execution(plan: List[Dict[str, Any]]) -> Dict[str, float]:
    inventory = {}
    cost = 0.0
    for step in plan:
        key = (step['location'], step['material'])
        qty = step['quantity']
        inventory[key] = inventory.get(key, 0.0) + qty
        if step['type'] == 1:
            cost += qty * 1.0
        elif step['type'] == 2:
            cost += qty * 0.5
        elif step['type'] == 3:
            cost += qty * 1.2
    return {'inventory': inventory, 'cost': cost}

# --- Plan Comparison Dashboard ---
def compare_plans(aps_plan, transformer_plan):
    def to_dict(plan):
        d = {}
        for p in plan:
            key = (p['type'], p['location'], p['material'], p['time'])
            d[key] = d.get(key, 0.0) + p['quantity']
        return d

    aps_dict = to_dict(aps_plan)
    trans_dict = to_dict(transformer_plan)
    keys = set(aps_dict) | set(trans_dict)
    diffs = [trans_dict.get(k, 0) - aps_dict.get(k, 0) for k in keys]

    plt.hist(diffs, bins=20, edgecolor='black')
    plt.title('Transformer vs APS Plan Quantity Difference')
    plt.xlabel('Quantity Difference')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()

# --- Training Loop ---
def train_model(model, dataloader, config):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()

    for epoch in range(config['epochs']):
        total_loss = 0
        for batch in dataloader:
            attention_mask = torch.ones(batch['type'].shape, dtype=torch.bool)
            target = batch.pop("target")
            optimizer.zero_grad()
            output = model(batch, attention_mask)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {total_loss:.4f}")

# --- Plot Attention Heatmap ---
def plot_attention_heatmap(attn_weights, title="Attention Heatmap"):
    avg_attn = attn_weights.mean(0).numpy()
    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_attn, cmap="viridis")
    plt.title(title)
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.tight_layout()
    plt.show()

# --- Generate and Encode Tokens ---
def generate_candidate_tokens(input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    bom_edges, route_defs, op_defs, res_defs = [], [], [], []
    if os.path.exists("bom_links.json"):
        with open("bom_links.json", "r") as f:
            bom_edges = json.load(f)
    if os.path.exists("route_defs.json"):
        with open("route_defs.json", "r") as f:
            route_defs = json.load(f)
    if os.path.exists("operation_defs.json"):
        with open("operation_defs.json", "r") as f:
            op_defs = json.load(f)
    if os.path.exists("resource_defs.json"):
        with open("resource_defs.json", "r") as f:
            res_defs = json.load(f)

    candidates = []
    for loc in range(config['num_locations']):
        for method in range(config['num_methods']):
            mat = method % config['num_materials']
            candidates.append({
                "type": 1,
                "location": loc,
                "time": 0,
                "material": mat,
                "method_id": method,
                "quantity": 0.0,
                "route_id": 0,
                "op_id": 0,
                "resource_id": 0
            })

    for d in input_data.get("demand", []):
        candidates.append({
            "type": 4,
            "location": d["location"],
            "time": d["time"],
            "material": d["material"],
            "method_id": 0,
            "quantity": d["quantity"],
            "route_id": 0,
            "op_id": 0,
            "resource_id": 0
        })

    for edge in bom_edges:
        candidates.append({
            "type": 5,
            "location": 0,
            "time": 0,
            "material": edge["from_material"],
            "method_id": edge["to_material"],
            "quantity": 0.0,
            "route_id": 0,
            "op_id": 0,
            "resource_id": 0
        })

    for route in route_defs:
        candidates.append({
            "type": 6,
            "location": 0,
            "time": 0,
            "material": 0,
            "method_id": route["method_id"],
            "quantity": route["route_id"],
            "route_id": route["route_id"],
            "op_id": 0,
            "resource_id": 0
        })

    for op in op_defs:
        candidates.append({
            "type": 7,
            "location": 0,
            "time": op["duration"],
            "material": op["resource_id"],
            "method_id": op["route_id"],
            "quantity": op["op_id"],
            "route_id": op["route_id"],
            "op_id": op["op_id"],
            "resource_id": op["resource_id"]
        })

    for res in res_defs:
        candidates.append({
            "type": 8,
            "location": 0,
            "time": 0,
            "material": res["resource_id"],
            "method_id": 0,
            "quantity": res["capacity"],
            "route_id": 0,
            "op_id": 0,
            "resource_id": res["resource_id"]
        })

    return candidates


def encode_tokens(token_list: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    def to_tensor(field, dtype=torch.long):
        return torch.tensor([t.get(field, 0) for t in token_list], dtype=dtype)

    return {
        "type": to_tensor("type"),
        "location": to_tensor("location"),
        "time": to_tensor("time"),
        "material": to_tensor("material"),
        "method_id": to_tensor("method_id"),
        "quantity": to_tensor("quantity", dtype=torch.float),
        "route_id": to_tensor("route_id"),
        "op_id": to_tensor("op_id"),
        "resource_id": to_tensor("resource_id"),
    }

# --- Attention Summary and Prediction ---
def print_attention_summary(model, tokens):
    hooks = []
    attention_maps = []

    def save_attention_hook(module, input, output):
        if hasattr(module, 'attn_output_weights'):
            attention_maps.append(module.attn_output_weights.detach().cpu())

    for i, layer in enumerate(model.encoder.layers):
        hooks.append(layer.self_attn.register_forward_hook(save_attention_hook))

    with torch.no_grad():
        attention_mask = torch.ones(tokens['type'].shape, dtype=torch.bool)
        model(tokens, attention_mask)

    for h in hooks:
        h.remove()

    if attention_maps:
        plot_attention_heatmap(attention_maps[0], "Transformer Attention (Layer 0)")

    return attention_maps

# --- Predict Plan ---
def predict_plan(model, input_data: Dict[str, Any], threshold=1.0) -> List[Dict[str, Any]]:
    model.eval()
    token_list = generate_candidate_tokens(input_data)
    tokens = encode_tokens(token_list)
    print_attention_summary(model, tokens)
    attention_mask = torch.ones(tokens['type'].shape, dtype=torch.bool)

    with torch.no_grad():
        predictions = model(tokens, attention_mask)

    plan = []
    for tok, qty in zip(token_list, predictions.tolist()):
        if qty > threshold:
            tok['quantity'] = qty
            plan.append(tok)

    return plan

if __name__ == "__main__":
    # Load model
    model = SCMTransformerModel(config)

    # Create dataset and dataloader
    dataset = SCMDataset("logs")
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # Train the model
    train_model(model, dataloader, config)

    # Save the trained model
    save_model(model, config['checkpoint_path'])

    # Example prediction
    dummy_input_data = {
        "demand": [{"location": 0, "material": 1, "time": 0, "quantity": 20.0}]
    }
    transformer_plan = predict_plan(model, dummy_input_data)

    # Simulate cost
    aps_plan = [{"type": 1, "location": 0, "material": 1, "time": 0, "method_id": 0, "quantity": 18.0}]
    sim_aps = simulate_execution(aps_plan)
    sim_trans = simulate_execution(transformer_plan)
    print("APS Cost:", sim_aps['cost'])
    print("Transformer Cost:", sim_trans['cost'])

    # Compare plans
    compare_plans(aps_plan, transformer_plan)

    # Print transformer plan
    for step in transformer_plan:
        print(step)
