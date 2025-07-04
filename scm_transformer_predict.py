# scm_transformer_predict.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
import json
import os
import matplotlib.pyplot as plt

# --- Config ---
config = {
    'num_token_types': 5,
    'num_locations': 10,
    'num_time_steps': 20,
    'num_materials': 30,
    'num_methods': 10,
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
        emb = e_type + e_loc + e_time + e_mat + e_method + e_qty
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
        input_tokens = encode_tokens(generate_candidate_tokens(entry["input"]))
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
        if step['type'] == 1:  # produce
            cost += qty * 1.0
        elif step['type'] == 2:  # ship
            cost += qty * 0.5
        elif step['type'] == 3:  # purchase
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

# --- Prediction Function ---
def predict_plan(model, input_data: Dict[str, Any], threshold=1.0) -> List[Dict[str, Any]]:
    model.eval()

    token_list = generate_candidate_tokens(input_data)
    tokens = encode_tokens(token_list)
    attention_mask = torch.ones(tokens['type'].shape, dtype=torch.bool)

    with torch.no_grad():
        predictions = model(tokens, attention_mask)

    plan = []
    for tok, qty in zip(token_list, predictions.tolist()):
        if qty > threshold:
            tok["quantity"] = qty
            plan.append(tok)

    return plan

def generate_candidate_tokens(input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = []
    for loc in range(config['num_locations']):
        for method in range(config['num_methods']):
            mat = method % config['num_materials']
            candidates.append({
                "type": 1,  # produce
                "location": loc,
                "time": 0,
                "material": mat,
                "method_id": method,
                "quantity": 0.0
            })
    for d in input_data.get("demand", []):
        candidates.append({
            "type": 4,  # demand
            "location": d["location"],
            "time": d["time"],
            "material": d["material"],
            "method_id": 0,
            "quantity": d["quantity"]
        })
    return candidates
    
def encode_tokens(token_list: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    def to_tensor(field):
        return torch.tensor([t[field] for t in token_list], dtype=torch.long)
    return {
        "type": to_tensor("type"),
        "location": to_tensor("location"),
        "time": to_tensor("time"),
        "material": to_tensor("material"),
        "method_id": to_tensor("method_id"),
        "quantity": torch.tensor([t.get("quantity", 0.0) for t in token_list], dtype=torch.float)
    }

# --- Sample Unit Tests ---
def run_unit_tests():
    print("Running unit tests...")

    sample_input = {
        "demand": [
            {"location": 0, "material": 0, "time": 0, "quantity": 10.0},
            {"location": 1, "material": 2, "time": 1, "quantity": 5.0}
        ]
    }

    plan = predict_plan(model, sample_input, threshold=0.0)
    assert isinstance(plan, list), "Output should be a list"
    assert all("quantity" in p for p in plan), "Each plan step must include quantity"
    print(f"Generated {len(plan)} plan steps. ✔️")

    sim_result = simulate_execution(plan)
    assert 'cost' in sim_result, "Sim result must include cost"
    assert isinstance(sim_result['inventory'], dict), "Inventory should be a dictionary"
    print(f"Simulated cost: {sim_result['cost']:.2f}. ✔️")

# --- Sample JSON Logs ---
def create_sample_log_files():
    os.makedirs("logs", exist_ok=True)
    for i in range(3):
        sample = {
            "input": {
                "demand": [
                    {"location": 0, "material": i, "time": 0, "quantity": 10.0 + i}
                ]
            },
            "aps_plan": [
                {"type": 1, "location": 0, "material": i, "time": 0, "method_id": 0, "quantity": 10.0 + i}
            ]
        }
        with open(f"logs/sample_{i}.json", "w") as f:
            json.dump(sample, f, indent=2)

# (All previous definitions remain unchanged...)

# --- Example Usage ---
if __name__ == "__main__":
# --- Example Usage ---
    create_sample_log_files()
    
    model = SCMTransformerModel(config)

    run_unit_tests()

    # Train
    dataset = SCMDataset("logs")
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    train_model(model, dataloader, config)
    save_model(model, config['checkpoint_path'])

    # Predict and evaluate
    dummy_input_data = {
        "demand": [{"location": 0, "material": 1, "time": 0, "quantity": 20.0}]
    }
    transformer_plan = predict_plan(model, dummy_input_data)

    # Stub APS plan for comparison (replace with real APS output)
    aps_plan = [{"type": 1, "location": 0, "material": 1, "time": 0, "method_id": 0, "quantity": 18.0}]

    # Simulate and compare
    sim_aps = simulate_execution(aps_plan)
    sim_trans = simulate_execution(transformer_plan)
    print("APS Cost:", sim_aps['cost'])
    print("Transformer Cost:", sim_trans['cost'])
    compare_plans(aps_plan, transformer_plan)

    for step in transformer_plan:
        print(step)
