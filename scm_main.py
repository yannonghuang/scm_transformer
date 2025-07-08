# scm_main.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import json

# --- Config ---
config = {
    "d_model": 64,
    "n_heads": 4,
    "d_ff": 128,
    "n_layers": 2,
    "dropout": 0.1,
    "batch_size": 8,
    "epochs": 5,
    "lr": 1e-3,
    "checkpoint_path": "scm_model.pt",
    "num_token_types": 9,
    "num_locations": 5,
    "num_materials": 10,
    "num_time_steps": 10,
    "num_methods": 5
}

# --- Sample Data ---
def create_sample_data(n=10):
    samples = []
    for _ in range(n):
        input_data = {
            "demand": [{
                "type": 4,
                "location": random.randint(0, 4),
                "material": random.randint(0, 9),
                "time": random.randint(0, 9),
                "quantity": round(random.uniform(10, 50), 2)
            } for _ in range(2)]
        }
        aps_plan = [{
            "type": 1,
            "location": d["location"],
            "material": d["material"],
            "time": d["time"],
            "method_id": 0,
            "quantity": d["quantity"] - random.uniform(0, 5)
        } for d in input_data["demand"]]
        samples.append({"input": input_data, "aps_plan": aps_plan})
    return samples

# --- Token Flattening ---
def generate_candidate_tokens(input_data):
    tokens = []
    for d in input_data.get("demand", []):
        tokens.append({
            "type": d["type"],
            "location": d["location"],
            "material": d["material"],
            "time": d["time"],
            "method_id": 0,
            "quantity": d["quantity"]
        })
    return tokens

# --- Encoding ---
def encode_tokens(token_list):
    def to_tensor(field, dtype=torch.long):
        return torch.tensor([t.get(field, 0) for t in token_list], dtype=dtype)

    return {
        "type": to_tensor("type"),
        "location": to_tensor("location"),
        "material": to_tensor("material"),
        "time": to_tensor("time"),
        "method_id": to_tensor("method_id"),
        "quantity": to_tensor("quantity", dtype=torch.float)
    }

# --- Transformer Model (Encoder-Only) ---
class SCMEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.type_emb = nn.Embedding(config["num_token_types"], config["d_model"])
        self.loc_emb = nn.Embedding(config["num_locations"], config["d_model"])
        self.time_emb = nn.Embedding(config["num_time_steps"], config["d_model"])
        self.mat_emb = nn.Embedding(config["num_materials"], config["d_model"])
        self.method_emb = nn.Embedding(config["num_methods"], config["d_model"])
        self.quantity_proj = nn.Linear(1, config["d_model"])
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, tokens):
        e_type = self.type_emb(tokens["type"])
        e_loc = self.loc_emb(tokens["location"])
        e_time = self.time_emb(tokens["time"])
        e_mat = self.mat_emb(tokens["material"])
        e_method = self.method_emb(tokens["method_id"])
        e_qty = self.quantity_proj(tokens["quantity"].unsqueeze(-1))
        return self.dropout(e_type + e_loc + e_time + e_mat + e_method + e_qty)

class SCMTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = SCMEmbedding(config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["n_heads"],
            dim_feedforward=config["d_ff"],
            dropout=config["dropout"],
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config["n_layers"])
        self.output_head = nn.Sequential(
            nn.Linear(config["d_model"], config["d_model"]),
            nn.ReLU(),
            nn.Linear(config["d_model"], 1)
        )

    def forward(self, tokens, mask=None):
        x = self.embedding(tokens)
        x = self.encoder(x, src_key_padding_mask=mask)
        return self.output_head(x).squeeze(-1)

# --- Dataset ---
class SCMDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        candidates = generate_candidate_tokens(entry["input"])
        input_tokens = encode_tokens(candidates)
        label_dict = {
            (a["type"], a["location"], a["material"], a["time"], a["method_id"]): a["quantity"]
            for a in entry["aps_plan"]
        }
        targets = []
        for i in range(len(input_tokens["type"])):
            key = (
                input_tokens["type"][i].item(),
                input_tokens["location"][i].item(),
                input_tokens["material"][i].item(),
                input_tokens["time"][i].item(),
                input_tokens["method_id"][i].item(),
            )
            targets.append(label_dict.get(key, 0.0))
        input_tokens["target"] = torch.tensor(targets, dtype=torch.float)
        return input_tokens

# --- Training ---
def train(model, dataloader, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(config["epochs"]):
        total_loss = 0
        for batch in dataloader:
            targets = batch.pop("target")
            attention_mask = torch.ones_like(batch["type"], dtype=torch.bool)
            optimizer.zero_grad()
            outputs = model(batch, mask=~attention_mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# --- Main ---
if __name__ == "__main__":
    samples = create_sample_data(n=20)
    dataset = SCMDataset(samples)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model = SCMTransformerModel(config)
    train(model, dataloader, config)
    torch.save(model.state_dict(), config["checkpoint_path"])
    print("Model training complete and saved.")
