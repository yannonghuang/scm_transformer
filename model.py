import torch
import torch.nn as nn

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
        # tokens is a dict of tensor fields: shape [batch, seq_len]
        e_type = self.type_emb(tokens['type'])
        e_loc = self.loc_emb(tokens['location'])
        e_time = self.time_emb(tokens['time'])
        e_mat = self.mat_emb(tokens['material'])
        e_method = self.method_emb(tokens['method_id'])
        e_qty = self.quantity_proj(tokens['quantity'].unsqueeze(-1).float())  # [B, L, 1] → [B, L, d_model]

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
            nn.Linear(config['d_model'], 1)  # Predict a quantity or score
        )

    def forward(self, tokens, attention_mask=None):
        x = self.embedding(tokens)  # [B, L, d_model]
        if attention_mask is not None:
            # PyTorch expects attention_mask: 1=keep, 0=mask
            # Convert to bool mask for Transformer
            attention_mask = attention_mask.bool()
        x = self.encoder(x, src_key_padding_mask=~attention_mask if attention_mask is not None else None)
        out = self.output_head(x)  # [B, L, 1]
        return out.squeeze(-1)

config = {
    'num_token_types': 5,      # inventory, produce, ship, purchase, demand
    'num_locations': 10,
    'num_time_steps': 20,
    'num_materials': 30,
    'num_methods': 10,
    'd_model': 128,
    'n_heads': 4,
    'd_ff': 256,
    'n_layers': 4,
    'dropout': 0.1
}

model = SCMTransformerModel(config)

# Dummy batch of tokens
batch_size = 2
seq_len = 10

tokens = {
    'type': torch.randint(0, 5, (batch_size, seq_len)),
    'location': torch.randint(0, 10, (batch_size, seq_len)),
    'time': torch.randint(0, 20, (batch_size, seq_len)),
    'material': torch.randint(0, 30, (batch_size, seq_len)),
    'method_id': torch.randint(0, 10, (batch_size, seq_len)),
    'quantity': torch.randn(batch_size, seq_len) * 10
}

attention_mask = torch.ones(batch_size, seq_len)

output = model(tokens, attention_mask)  # shape: [batch_size, seq_len]
print("Output shape:", output.shape)

def predict_plan(model, input_data, threshold=1.0):
    """
    Generate a production plan using the transformer model.
    
    input_data: {
        'inventory': [...],
        'demand': [...],
        'date': ..., etc.
    }
    """
    model.eval()
    
    # 1. Generate candidate tokens (simplified example)
    token_list = generate_candidate_tokens(input_data)
    tokens = encode_tokens(token_list)  # Dict of tensors for model input
    
    # 2. Predict quantities for each token
    with torch.no_grad():
        predictions = model(tokens)  # shape: [batch_size, seq_len]
    
    # 3. Filter by threshold
    plan = []
    for tok, qty in zip(token_list, predictions.squeeze(-1).tolist()):
        if qty > threshold:
            tok["quantity"] = qty
            plan.append(tok)
    
    return plan

def generate_candidate_tokens(input_data):
    candidates = []
    
    for loc in all_locations:
        for time in [input_data['date']]:
            for method in all_methods:
                mat = method_output_material(method)
                candidates.append({
                    "type": "produce",
                    "location": loc,
                    "time": time,
                    "material": mat,
                    "method_id": method,
                    "quantity": 0  # to be predicted
                })
    
    # Add demand tokens as input-only (not part of output plan)
    for d in input_data["demand"]:
        candidates.append({
            "type": "demand",
            "location": d["location"],
            "time": d["time"],
            "material": d["material"],
            "quantity": d["quantity"]
        })
    
    return candidates

def encode_tokens(token_list):
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
