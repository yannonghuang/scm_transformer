import torch
from constraint import apply_bom_mask, apply_field_constraints, apply_demand_constraints
from config import get_token_type, get_token_label
from data import generate_encoder_input

@torch.no_grad()
def predict_plan(model, src_tokens, max_steps=512):
    model.eval()
    device = src_tokens['type'].device

    for k in src_tokens:
        if src_tokens[k].dim() == 1:
            src_tokens[k] = src_tokens[k].unsqueeze(0)

    # Initial output sequence (batch size 1)
    prev_tokens = {k: [0] for k in src_tokens if k not in ["parent", "child", "method"]}
    planned_demand_ids = set()
    step = 0

    #src_tokens = {k: src_tokens[k].unsqueeze(0) for k in src_tokens}

    while step < max_steps:
        # === 1. Find the next demand in src_tokens not yet planned ===
        found = False
        B, S = src_tokens['type'].shape[:2]
        for i in range(S):
            token_type = src_tokens['type'][0, i].item()
            if token_type == get_token_type('demand'):
                demand_id = src_tokens['demand'][0, i].item()
                if demand_id not in planned_demand_ids:
                    found = True
                    break

        if not found:
            break  # No more unplanned demands

        # === 2. Start a new plan with the found demand token ===
        for k in prev_tokens:
            prev_tokens[k].append(src_tokens[k][0, i].item())
        planned_demand_ids.add(demand_id)
        step += 1

        # === 3. Predict workorders until EOD for current demand ===
        while step < max_steps:
            tgt_tokens = {
                k: torch.tensor([v], device=device)
                for k, v in prev_tokens.items()
            }

            logits_dict = model(src_tokens, tgt_tokens)

            apply_bom_mask(logits_dict['material'], src_tokens, tgt_tokens)
            apply_field_constraints(logits_dict, src_tokens, tgt_tokens)
            apply_demand_constraints(logits_dict, src_tokens, tgt_tokens)

            next_token = {
                k: torch.argmax(logits_dict[k][0, -1]).item()
                for k in logits_dict
            }

            for k in prev_tokens:
                prev_tokens[k].append(next_token[k])
            step += 1

            if next_token['type'] == get_token_type('eod'):
                break

    # Convert list to tensor: shape [1, T]
    predicted_tokens = {
        k: torch.tensor([v], device=device) for k, v in prev_tokens.items()
    }
    return predicted_tokens



def mock_src_tokens():
    """Creates a mock source with two demands."""
    B, S = 1, 5  # Batch size 1, 5 tokens max
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def pad(length):
        return [0] * (S - length)

    # Fields: all have shape (B, S)
    _src_tokens = {
        'type': torch.tensor([[get_token_type('demand'), get_token_type('demand')] + pad(2)], device=device),
        'material': torch.tensor([[2, 5] + pad(2)], device=device),
        'location': torch.tensor([[3, 3] + pad(2)], device=device),
        'source_location': torch.tensor([[3, 3] + pad(2)], device=device),
        'demand': torch.tensor([[5, 8] + pad(2)], device=device),
        'quantity': torch.tensor([[5, 8] + pad(2)], device=device),
        'start_time': torch.tensor([[2, 4] + pad(2)], device=device),
        'end_time': torch.tensor([[6, 10] + pad(2)], device=device),
        'request_time': torch.tensor([[6, 10] + pad(2)], device=device),
        'commit_time': torch.tensor([[6, 10] + pad(2)], device=device),
        'lead_time': torch.tensor([[6, 10] + pad(2)], device=device),                          
    }

    import pandas as pd

    df = pd.read_csv("data/samples/depth_0/sample_0/demands.csv")

    _src_tokens = {
        "type":        torch.tensor([[0] * len(df)], device=device), # assuming 0 = demand
        "material":    torch.tensor([df["material_id"].tolist()], device=device),
        "location":    torch.tensor([df["location_id"].tolist()],device=device),
        "source_location":    torch.tensor([df["location_id"].tolist()],device=device),
        "quantity":    torch.tensor([df["quantity"].tolist()], device=device),
        "request_time":  torch.tensor([df["request_time"].tolist()], device=device),
        #"end_time":   [(r + 4) for r in df["request_time"]],  # mock commit_time = request + 4
        "demand":      torch.tensor([df["demand_id"].tolist()], device=device),

        "start_time":  torch.tensor([[0] * len(df)], device=device),
        "end_time":  torch.tensor([[0] * len(df)], device=device),
        "commit_time":  torch.tensor([[0] * len(df)], device=device),
        "lead_time":  torch.tensor([[0] * len(df)], device=device),
    }

    src_tokens = {
        "demand": torch.tensor(df["demand_id"].values, dtype=torch.long), 
        "type": torch.zeros((len(df)), dtype=torch.long),
        "location": torch.tensor(df["location_id"].values, dtype=torch.long),
        "material": torch.tensor(df["material_id"].values, dtype=torch.long),
        #"time": torch.tensor(df["request_time"].values, dtype=torch.long),

        #"quantity": torch.tensor(df["quantity"].values, dtype=torch.float),
        "quantity": torch.tensor(df["quantity"].values, dtype=torch.long),

        "start_time": torch.zeros((len(df)), dtype=torch.long),
        "end_time": torch.zeros((len(df)), dtype=torch.long),
        "request_time": torch.tensor(df["request_time"].values, dtype=torch.long),
        "commit_time": torch.tensor(df["commit_time"].values, dtype=torch.long),
        "lead_time": torch.zeros((len(df)), dtype=torch.long),

        "parent": torch.zeros((len(df)), dtype=torch.long),
        "child": torch.zeros((len(df)), dtype=torch.long),

        "source_location": torch.tensor(df["location_id"].values, dtype=torch.long),

        "seq_in_demand": torch.zeros((len(df)), dtype=torch.long),
        "total_in_demand": torch.zeros((len(df)), dtype=torch.long),
        "successor": torch.zeros((len(df)), dtype=torch.long),
    }
    
    #return src_tokens
    return generate_encoder_input(src_tokens, istensor=True)


def decode_tokens(tokens):
    """Pretty prints the output plan."""
    T = tokens['type'].shape[1]
    rows = []
    for t in range(T):
        row = {
            k: tokens[k][0, t].item() for k in tokens
        }
        row['type_str'] = get_token_label(row['type'])  # Assuming get_token_label exists
        rows.append(row)
    return rows

def test_predict_plan(model):
    print("=== Running test_predict_plan ===")
    src_tokens = mock_src_tokens()
    
    predicted = predict_plan(model, src_tokens, max_steps=100)
    rows = decode_tokens(predicted)
    for i, row in enumerate(rows):
        print(f"{i:02d}: {row}")
