import json
from datetime import date
from aps import run_aps
from model import load_model, predict_plan

def run_and_log_day(input_data):
    aps_plan = run_aps(input_data)
    model = load_model("checkpoint.pt")
    transformer_plan = predict_plan(model, input_data)

    log = {
        "date": str(date.today()),
        "input": input_data,
        "aps_plan": aps_plan,
        "transformer_plan": transformer_plan,
    }

    with open(f"logs/{log['date']}.json", "w") as f:
        json.dump(log, f, indent=2)
