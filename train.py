from model import train_model, load_model
from data_utils import load_training_data

def run_training():
    data = load_training_data("logs/")
    model = load_model("checkpoint.pt")
    train_model(model, data)
