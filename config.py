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

# Ensure the logs/ directory exists
os.makedirs("logs", exist_ok=True)

# Create a rotating file handler: 5 MB per file, keep up to 3 files
rotating_handler = RotatingFileHandler(
    "logs/training.log", maxBytes=50 * 1024 * 1024, backupCount=5
)

# Console handler to also log to stdout
console_handler = logging.StreamHandler()

# Set formatter for both handlers
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
rotating_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Get the root logger and set the level
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Set global log level here

# Add both handlers to the logger
logger.addHandler(rotating_handler)
logger.addHandler(console_handler)


# --- Config ---
token_types = [
    'demand',
    'make',
    'purchase',
    'move',
    'material',
    'location',
    'method',
    'bom',
    'eod'
]

def get_token_type(t):
    return token_types.index(t)

def get_token_label(id):
    return token_types[id]

config = {
    'num_token_types': len(token_types), #9,
    'num_demands': 10,
    'num_locations': 4,
    'num_time_steps': 30,
    'request_time_range': 10,
    'num_materials': 70,
    'num_methods': 600,
    'd_model': 128,
    'n_heads': 4,
    'd_ff': 256,
    'n_layers': 4,
    'dropout': 0.1,
    'batch_size': 8,
    'lr': 1e-4,
    'epochs': 20,
    'checkpoint_name': 'scm_transformer',
    "max_train_samples": 1000,
    'quantity_scale': 1,  # updated to allow integer binning
    'max_quantity': 100 #1e5
}
