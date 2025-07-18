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

from transformer import restore_model
from utils import update_config_from_static_data
from train import train, train_stepwise
from predict import predict_plan, evaluate_plan, print_plan_vs_demand

from config import config, logger

# --- CLI Entrypoint ---
def main():
    update_config_from_static_data(config)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train_stepwise", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("-d", "--depth", help='Starting from 0')

    args = parser.parse_args()

    if args.train_stepwise:
        train_stepwise()
    elif args.train:
        train(args)
    elif args.predict:
        _, model = restore_model()
        input_example = {
            "input": {
                "demand": [
                    {"type": 0, "demand_id": 0, "location_id": 1, "material_id": 89, "request_time": 8, "commit_time": 8, "start_time": 8, "end_time": 8, "quantity": 339},
                ]
            },
            "tgt": []
        }

        aps_example = [
            {"type": 2, "demand_id": 0, "location_id": 1, "material_id": 89, "request_time": 8, "commit_time": 8, "start_time": 5, "end_time": 8, "quantity": 339},
        ]

        plan = predict_plan(model, input_example)
        logger.info("\nPredicted Plan:")
        for step in plan:
            logger.info(step)

        logger.info("\nEvaluation vs APS:")
        evaluate_plan(plan, aps_example)

        print_plan_vs_demand(plan, aps_example)

if __name__ == "__main__":
    main()

