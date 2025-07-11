import os
import random
import pandas as pd
import networkx as nx
import argparse

random.seed(42)

# --- Configuration ---
NUM_MATERIALS = 10
NUM_LOCATIONS = 4
MAX_CHILDREN = 3
MAKE_PROB = 0.8
PURCHASE_PROB = 0.8
MOVE_PROB = 0.3
NUM_DEMANDS = 100
REQUEST_TIME_RANGE = 10

OUTDIR = 'data'
LOGDIR = os.path.join(OUTDIR, 'logs')
TOKENIZED = os.path.join(OUTDIR, 'samples')
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(TOKENIZED, exist_ok=True)

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=1)
args = parser.parse_args()

# --- Generate Static Data Once ---
static_flag = os.path.exists(f"{OUTDIR}/material.csv")
if not static_flag:
    # --- Generate Materials ---
    materials = pd.DataFrame({
        'id': range(NUM_MATERIALS),
        'description': [f"Material_{i}" for i in range(NUM_MATERIALS)]
    })
    materials.to_csv(f"{OUTDIR}/material.csv", index=False)

    # --- Generate Locations ---
    locations = pd.DataFrame({
        'id': range(NUM_LOCATIONS),
        'description': [f"Location_{i}" for i in range(NUM_LOCATIONS)]
    })
    locations.to_csv(f"{OUTDIR}/location.csv", index=False)

    # --- Generate BOM as DAG ---
    G = nx.DiGraph()
    G.add_nodes_from(range(NUM_MATERIALS))
    for parent in range(NUM_MATERIALS):
        num_children = random.randint(0, MAX_CHILDREN)
        children = random.sample(range(parent), min(num_children, parent))
        for child in children:
            G.add_edge(parent, child)
    bom_df = pd.DataFrame([(p, c) for p, c in G.edges()], columns=['parent', 'child'])
    bom_df['id'] = range(len(bom_df))
    bom_df.to_csv(f"{OUTDIR}/bom.csv", index=False)

    # --- Identify leaves and roots ---
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    leaves = [n for n, deg in out_degrees.items() if deg == 0]
    roots = [n for n, deg in in_degrees.items() if deg == 0]

    # --- Generate Methods ---
    methods = []
    method_id = 0
    for mat in range(NUM_MATERIALS):
        for loc in range(NUM_LOCATIONS):
            if mat in leaves: # and random.random() < PURCHASE_PROB:
                methods.append({
                    'id': method_id, 'type': 'purchase', 'material_id': mat,
                    'lead_time': random.randint(1, 5), 'location_id': loc,
                    'target_location_id': None, 'route_id': None, 'bom_id': None
                })
                method_id += 1
            if mat not in leaves: # and random.random() < MAKE_PROB:
                methods.append({
                    'id': method_id, 'type': 'make', 'material_id': mat,
                    'lead_time': random.randint(1, 5), 'location_id': loc,
                    'target_location_id': None, 'route_id': None, 'bom_id': None
                })
                method_id += 1
            if random.random() < MOVE_PROB:
                tgt_loc = random.choice([l for l in range(NUM_LOCATIONS) if l != loc])
                methods.append({
                    'id': method_id, 'type': 'move', 'material_id': mat,
                    'lead_time': random.randint(1, 5), 'location_id': loc,
                    'target_location_id': tgt_loc, 'route_id': None, 'bom_id': None
                })
                method_id += 1
    methods_df = pd.DataFrame(methods)
    methods_df.to_csv(f"{OUTDIR}/method.csv", index=False)

    # Cache leaves and roots for reuse
    with open(f"{OUTDIR}/roots.txt", "w") as f:
        f.write("\n".join(map(str, roots)))
    with open(f"{OUTDIR}/leaves.txt", "w") as f:
        f.write("\n".join(map(str, leaves)))
else:
    with open(f"{OUTDIR}/roots.txt") as f:
        roots = list(map(int, f.read().splitlines()))

# --- Generate N Dynamic Samples ---
for i in range(args.num_samples):
    demands = pd.DataFrame({
        'id': range(NUM_DEMANDS),
        'material_id': random.choices(roots, k=NUM_DEMANDS),
        'location_id': random.choices(range(NUM_LOCATIONS), k=NUM_DEMANDS),
        'quantity': random.choices(range(1000), k=NUM_DEMANDS), #random.randint(0, 1000),
        'request_time': random.choices(range(0, REQUEST_TIME_RANGE), k=NUM_DEMANDS),
        'commit_time': [None]*NUM_DEMANDS
    })
    #sample_dir = f"{TOKENIZED}/sample_{i}"
    sample_dir = f"{LOGDIR}/sample_{i}"
    #demand_path = f"{LOGDIR}/demand_{i}.csv"
    demand_path = f"{sample_dir}/demands.csv"
    os.makedirs(sample_dir, exist_ok=True)
    demands.to_csv(demand_path, index=False)
    os.system(f"python simulate_solver.py --input {demand_path} --output {sample_dir}")

print(f"✅ Generated {args.num_samples} samples using shared static dataset.")
