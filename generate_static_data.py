import os
import random
import pandas as pd
import networkx as nx
import argparse

from scm_transformer import get_token_type, config

random.seed(42)

# --- Configuration ---
NUM_MATERIALS = config['num_materials'] #10
NUM_LOCATIONS = config['num_locations']
MAX_CHILDREN = 3
MAKE_PROB = 0.8
PURCHASE_PROB = 0.8
MOVE_PROB = 0.3
NUM_DEMANDS = config['num_demands'] #100
REQUEST_TIME_RANGE = 10
MAX_QUANTITY = config['max_quantity']
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
                    'id': method_id, 'type': get_token_type('purchase'), 'material_id': mat,
                    'lead_time': random.randint(1, 5), 'location_id': loc,
                    'source_location_id': None, 'route_id': None, 'bom_id': None
                })
                method_id += 1
            if mat not in leaves: # and random.random() < MAKE_PROB:
                methods.append({
                    'id': method_id, 'type': get_token_type('make'), 'material_id': mat,
                    'lead_time': random.randint(1, 5), 'location_id': loc,
                    'source_location_id': None, 'route_id': None, 'bom_id': None
                })
                method_id += 1
            if random.random() < MOVE_PROB:
                src_loc = random.choice([l for l in range(NUM_LOCATIONS) if l != loc])
                methods.append({
                    'id': method_id, 'type': get_token_type('move'), 'material_id': mat,
                    'lead_time': random.randint(1, 5), 'location_id': loc,
                    'source_location_id': src_loc, 'route_id': None, 'bom_id': None
                })
                method_id += 1
    methods_df = pd.DataFrame(methods)
    methods_df.to_csv(f"{OUTDIR}/method.csv", index=False)

    # --- Compute BOM Depth for Each Material ---
    depths = {}
    for node in reversed(list(nx.topological_sort(G))):  # process from leaves up
        children = list(G.successors(node))
        if not children:
            depths[node] = 0
        else:
            child_depths = [depths[c] for c in children if c in depths]
            if len(child_depths) != len(children):
                raise ValueError(f"Child depth missing for node {node}, children: {children}")
            depths[node] = 1 + max(child_depths)
    # Save BOM depth info
    depth_df = pd.DataFrame([
        {'material_id': m, 'bom_depth': d} for m, d in depths.items()
    ])
    depth_df.to_csv(f"{OUTDIR}/bom_depth.csv", index=False)

    # Cache leaves and roots for reuse
    with open(f"{OUTDIR}/roots.txt", "w") as f:
        f.write("\n".join(map(str, roots)))
    with open(f"{OUTDIR}/leaves.txt", "w") as f:
        f.write("\n".join(map(str, leaves)))
else:
    with open(f"{OUTDIR}/roots.txt") as f:
        roots = list(map(int, f.read().splitlines()))

# --- Generate N Dynamic Samples ---
# Load BOM depth info
depth_df = pd.read_csv(f"{OUTDIR}/bom_depth.csv")
depth_map = dict(zip(depth_df['material_id'], depth_df['bom_depth']))
# Determine max BOM depth
max_depth = max(depth_map.values())
print(f"✅ Max depth: {max_depth}")
if max_depth > args.num_samples:
    raise ValueError(f"Number of samples should be no less than max_depth {max_depth}")
#for i in range(args.num_samples):
#samples_per_depth = max(1, args.num_samples // (max_depth + 1))
sample_count = 0
for target_depth in range(max_depth + 1):
    #if sample_count >= args.num_samples:
    #    break

    # Filter root materials for current depth
    eligible_roots = [m for m in roots if depth_map.get(m, 999) == target_depth]
    if not eligible_roots:
        continue

    #num_to_generate = min(samples_per_depth, args.num_samples - sample_count)

    #for _ in range(num_to_generate):
    for i in range(args.num_samples):
        demands = pd.DataFrame({
            'demand_id': range(NUM_DEMANDS),
            'material_id': random.choices(eligible_roots, k=NUM_DEMANDS),
            'location_id': random.choices(range(NUM_LOCATIONS), k=NUM_DEMANDS),
            'quantity': random.choices(range(MAX_QUANTITY), k=NUM_DEMANDS),
            'request_time': random.choices(range(0, REQUEST_TIME_RANGE), k=NUM_DEMANDS),
            'commit_time': [None]*NUM_DEMANDS
        })
        #sample_dir = f"{LOGDIR}/sample_{sample_count}/depth_{target_depth}"
        sample_dir = f"{LOGDIR}/depth_{target_depth}/sample_{i}"
        demand_path = f"{sample_dir}/demands.csv"
        os.makedirs(sample_dir, exist_ok=True)
        demands.to_csv(demand_path, index=False)
        os.system(f"python simulate_solver.py --input {demand_path} --output {sample_dir}")
        #sample_count += 1

print(f"✅ Generated {args.num_samples} samples using shared static dataset.")

