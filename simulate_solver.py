import argparse
import os
import pandas as pd
from collections import defaultdict

from scm_transformer import get_token_type

# --- Loaders ---
def load_csv(path):
    return pd.read_csv(path)

def load_materials(path):
    return load_csv(path).set_index('id').to_dict(orient='index')

def load_locations(path):
    return load_csv(path).set_index('id').to_dict(orient='index')

def load_methods(path):
    methods = defaultdict(list)
    for _, row in load_csv(path).iterrows():
        key = (row['material_id'], row['location_id'])
        methods[key].append(row)
    return methods

def load_bom(path):
    bom = defaultdict(list)
    for _, row in load_csv(path).iterrows():
        bom[row['parent']].append(row['child'])
    return bom

def load_demands(path):
    return load_csv(path).to_dict(orient='records')

# --- Work Order Creation ---
def create_work_order(method, start_time, end_time, quantity):
    return {
        'type': method['type'],
        'material_id': method['material_id'],
        'location_id': method['location_id'],
        'target_location_id': method.get('target_location_id', None),
        'route_id': method.get('route_id', None),
        'bom_id': method.get('bom_id', None),
        'start_time': start_time,
        'end_time': end_time,
        'quantity': quantity
    }

# --- Recursive Solver ---
def solve_demand(d, methods, bom, now):
    m_id = d['material_id']
    loc_id = d['location_id']
    req_time = d['request_time']
    quantity = d['quantity']

    key = (m_id, loc_id)
    if key not in methods:
        d['commit_time'] = None
        return d, []

    method = methods[key][0]  # pick first valid method
    child_wos = []
    commit_times = []

    for c in bom.get(m_id, []):
        c_demand = {
            'material_id': c,
            'location_id': method['location_id'],
            'request_time': max([req_time - method['lead_time'], now]), # req_time 
            'quantity': quantity
        }
        solved_c, c_wos = solve_demand(c_demand, methods, bom, now)
        child_wos += c_wos
        if solved_c['commit_time'] is not None:
            commit_times.append(solved_c['commit_time'])
        else:
            d['commit_time'] = None
            return d, []
        
    start_time = max([req_time - method['lead_time'], *commit_times, now])
    end_time = start_time + method['lead_time']

    wo = create_work_order(method, start_time, end_time, quantity)
    d['commit_time'] = end_time
    #return d, child_wos + [wo]
    return d, [wo] + child_wos

# --- CLI Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input demand.csv')
    parser.add_argument('--output', required=True, help='Directory to save outputs')
    parser.add_argument('--data_dir', default='data', help='Directory where static files are stored')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    materials = load_materials(os.path.join(args.data_dir, 'material.csv'))
    locations = load_locations(os.path.join(args.data_dir, 'location.csv'))
    methods = load_methods(os.path.join(args.data_dir, 'method.csv'))
    bom = load_bom(os.path.join(args.data_dir, 'bom.csv'))
    demands = load_demands(args.input)

    rows = []
    now = 0

    for d in demands:
        solved_d, wos = solve_demand(d, methods, bom, now)
        rows.append({'type': get_token_type('demand'), **solved_d})
        for wo in wos:
            rows.append({'type': 'workorder', **wo})

    pd.DataFrame(rows).to_csv(os.path.join(args.output, 'combined_output.csv'), index=False)
    print("✅ Simulation complete. Outputs saved to:", args.output)

if __name__ == '__main__':
    main()
