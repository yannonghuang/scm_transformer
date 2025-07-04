def evaluate_plan_difference(aps, trans):
    from collections import defaultdict

    def plan_to_dict(plan):
        d = defaultdict(int)
        for action in plan:
            key = (action["type"], action["location"], action["material"], action["time"])
            d[key] += action["quantity"]
        return d

    aps_dict = plan_to_dict(aps)
    trans_dict = plan_to_dict(trans)

    all_keys = set(aps_dict) | set(trans_dict)
    diff = {
        "total_tokens": len(all_keys),
        "total_quantity_diff": 0,
        "avg_abs_delta": 0,
    }

    total_abs = 0
    for k in all_keys:
        aps_qty = aps_dict.get(k, 0)
        trans_qty = trans_dict.get(k, 0)
        total_abs += abs(aps_qty - trans_qty)

    diff["total_quantity_diff"] = sum(trans_dict.values()) - sum(aps_dict.values())
    diff["avg_abs_delta"] = total_abs / max(1, len(all_keys))
    return diff
