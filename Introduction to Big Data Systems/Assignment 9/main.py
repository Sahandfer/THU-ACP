import csv
import json
from tqdm.auto import tqdm


def read_graph(file_name):
    with open(file_name) as input_file:
        rows = csv.reader(input_file, delimiter=",")
        graph = {}
        num_edges = 0
        # Skip the header
        next(rows, None)
        for row in rows:
            # Read row values
            src, dst, amt, strategy, bus, src_name, dst_name, node_type = row
            # Calculate destination id based on node type (card or account)
            src_v = int(src)
            dst_v = (1 - int(node_type)) * 800000 + int(dst)
            # Create edge tuple
            e = (
                src_v,
                dst_v,
                src_name,
                dst_name,
                f"{amt}-{strategy}-{bus}-{int(node_type)}",
            )

            # Save in and out edges
            graph[src_v] = graph.get(src_v, []) + [e]
            graph[dst_v] = graph.get(dst_v, []) + [e]

            num_edges += 1

    print(f"Graph loaded - # Vertices: {len(graph)} - # Edges: {num_edges}")

    return graph


def make_edge_embedding(e):
    return "-".join(str(item) for item in e[2:])


def make_pattern(edges):
    return "_".join(sorted([make_edge_embedding(e) for e in edges]))


def make_json(patterns):
    res = []
    for v in patterns.values():
        if v["frequency"] >= 10000:
            obj = {"frequency": v["frequency"], "nodes": [], "edges": []}
            for i, e in enumerate(v["orig"]):
                obj["nodes"].append({"node_id": i, "name": e[2]})
                amt, strategy, bus, node_type = e[4].split("-")
                obj["edges"].append(
                    {
                        "source_node_id": e[0],
                        "target_node_id": e[1] - 800000 * (1 - int(node_type)),
                        "amt": int(float(amt)),
                        "strategy_name": strategy,
                        "buscode": bus,
                        "type": "account-to-account"
                        if int(node_type)
                        else "account-to-card",
                    }
                )
                if i == 2:
                    obj["nodes"].append({"node_id": i + 1, "name": e[3]})

            res.append(obj)

    with open("results.json", "w") as outfile:
        outfile.write(json.dumps(res, indent=2))


def find_frequent_subgraphs(file_name):
    graph = read_graph(file_name)
    patterns = {}
    used_subs = {}
    for i in tqdm(graph.keys()):
        adj_e_i = graph[i]
        for e_i in adj_e_i:
            j = e_i[1] if e_i[0] == i else e_i[0]
            adj_e_j = graph.get(j, [])
            for e_j in adj_e_j:
                if e_j != e_i:
                    k = e_j[1] if e_j[0] == j else e_j[0]
                    adj_e_k = graph.get(k, [])
                    for e_k in adj_e_k:
                        if e_k not in [e_i, e_j]:
                            es = [e_i, e_j, e_k]
                            cur_sub = "-".join(sorted([f"{e[0]}_{e[1]}" for e in es]))
                            # Checking if the subgraph has already been used.
                            if not used_subs.get(cur_sub, 0):
                                used_subs[cur_sub] = 1
                                p_key = make_pattern(es)
                                if not patterns.get(p_key, 0):
                                    p_val = {"frequency": 1, "orig": es}
                                    patterns[p_key] = p_val
                                else:
                                    patterns[p_key]["frequency"] += 1

    make_json(patterns)


if __name__ == "__main__":
    find_frequent_subgraphs("data/full.csv")
