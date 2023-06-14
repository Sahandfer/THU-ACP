import json
import networkx as nx
import matplotlib.pyplot as plt

with open("results.json", "r") as f:
    data = json.load(f)
    print(f"We have {len(data)} frequent subgraphs!")

for j, t in enumerate(data):
    graph = nx.DiGraph()
    vertices = t["nodes"]
    edges = t["edges"]
    vs = {}
    t = False
    for i,e in enumerate(edges):
        graph.add_edge(e["source_node_id"], e["target_node_id"])
        graph.add_node(e["source_node_id"])
        graph.add_node(e["target_node_id"])
    
    fsize = (5, 5)
    plt.figure(figsize=fsize)
    pos = nx.spectral_layout(graph)
    nx.draw_networkx(graph, pos, arrows=True)
    nx.draw_networkx_edge_labels(graph,pos)
    plt.savefig(f"figures/{j}.png")