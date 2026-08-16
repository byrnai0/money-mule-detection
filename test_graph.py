import torch
print("Graph loaded successfully")
path = "data/graphs/ibm_aml_graph.pt"

payload = torch.load(
    path,
    map_location="cpu",
    weights_only=False,
)

graph = payload["graph"]
accounts = payload["account_list"]

print("Nodes:", graph.num_nodes)
print("Edges:", graph.edge_index.shape[1])
print("Laundering edges:", int(graph.edge_label.sum().item()))
print("Accounts:", len(accounts))
print("Edge index shape:", tuple(graph.edge_index.shape))