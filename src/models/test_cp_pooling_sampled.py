
from pathlib import Path
import torch
from torch_geometric.loader import NeighborLoader

from cp_pooling_gnn import CPPoolingGNN


def main():
    root = Path(__file__).resolve().parents[2]
    graph_path = root / "data" / "graphs" / "training_graph.pt"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print("Loading graph...")

    payload = torch.load(
        graph_path, map_location="cpu", weights_only=False
    )
    data = payload["data"]

    loader = NeighborLoader(
        data,
        input_nodes=data.train_mask,
        num_neighbors=[15, 10],
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))
    batch = batch.to(device)

    print(f"Sampled nodes: {batch.num_nodes}")
    print(f"Sampled edges: {batch.edge_index.size(1)}")
    print(f"Seed nodes: {batch.batch_size}")

    model = CPPoolingGNN(
        in_channels=batch.x.size(1),
        hidden_channels=64,
        out_channels=2,
        rank=16,
        dropout=0.30,
    ).to(device)

    model.train()
    model.zero_grad(set_to_none=True)

    logits = model(batch.x, batch.edge_index)
    seed_logits = logits[:batch.batch_size]
    seed_labels = batch.y[:batch.batch_size]

    assert seed_logits.shape == (batch.batch_size, 2)
    assert torch.isfinite(seed_logits).all()

    loss = torch.nn.functional.cross_entropy(
        seed_logits, seed_labels
    )
    loss.backward()

    assert model.cp1.W.grad is not None
    assert model.cp2.W.grad is not None
    assert torch.isfinite(model.cp1.W.grad).all()
    assert torch.isfinite(model.cp2.W.grad).all()

    print(f"Seed logits shape: {tuple(seed_logits.shape)}")
    print(f"Loss: {loss.item():.6f}")
    print("PASS: sampled-batch forward and backward")


if __name__ == "__main__":
    main()
