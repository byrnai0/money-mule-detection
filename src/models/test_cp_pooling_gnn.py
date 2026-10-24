
import torch
from cp_pooling_gnn import CPLayer, CPPoolingGNN


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Small synthetic graph: 20 nodes and 60 directed edges.
    x = torch.randn(20, 26, device=device)

    edge_index = torch.randint(
        0, 20, (2, 60), device=device
    )

    model = CPPoolingGNN(
        in_channels=26,
        hidden_channels=64,
        out_channels=2,
        rank=16,
        dropout=0.30,
    ).to(device)

    model.train()
    logits = model(x, edge_index)

    assert logits.shape == (20, 2), logits.shape
    assert torch.isfinite(logits).all(), "Non-finite logits"

    labels = torch.randint(0, 2, (20,), device=device)
    loss = torch.nn.functional.cross_entropy(
        logits, labels
    )

    loss.backward()

    assert model.cp1.W.grad is not None
    assert model.cp2.W.grad is not None
    assert torch.isfinite(model.cp1.W.grad).all()
    assert torch.isfinite(model.cp2.W.grad).all()

    print(f"Device: {device}")
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Loss: {loss.item():.6f}")
    print("PASS: forward pass, finite outputs, gradients")


if __name__ == "__main__":
    main()
