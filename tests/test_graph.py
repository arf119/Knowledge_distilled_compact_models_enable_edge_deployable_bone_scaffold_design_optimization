import torch

from compact_scaffold.data.graph import build_radius_graph, periodic_displacements


def test_periodic_displacements_wrap_cell() -> None:
    positions = torch.tensor([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]])
    cell = torch.eye(3) * 10.0
    displacement, _ = periodic_displacements(positions, cell)
    assert torch.allclose(displacement[0, 1], torch.tensor([0.2, 0.0, 0.0]), atol=1.0e-5)


def test_radius_graph_is_bidirectional() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    edges, vectors, radii = build_radius_graph(positions, torch.eye(3) * 5.0, 2.0, 8)
    assert edges.shape == (2, 2)
    assert vectors.shape == (2, 3)
    assert torch.allclose(radii, torch.ones(2))

