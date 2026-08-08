import torch

from compact_scaffold.search.cmaes import CMAES


def test_cmaes_improves_sphere() -> None:
    optimizer = CMAES(3, 8, torch.tensor([3.0, 3.0, 3.0]), 1.0, 4)
    initial = float(optimizer.mean.square().sum())
    for _ in range(20):
        population = optimizer.ask()
        optimizer.tell(population, -population.square().sum(-1))
    assert float(optimizer.mean.square().sum()) < initial

