from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


class DeploymentAdapter(nn.Module):
    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(
        self,
        atomic_numbers: Tensor,
        positions: Tensor,
        edge_index: Tensor,
        edge_vectors: Tensor,
        edge_lengths: Tensor,
        graph_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        output = self.network(
            atomic_numbers,
            positions,
            edge_index,
            edge_vectors,
            edge_lengths,
            graph_index,
        )
        return output.properties, output.elastic_tensor


def export_onnx(
    adapter: nn.Module,
    inputs: tuple[Tensor, ...],
    destination: str | Path,
) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        adapter,
        inputs,
        path,
        input_names=[
            "atomic_numbers",
            "positions",
            "edge_index",
            "edge_vectors",
            "edge_lengths",
            "graph_index",
        ],
        output_names=["properties", "elastic_tensor"],
        dynamic_axes={"atomic_numbers": {0: "atoms"}, "edge_index": {1: "edges"}},
        opset_version=18,
    )
