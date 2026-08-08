from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Candidate:
    composition: Tensor
    porosity: float
    score: float


@dataclass(frozen=True)
class SearchResult:
    candidates: tuple[Candidate, ...]
    generations: int
    elapsed_seconds: float


class CMAES:
    def __init__(
        self,
        dimension: int,
        population: int,
        initial_mean: Tensor,
        initial_sigma: float,
        seed: int,
    ) -> None:
        self.dimension = dimension
        self.population = population
        self.mean = initial_mean.clone()
        self.sigma = initial_sigma
        self.covariance = torch.eye(dimension, dtype=initial_mean.dtype)
        self.generator = torch.Generator(device=initial_mean.device).manual_seed(seed)
        self.parents = population // 2
        ranks = torch.arange(1, self.parents + 1, dtype=initial_mean.dtype)
        weights = torch.log(torch.tensor(self.parents + 0.5)) - torch.log(ranks)
        self.weights = weights / weights.sum()
        self.path_covariance = torch.zeros(dimension, dtype=initial_mean.dtype)
        self.path_sigma = torch.zeros(dimension, dtype=initial_mean.dtype)
        self.mu_effective = 1.0 / self.weights.square().sum()
        self.c_covariance = (4.0 + self.mu_effective / dimension) / (
            dimension + 4.0 + 2.0 * self.mu_effective / dimension
        )
        self.c_sigma = (self.mu_effective + 2.0) / (dimension + self.mu_effective + 5.0)
        self.damping = 1.0 + 2.0 * max(
            0.0, math.sqrt((self.mu_effective - 1.0) / (dimension + 1.0)) - 1.0
        )

    def ask(self) -> Tensor:
        eigenvalues, eigenvectors = torch.linalg.eigh(self.covariance)
        transform = eigenvectors @ torch.diag(eigenvalues.clamp_min(1.0e-10).sqrt())
        noise = torch.randn(
            self.population,
            self.dimension,
            generator=self.generator,
            dtype=self.mean.dtype,
            device=self.mean.device,
        )
        result: Tensor = self.mean + self.sigma * (noise @ transform.T)
        return result

    def tell(self, population: Tensor, scores: Tensor) -> None:
        order = torch.argsort(scores, descending=True)
        selected = population[order[: self.parents]]
        old_mean = self.mean.clone()
        self.mean = (selected * self.weights[:, None]).sum(0)
        displacement = (self.mean - old_mean) / self.sigma
        inverse_root = torch.linalg.inv(torch.linalg.cholesky(self.covariance))
        factor = math.sqrt(self.c_sigma * (2.0 - self.c_sigma) * float(self.mu_effective))
        self.path_sigma = (1.0 - self.c_sigma) * self.path_sigma + factor * (
            inverse_root @ displacement
        )
        covariance_factor = math.sqrt(
            self.c_covariance * (2.0 - self.c_covariance) * float(self.mu_effective)
        )
        self.path_covariance = (
            (1.0 - self.c_covariance) * self.path_covariance + covariance_factor * displacement
        )
        centered = (selected - old_mean) / self.sigma
        rank_mu = sum(
            weight * torch.outer(vector, vector)
            for weight, vector in zip(self.weights, centered, strict=True)
        )
        rank_one = torch.outer(self.path_covariance, self.path_covariance)
        self.covariance = (
            (1.0 - self.c_covariance) * self.covariance
            + self.c_covariance * 0.5 * rank_one
            + self.c_covariance * 0.5 * rank_mu
        )
        expected_norm = math.sqrt(self.dimension) * (
            1.0 - 1.0 / (4.0 * self.dimension) + 1.0 / (21.0 * self.dimension**2)
        )
        self.sigma *= math.exp(
            self.c_sigma
            / self.damping
            * (float(torch.linalg.vector_norm(self.path_sigma)) / expected_norm - 1.0)
        )

    def optimize(
        self,
        scorer: Callable[[Tensor], Tensor],
        budget_seconds: float,
        tolerance: float,
        patience: int,
        top_k: int = 3,
    ) -> SearchResult:
        start = time.perf_counter()
        best = -math.inf
        stale = 0
        generation = 0
        archive: list[Candidate] = []
        while time.perf_counter() - start < budget_seconds and stale < patience:
            population = self.ask()
            scores = scorer(population)
            self.tell(population, scores)
            generation += 1
            current = float(scores.max())
            stale = stale + 1 if current - best < tolerance else 0
            best = max(best, current)
            for vector, score in zip(population, scores, strict=True):
                composition = torch.softmax(vector[:-1], dim=0)
                porosity = float(torch.sigmoid(vector[-1]))
                archive.append(Candidate(composition, porosity, float(score)))
        archive.sort(key=lambda candidate: candidate.score, reverse=True)
        return SearchResult(tuple(archive[:top_k]), generation, time.perf_counter() - start)
