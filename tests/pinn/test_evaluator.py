import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.func import functional_call, grad, vmap
from src.nsga_pinn.evaluator import VectorizedPinnEvaluator
from src.nsga_neuro_evolution_core.interface import PytorchGenomeInterface

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1) # y = w*x + b

    def forward(self, x):
        return self.fc(x)

def physics_residual_fn(model_template, params, x):
    # Physics: dy/dx = 2.0
    # Residual = (dy/dx - 2.0)^2

    # We need per-sample gradient.
    # params is a single instance (dict of tensors) here (inside vmap).

    def model_fn(x_in):
        return functional_call(model_template, params, (x_in,))

    def single_grad(x_s):
        # x_s is (1,) -> model -> (1,). Sum for scalar.
        return grad(lambda z: model_fn(z).sum())(x_s)

    # vmap over x samples
    grads = vmap(single_grad)(x) # (N, 1)

    residual = (grads - 2.0) ** 2
    return residual.mean()

def test_evaluator_fitness_mode():
    model = SimpleModel()
    interface = PytorchGenomeInterface(model)

    # Data: y = 2x.
    input_data = torch.tensor([[0.0], [1.0], [2.0]])
    target_data = torch.tensor([[0.0], [2.0], [4.0]])

    # Physics points
    input_physics = torch.tensor([[0.5], [1.5]])
    input_physics.requires_grad_(True)

    evaluator = VectorizedPinnEvaluator(
        model, interface, input_data, target_data, input_physics, physics_residual_fn
    )

    # Population of size 2
    # Individual 1: w=2, b=0 (Perfect)
    # Individual 2: w=1, b=0 (Bad)

    pop_size = 2
    genome_size = interface.total_params
    population = np.zeros((pop_size, genome_size))

    # Set weights manually
    # w is index 0, b is index 1 (approx, depends on iteration order)
    # fc.weight (1,1), fc.bias (1,)
    # interface flattens in order of named_parameters

    # Indiv 1
    # Weight=2, Bias=0
    w_idx = 0
    b_idx = 1
    population[0, w_idx] = 2.0
    population[0, b_idx] = 0.0

    # Indiv 2
    # Weight=1, Bias=0
    population[1, w_idx] = 1.0
    population[1, b_idx] = 0.0

    results = evaluator.evaluate_population(population, mode='fitness')

    assert results.shape == (2, 2)

    # Indiv 1: DataLoss=0, PhysLoss=0 ((2-2)^2)
    assert np.allclose(results[0], [0.0, 0.0], atol=1e-5)

    # Indiv 2: DataLoss: pred=[0,1,2], target=[0,2,4]. diff=[0,-1,-2]. sq=[0,1,4]. mean=5/3=1.666
    # PhysLoss: grad=1. (1-2)^2 = 1.
    assert np.allclose(results[1], [5.0/3.0, 1.0], atol=1e-5)

def test_evaluator_gradient_mode():
    model = SimpleModel()
    interface = PytorchGenomeInterface(model)

    input_data = torch.randn(5, 1)
    target_data = torch.randn(5, 1)
    input_physics = torch.randn(5, 1)
    input_physics.requires_grad_(True)

    evaluator = VectorizedPinnEvaluator(
        model, interface, input_data, target_data, input_physics, physics_residual_fn
    )

    # Single individual
    population = np.random.randn(1, interface.total_params)

    results = evaluator.evaluate_population(population, mode='gradient')

    # Check shape
    assert results.shape == (1, 2)

    # Check graph
    assert results.requires_grad
    assert results.grad_fn is not None
