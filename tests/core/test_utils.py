import pytest
import torch
import torch.nn as nn
from src.nsga_neuro_evolution_core.utils import nsga_evaluation_context, adam_update_context

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, 1))

    def forward(self, x):
        return x * self.weight

def test_nsga_context_behavior():
    model = SimpleModel()
    x = torch.tensor([2.0], requires_grad=False)

    # Pre-check
    assert model.weight.requires_grad
    assert not x.requires_grad

    with nsga_evaluation_context(model, x):
        # Check inside context
        assert not model.weight.requires_grad
        assert x.requires_grad

        # Forward pass
        y = model(x)

        # Check gradients
        # Grad w.r.t input should work
        grad_x = torch.autograd.grad(y, x, create_graph=True)[0]
        assert grad_x is not None

        # Grad w.r.t weight should fail (or return None if allow_unused=True, but we expect error here usually)
        with pytest.raises(RuntimeError):
            torch.autograd.grad(y, model.weight, create_graph=True)

    # Post-check: restore state
    assert model.weight.requires_grad
    assert not x.requires_grad

def test_adam_context():
    with adam_update_context():
        assert torch.is_grad_enabled()
