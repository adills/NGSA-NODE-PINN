import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, ANY

from src.nsga_pinn.orchestrator import HybridPinnOrchestrator

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.tensor([1.0]))
    def forward(self, x):
        return x * self.p

def test_orchestrator_flow():
    model = SimpleModel()
    interface = MagicMock()
    evaluator = MagicMock()
    problem_cls = MagicMock()
    selector = MagicMock()

    # Setup Mocks
    # Interface
    interface.to_genome.return_value = np.array([1.0])
    interface.genome_to_state_dict.return_value = {'p': torch.tensor([2.0])}

    # Evaluator
    evaluator.evaluate_module.return_value = torch.stack([torch.tensor(0.5, requires_grad=True), torch.tensor(0.5, requires_grad=True)])

    # Problem & NSGA
    # We can't easily mock minimize() as it is imported.
    # We should rely on integration or mock the minimize function in the module if needed.
    # But usually unit tests shouldn't run full NSGA2.
    # Let's verify the calls by mocking `minimize`.

    # Actually, orchestrator imports minimize.
    # We can patch it.

    with pytest.MonkeyPatch.context() as mp:
        mock_minimize = MagicMock()
        # minimize returns Result object
        mock_res = MagicMock()
        mock_res.X = np.array([[2.0]])
        mock_res.F = np.array([[0.1, 0.1]])
        mock_minimize.return_value = mock_res

        mp.setattr("src.nsga_pinn.orchestrator.minimize", mock_minimize)

        # Selector
        selector.select_knee_point.return_value = np.array([2.0])

        orch = HybridPinnOrchestrator(model, interface, evaluator, problem_cls, selector)

        # Run 1 epoch, 1 step each
        history = orch.train(epochs=1, adam_steps_per_epoch=1, nsga_gens_per_epoch=1)

        # Verify ADAM step
        # Optimizer should step
        # Evaluator should be called
        evaluator.evaluate_module.assert_called()

        # Verify NSGA step
        problem_cls.assert_called()
        mock_minimize.assert_called()

        # Verify Handoff
        selector.select_knee_point.assert_called()
        interface.genome_to_state_dict.assert_called_with(np.array([2.0]))

        # Verify weight update
        # We mocked genome_to_state_dict to return 2.0.
        # Check model param
        assert model.p.item() == 2.0

        assert len(history) == 1
