import pytest
import numpy as np
from src.nsga_pinn.problem import NsgaPinnProblem
from unittest.mock import MagicMock

def test_problem_initialization_and_bounds():
    current_weights = np.array([1.0, 2.0, 3.0])
    evaluator = MagicMock()

    problem = NsgaPinnProblem(evaluator, current_weights, bounds_radius=0.5)

    assert problem.n_var == 3
    assert problem.n_obj == 2

    # Check bounds
    np.testing.assert_allclose(problem.xl, [0.5, 1.5, 2.5])
    np.testing.assert_allclose(problem.xu, [1.5, 2.5, 3.5])

def test_problem_evaluation_delegation():
    current_weights = np.array([1.0, 1.0])
    evaluator = MagicMock()
    # Mock return
    evaluator.evaluate_population.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

    problem = NsgaPinnProblem(evaluator, current_weights)

    population = np.array([[1.1, 1.1], [0.9, 0.9]])
    out = {}

    problem._evaluate(population, out)

    # Verify delegation
    evaluator.evaluate_population.assert_called_once()
    args, kwargs = evaluator.evaluate_population.call_args
    np.testing.assert_array_equal(args[0], population)
    assert kwargs['mode'] == 'fitness'

    # Verify output
    np.testing.assert_array_equal(out["F"], [[0.1, 0.2], [0.3, 0.4]])
