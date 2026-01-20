from pymoo.core.problem import Problem
import numpy as np

class NsgaPinnProblem(Problem):
    def __init__(self, evaluator, current_weights, bounds_radius=1.0):
        """
        Pymoo Problem definition for NSGA-PINN.
        Args:
            evaluator: VectorizedPinnEvaluator instance.
            current_weights: (D,) numpy array of current ADAM weights.
            bounds_radius: Float, defines box constraints [w-r, w+r].
        """
        self.evaluator = evaluator
        self.current_weights = current_weights

        n_var = len(current_weights)
        xl = current_weights - bounds_radius
        xu = current_weights + bounds_radius

        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        # x is (Pop_Size, n_var)
        # Evaluate using fitness mode (vmap)
        losses = self.evaluator.evaluate_population(x, mode='fitness')
        out["F"] = losses
