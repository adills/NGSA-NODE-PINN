import numpy as np

class ParetoSelector:
    """
    Selection logic for picking the best candidate from the Pareto Front.
    """

    def select_best_data(self, front: np.ndarray, F: np.ndarray) -> np.ndarray:
        """Returns individual with minimum Data Loss (Obj 0)."""
        idx = np.argmin(F[:, 0])
        return front[idx]

    def select_best_physics(self, front: np.ndarray, F: np.ndarray) -> np.ndarray:
        """Returns individual with minimum Physics Loss (Obj 1)."""
        idx = np.argmin(F[:, 1])
        return front[idx]

    def select_hybrid(self, front: np.ndarray, F: np.ndarray, alpha=0.5) -> np.ndarray:
        """Returns individual minimizing alpha*Data + (1-alpha)*Physics."""
        # Note: This does simple scalarization without normalization, assuming user handles alpha scaling.
        # Often it's better to normalize, but task implies simple formula.
        score = alpha * F[:, 0] + (1 - alpha) * F[:, 1]
        idx = np.argmin(score)
        return front[idx]

    def select_knee_point(self, front: np.ndarray, F: np.ndarray) -> np.ndarray:
        """
        Selects the 'Knee' or 'Elbow' point using the Max Distance to Chord method.
        Normalizes objectives to [0,1] first to ensure fair distance calculation.
        """
        if len(F) == 0:
            raise ValueError("Empty front passed to selector.")
        if len(F) == 1:
            return front[0]

        # 1. Normalize objectives
        min_vals = np.min(F, axis=0)
        max_vals = np.max(F, axis=0)

        diff = max_vals - min_vals
        # Handle case where all points have same value for an objective
        diff[diff == 0] = 1.0

        F_norm = (F - min_vals) / diff

        # 2. Sort by first objective (Data Loss) to define the curve
        # Usually trade-off: Low Data -> High Phys. High Data -> Low Phys.
        indices = np.argsort(F_norm[:, 0])
        F_sorted = F_norm[indices]
        front_sorted = front[indices]

        # 3. Define Chord (Line between extremes)
        start_pt = F_sorted[0]   # Min Data
        end_pt = F_sorted[-1]    # Min Physics (usually, or Max Data)

        # If extremes are essentially the same point
        if np.allclose(start_pt, end_pt):
            return front_sorted[0]

        # 4. Calculate perpendicular distance from each point to the line
        # Line P1->P2. Point P0.
        # d = |cross_product((P2-P1), (P1-P0))| / |P2-P1|
        # In 2D: |(x2-x1)(y1-y0) - (x1-x0)(y2-y1)| / dist

        x1, y1 = start_pt
        x2, y2 = end_pt

        # P0 = (x0, y0)
        x0 = F_sorted[:, 0]
        y0 = F_sorted[:, 1]

        # Numerator: |(y2-y1)x0 - (x2-x1)y0 + x2y1 - y2x1|
        # Wait, standard point-line dist form: |Ax0 + By0 + C| / sqrt(A^2+B^2)
        # Line: (y1 - y2)x + (x2 - x1)y + x1y2 - x2y1 = 0
        A = y1 - y2
        B = x2 - x1
        C = x1*y2 - x2*y1

        numer = np.abs(A*x0 + B*y0 + C)
        denom = np.sqrt(A**2 + B**2)

        dists = numer / denom

        best_idx_local = np.argmax(dists)
        return front_sorted[best_idx_local]
