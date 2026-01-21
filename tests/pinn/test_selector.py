import pytest
import numpy as np
from src.nsga_neuro_evolution_core.selector import ParetoSelector

def test_best_data_and_physics():
    selector = ParetoSelector()

    # Front: 3 points
    # P1: Data=0.1, Phys=1.0
    # P2: Data=0.5, Phys=0.5
    # P3: Data=1.0, Phys=0.1
    front = np.array([[1,1], [2,2], [3,3]]) # Dummy genomes
    F = np.array([[0.1, 1.0], [0.5, 0.5], [1.0, 0.1]])

    best_data = selector.select_best_data(front, F)
    np.testing.assert_array_equal(best_data, front[0])

    best_phys = selector.select_best_physics(front, F)
    np.testing.assert_array_equal(best_phys, front[2])

def test_knee_point_selection_convex():
    selector = ParetoSelector()

    # Synthetic convex front (Quarter circle-ish)
    # A=(0, 1), B=(1, 0)
    # Knee ~ (0.29, 0.29) approx distance?
    # Let's use simple points: (0, 10), (1, 1), (10, 0)
    # Normalized: (0, 1), (0.1, 0.1), (1, 0)
    # Knee is definitely middle.

    front = np.array([[0], [1], [2]])
    F = np.array([[0.0, 10.0], [1.0, 1.0], [10.0, 0.0]])

    knee = selector.select_knee_point(front, F)
    np.testing.assert_array_equal(knee, front[1])

def test_knee_point_selection_flat():
    selector = ParetoSelector()
    # Linear front: (0,1), (0.5, 0.5), (1,0)
    # Distance to chord is 0 for all.
    # It should pick one? argmax might pick first.
    # But usually numerical noise makes one stick out.
    # If perfectly linear, middle one is on line.
    # Let's add slight bump. (0.5, 0.4) is "better" (below line).
    # But "Distance to Chord" usually assumes *convex* front bulging towards origin?
    # No, usually we look for point furthest from the worst line (Nadir-Ideal) or similar.
    # The "Knee" in Pareto usually means the bulge towards the origin.
    # The line connects extremes.
    # Points "below" the line have distance.
    # My implementation uses abs(Ax+By+C).
    # It finds point *furthest* from line.
    # If front is convex (bulging to origin), the knee is furthest.

    front = np.array([[0], [1], [2]])
    F = np.array([[0.0, 1.0], [0.4, 0.4], [1.0, 0.0]]) # Middle is (0.4, 0.4)
    # Line (0,1) to (1,0). y = 1-x => x+y-1=0.
    # Mid: 0.4+0.4-1 = -0.2. Dist ~ 0.2.
    # Extremes: 0. Dist=0.
    # So Middle should be picked.

    knee = selector.select_knee_point(front, F)
    np.testing.assert_array_equal(knee, front[1])
