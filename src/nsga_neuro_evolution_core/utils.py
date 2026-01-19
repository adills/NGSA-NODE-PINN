from contextlib import contextmanager
import torch
import torch.nn as nn

@contextmanager
def nsga_evaluation_context(model: nn.Module, inputs: torch.Tensor):
    """
    Context manager for NSGA evaluation (Fitness Mode).
    - Freezes model weights (no gradients w.r.t weights).
    - Enables gradients w.r.t inputs (for Physics residuals).
    - NOT strictly 'no_grad' because we need input gradients.
    """
    # Store original states
    param_states = {p: p.requires_grad for p in model.parameters()}
    input_state = inputs.requires_grad

    try:
        # Freeze weights
        model.requires_grad_(False)
        # Enable input gradients
        inputs.requires_grad_(True)
        yield
    finally:
        # Restore weights
        for p, original_state in param_states.items():
            p.requires_grad_(original_state)
        # Restore inputs
        inputs.requires_grad_(input_state)

@contextmanager
def adam_update_context():
    """
    Context manager for ADAM updates (Gradient Mode).
    Ensures autograd is enabled.
    """
    with torch.enable_grad():
        yield
