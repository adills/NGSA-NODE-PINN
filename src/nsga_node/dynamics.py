import torch
import torch.nn as nn
from torch.func import functional_call, vmap

class NodeDynamicsWrapper(nn.Module):
    def __init__(self, model_template):
        """
        Wraps a PyTorch model to be used as dynamics function f(t, y, theta)
        compatible with torchode and population-based training.

        Args:
            model_template: The PyTorch model structure (nn.Module).
        """
        super().__init__()
        self.model_template = model_template

    def forward(self, t, y, batched_state_dict):
        """
        Forward pass for a batch of parameters (population).

        Args:
            t: Time scalar or tensor (handled by vmap if necessary, but usually scalar for ODE).
            y: State tensor of shape (Pop_Size, State_Dim).
            batched_state_dict: Dictionary where each value is (Pop_Size, ...).

        Returns:
            dy/dt: Tensor of shape (Pop_Size, State_Dim).
        """
        # We want to evaluate: model_i(t, y_i) for i in 0..Pop_Size
        # The model usually takes 'y' (state) and optionally 't'.
        # For autonomous systems, it's just model(y).
        # For non-autonomous, model(torch.cat([y, t])) or similar.

        # Assumption: The model_template's forward method takes 'y' (and maybe 't' inside if it wraps it).
        # But usually NN ODEs take 't' and 'y'.
        # Standard signature for ODE func is f(t, y).
        # The user task says: "Create dynamics function f(t, y, theta)".

        # We define a helper that evaluates a SINGLE model instance
        def single_model_call(params, state_y):
            # params: unbatched state_dict
            # state_y: (State_Dim,)

            # Functional call to the model
            # We assume the model's forward accepts 't'? Or just 'y'?
            # Most simple NODEs are autonomous: dy/dt = NN(y).
            # If explicit time dependence is needed, usually t is appended to y.
            # However, torchode/torchdiffeq pass t.
            # Let's assume the underlying model expects (t, y) OR just (y) if we handle t.
            # But functional_call replaces 'self.model_template'.

            # If self.model_template.forward(t, y) exists:
            # return functional_call(self.model_template, params, (t, state_y))

            # If self.model_template(y):
            # return functional_call(self.model_template, params, (state_y,))

            # To be safe and generic, let's try to pass t if the model accepts it,
            # or handle the case where the user provided model is just NN(x).
            # The Example verification task uses a model that takes 't' as input?
            # "Physics Model (Incomplete): x_tt + 2.0 x = u_NN(x_t)" -> u_NN takes velocity.
            # Wait, the verification task model:
            # class Net(nn.Module): forward(self, t): return self.net(t) -> This was the baseline solver.
            # For NODE: dy/dt = f(t, y).
            # If the user provides a NN, say for the oscillator:
            # y = [x, v]. dy/dt = [v, -2x - NN(v)].
            # The NN takes part of state.

            # The wrapper should probably just call the model with whatever arguments it expects?
            # But `forward` here receives `t` and `y` from torchode.
            # And the model template is likely the NN part $u_\theta$.
            # BUT, NodeDynamicsWrapper usually IS the full ODE function.
            # If NodeDynamicsWrapper IS the physics function, then it contains the physics + NN.

            # Task 2.2 says: "Create class NodeDynamicsWrapper(torch.nn.Module)".
            # It seems this wrapper IS the system dynamics definition.
            # But wait, "Implement forward(self, t, y, batched_state_dict)".
            # And it uses `functional_call`.
            # This implies `NodeDynamicsWrapper` wraps the *learnable* part (the NN) and potentially defines the physics around it?
            # OR, `NodeDynamicsWrapper` IS the learnable model that outputs dy/dt directly?
            # "Proxy for physics": The task mentions "Physics Model (Incomplete): ... = u_NN(...)".

            # If I look at Task 2.3:
            # "evaluate_population... mode='fitness'... term = torchode.ODETerm(self.dynamics, with_args=True)".
            # `self.dynamics` is likely an instance of `NodeDynamicsWrapper`.

            # If `NodeDynamicsWrapper` is generic, it should just execute the NN.
            # But a generic NN doesn't know about "Physics".
            # The Physics logic (e.g. Oscillator equations) must reside somewhere.
            # In PINN, we pass `physics_residual_fn`.
            # In NODE, the "Physics" is usually part of the dynamics function structure.

            # If `NodeDynamicsWrapper` is generic, it can't encode the specific oscillator physics.
            # So `NodeDynamicsWrapper` might just be a mechanism to apply `functional_call` to a user-provided `dynamics_fn`?
            # But the instructions say: "Implement NodeDynamicsWrapper... forward... Use torch.func.functional_call".

            # Interpretation:
            # The user (me, in the example) will define a class inheriting from `NodeDynamicsWrapper`?
            # OR `NodeDynamicsWrapper` takes the `model_template` AND a `physics_fn`?
            # OR `NodeDynamicsWrapper` assumes the `model_template` IS the full dynamics $f(t, y)$.

            # If the latter (Model IS dynamics), then for the Oscillator, the Model must output [v, acc].
            # And inside the Model's forward, it calls the sub-network.
            # That seems the most logical for a "Wrapper" that just handles Batching.

            # So:
            # User defines:
            # class MyODE(nn.Module):
            #    def __init__(self): self.net = ...
            #    def forward(self, t, y):
            #       # physics logic
            #       return dy_dt

            # Then we wrap it: wrapper = NodeDynamicsWrapper(MyODE())
            # wrapper.forward(t, y, params) calls functional_call(MyODE, params, (t, y)).

            # This works.

            return functional_call(self.model_template, params, (t, state_y))

        # vmap logic:
        # t: None (broadcasted scalar) or batched? Usually scalar in torchode steps.
        # params: 0 (batched)
        # y: 0 (batched state)

        # Note: If t is a tensor of shape (1,), vmap might complain if we don't handle it.
        # But usually t is a scalar float or 0-d tensor in solvers.
        # If t is batched (e.g. different times for different samples?), we'd put 0.
        # Standard ODE solve: all particles at same time t.

        return vmap(single_model_call, in_dims=(0, 0))(batched_state_dict, y)
