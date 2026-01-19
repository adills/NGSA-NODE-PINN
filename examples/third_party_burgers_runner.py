from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Tuple

import torch


def _default_third_party_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "third_party" / "pinn_burgers" / "00_burgers"


def _split_indices(
    n: int, val_frac: float, generator: torch.Generator
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n == 0:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty
    if val_frac <= 0.0:
        idx = torch.arange(n)
        return idx, idx
    perm = torch.randperm(n, generator=generator)
    n_val = int(n * val_frac)
    if n_val <= 0:
        return perm, perm
    val_idx = perm[:n_val]
    trn_idx = perm[n_val:]
    if trn_idx.numel() == 0:
        trn_idx = val_idx
    return trn_idx, val_idx


def _split_xyu(
    x: torch.Tensor,
    t: torch.Tensor,
    u: torch.Tensor,
    val_frac: float,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    trn_idx, val_idx = _split_indices(x.shape[0], val_frac, generator)
    return x[trn_idx], t[trn_idx], u[trn_idx], x[val_idx], t[val_idx], u[val_idx]


def _split_xy(
    x: torch.Tensor,
    t: torch.Tensor,
    val_frac: float,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    trn_idx, val_idx = _split_indices(x.shape[0], val_frac, generator)
    return x[trn_idx], t[trn_idx], x[val_idx], t[val_idx]


def run_third_party_burgers(
    input_data: torch.Tensor,
    target_data: torch.Tensor,
    input_phys: torch.Tensor,
    device: Optional[torch.device] = None,
    epochs: Optional[int] = None,
    batch: Optional[int] = None,
    tol: Optional[float] = None,
    f_mntr: Optional[int] = None,
    third_party_dir: Optional[Path] = None,
    val_frac: float = 0.2,
    seed: int = 0,
):
    third_party_dir = _default_third_party_dir() if third_party_dir is None else Path(third_party_dir)
    if not third_party_dir.exists():
        raise FileNotFoundError(f"third_party_dir not found: {third_party_dir}")

    sys.path.insert(0, str(third_party_dir))
    try:
        from pinn import PINN
        from parameters import params as pb_params
    finally:
        sys.path.pop(0)

    (
        f_in, f_out, width, depth,
        w_init, b_init, act,
        lr, opt,
        f_scl, nu,
        w_ini, w_bnd, w_pde, bc,
        f_mntr, r_seed, d_type, device_tp,
        n_epch, n_btch, c_tol
    ) = pb_params()

    if epochs is not None:
        n_epch = epochs
    if batch is not None:
        n_btch = batch
    if tol is not None:
        c_tol = tol
    if device is not None:
        device_tp = device
    if f_mntr is not None:
        f_mntr = int(f_mntr)
    elif n_epch < 20:
        f_mntr = 1

    data = input_data.detach().cpu()
    targets = target_data.detach().cpu()
    phys = input_phys.detach().cpu()

    t_data = data[:, 0].reshape(-1)
    x_data = data[:, 1].reshape(-1)
    u_data = targets.reshape(-1)

    is_ic = torch.isclose(t_data, torch.zeros_like(t_data), atol=1e-6)
    if not torch.any(is_ic) or torch.all(is_ic):
        raise ValueError("input_data must include both IC (t=0) and BC (t>0) points")

    x_ini = x_data[is_ic]
    t_ini = t_data[is_ic]
    u_ini = u_data[is_ic]

    x_bnd = x_data[~is_ic]
    t_bnd = t_data[~is_ic]
    u_bnd = u_data[~is_ic]

    t_pde = phys[:, 0].reshape(-1)
    x_pde = phys[:, 1].reshape(-1)

    generator = torch.Generator().manual_seed(seed)

    x_ini_trn, t_ini_trn, u_ini_trn, x_ini_val, t_ini_val, u_ini_val = _split_xyu(
        x_ini, t_ini, u_ini, val_frac, generator
    )
    x_bnd_trn, t_bnd_trn, u_bnd_trn, x_bnd_val, t_bnd_val, u_bnd_val = _split_xyu(
        x_bnd, t_bnd, u_bnd, val_frac, generator
    )
    x_pde_trn, t_pde_trn, x_pde_val, t_pde_val = _split_xy(
        x_pde, t_pde, val_frac, generator
    )

    pinn = PINN(
        x_ini_trn, t_ini_trn, u_ini_trn,
        x_bnd_trn, t_bnd_trn, u_bnd_trn,
        x_pde_trn, t_pde_trn,
        x_ini_val, t_ini_val, u_ini_val,
        x_bnd_val, t_bnd_val, u_bnd_val,
        x_pde_val, t_pde_val,
        f_in, f_out, width, depth,
        w_init, b_init, act,
        lr, opt,
        f_scl, nu,
        w_ini, w_bnd, w_pde, bc,
        f_mntr, r_seed, d_type, device_tp
    )
    pinn.to(device_tp)
    pinn.train(epoch=n_epch, batch=n_btch, tol=c_tol)
    return pinn


def evaluate_third_party(
    pinn,
    input_data: torch.Tensor,
    target_data: torch.Tensor,
    input_phys: torch.Tensor,
):
    device = pinn.device
    data = input_data.detach().to(device)
    targets = target_data.detach().to(device)
    phys = input_phys.detach().to(device)

    # Third-party PINN expects inputs in (x, t) order.
    xt_data = torch.stack([data[:, 1], data[:, 0]], dim=1)
    xt_phys = torch.stack([phys[:, 1], phys[:, 0]], dim=1)

    preds = pinn.forward(xt_data)
    data_loss = torch.mean((preds - targets) ** 2)
    phys_loss = pinn.loss_pde(xt_phys)
    return data_loss, phys_loss
