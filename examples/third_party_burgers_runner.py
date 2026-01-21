from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Tuple
import time
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
    verbose: bool = True,
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
        f_mntr0, r_seed, d_type, device_tp,
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
    else:
        f_mntr = f_mntr0

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
    start_time = time.time()
    if verbose:
        pinn.train(epoch=n_epch, batch=n_btch, tol=c_tol)
    else:
        _train_with_pbar(pinn, epoch=n_epch, tol=c_tol)
    end_time = time.time()
    train_time = end_time - start_time
    return pinn, train_time


def _train_with_pbar(pinn, epoch, tol):
    from tqdm import tqdm

    pbar = tqdm(range(epoch), desc="Baseline PINN", unit="Epoch")
    t0 = time.time()
    for ep in pbar:
        loss_trn = pinn.loss_glb(
            pinn.x_ini_trn, pinn.t_ini_trn, pinn.u_ini_trn,
            pinn.x_bnd_trn, pinn.t_bnd_trn, pinn.u_bnd_trn,
            pinn.x_pde_trn, pinn.t_pde_trn
        )
        loss_val = pinn.loss_glb(
            pinn.x_ini_val, pinn.t_ini_val, pinn.u_ini_val,
            pinn.x_bnd_val, pinn.t_bnd_val, pinn.u_bnd_val,
            pinn.x_pde_val, pinn.t_pde_val
        )
        pinn.optimzer.zero_grad()
        loss_trn.backward()
        pinn.optimzer.step()
        ep_loss_trn = loss_trn.item()
        ep_loss_val = loss_val.item()
        pinn.loss_trn_log.append(ep_loss_trn)
        pinn.loss_val_log.append(ep_loss_val)

        elps = time.time() - t0
        pbar.set_postfix({
            "loss_trn": f"{ep_loss_trn:.3f}",
            "loss_val": f"{ep_loss_val:.3f}",
            "elps": f"{elps:.3f}",
        })
        t0 = time.time()
        if ep_loss_val < tol:
            break
    pbar.close()


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
