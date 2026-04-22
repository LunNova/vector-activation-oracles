"""Optimizers for steering vector training.

NaturalGradient: Per-parameter full Fisher preconditioning. For 1D parameters
    (steering vectors of shape (D,)), maintains a full DxD empirical Fisher
    matrix. The preconditioned gradient F^{-1}g captures cross-dimension
    curvature that diagonal optimizers (Adam) miss.
    Memory: O(D²) per parameter. For 36 layers × 4096 dim = ~2.3 GB.

SpectralScion: Stacks all steering vectors into a (N_layers, D) matrix and
    applies Newton-Schulz orthogonalization (spectral LMO) for cross-layer
    normalized updates. Inspired by Scion (Pethick et al., arXiv:2502.07529)
    and Muon (Keller Jordan). Very cheap — NS iterations on the (N, N) inner
    product. May struggle with highly non-square matrices (e.g. 36 × 4096).

AdEMAMix: Mixes a fast (β1) and slow (β3) EMA of gradients so old gradients
    keep contributing, along with the standard Adam second moment. Paper:
    Pagliardini et al., "The AdEMAMix Optimizer: Better, Faster, Older"
    (arXiv:2409.03137). Uses linear α warmup and log-half-life β3 warmup to
    avoid early-training instability from the high β3.
"""

import math

import torch
from torch import Tensor
from torch.optim import Optimizer


@torch.compile
def _zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """Newton-Schulz iteration for the zeroth power (orthogonalization) of G.

    Quintic iteration with coefficients maximizing slope at zero.
    From Keller Jordan (Muon) / modded-nanogpt.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class NaturalGradient(Optimizer):
    """Natural gradient with full empirical Fisher preconditioning.

    Args:
        params: Parameters to optimize (should be 1D vectors).
        lr: Learning rate.
        momentum: Momentum on preconditioned gradients (0 = no momentum).
        beta_fisher: EMA decay for Fisher matrix estimate.
        damping: Tikhonov damping (F + εI) for numerical stability.
        weight_decay: Decoupled weight decay (AdamW-style).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        beta_fisher: float = 0.99,
        damping: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            beta_fisher=beta_fisher,
            damping=damping,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            beta = group["beta_fisher"]
            damping = group["damping"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    D = p.shape[0]
                    state["fisher"] = torch.zeros(
                        D, D, device=p.device, dtype=torch.float32
                    )
                    if momentum > 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                state["step"] += 1
                g = p.grad.float()

                # Update Fisher EMA: F = β*F + (1-β) * g⊗gᵀ
                state["fisher"].mul_(beta).addr_(g, g, alpha=1 - beta)

                # Bias-corrected damped Fisher, then solve F x = g
                bc = 1 - beta ** state["step"]
                F = state["fisher"] / bc
                F.diagonal().add_(damping)
                nat_grad = torch.linalg.solve(F, g).to(p.dtype)

                # Momentum on preconditioned gradient
                if momentum > 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(nat_grad)
                    nat_grad = buf

                # Decoupled weight decay
                if wd > 0:
                    p.mul_(1 - lr * wd)

                p.add_(nat_grad, alpha=-lr)

        return loss


class SpectralScion(Optimizer):
    """Scion-style optimizer using spectral LMO on stacked steering vectors.

    Gathers all parameters into a single (N, D) matrix each step, applies
    momentum, then Newton-Schulz orthogonalization as the LMO direction.
    Scatters the result back to individual parameters.

    This gives cross-layer normalized updates — layers with larger gradients
    don't dominate. May not work well when N << D (e.g. 36 << 4096) since
    Newton-Schulz orthogonalizes in the smaller dimension.

    Args:
        params: Parameters to optimize (should be 1D vectors of same size).
        lr: Learning rate (step size for the LMO direction).
        rho: Scaling factor for the update magnitude.
        alpha: Momentum decay (G ← (1-α)G + grad). Higher = less momentum.
        weight_decay: Decoupled weight decay (AdamW-style).
        newton_steps: Number of Newton-Schulz iterations (5 is standard).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        rho: float = 1.0,
        alpha: float = 0.1,
        weight_decay: float = 0.0,
        newton_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            rho=rho,
            alpha=alpha,
            weight_decay=weight_decay,
            newton_steps=newton_steps,
        )
        super().__init__(params, defaults)
        # Collect all params once for stacking order
        self._all_params: list[torch.nn.Parameter] = []
        for group in self.param_groups:
            self._all_params.extend(group["params"])
        if self._all_params:
            D = self._all_params[0].shape[0]
            assert all(p.shape == (D,) for p in self._all_params), (
                f"All params must be 1D vectors of same size, got shapes: {[p.shape for p in self._all_params]}"
            )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr = group["lr"]
        rho = group["rho"]
        alpha = group["alpha"]
        wd = group["weight_decay"]
        ns_steps = group["newton_steps"]

        params_with_grad = [p for p in self._all_params if p.grad is not None]
        if not params_with_grad:
            return loss

        # Gather gradients into (N, D) matrix
        grad_matrix = torch.stack([p.grad for p in params_with_grad])  # (N, D)

        # Momentum: maintain accumulated gradient in state
        # Use first param as the state key for the shared momentum buffer
        state = self.state[self._all_params[0]]
        if len(state) == 0:
            state["momentum"] = torch.zeros_like(grad_matrix)
        momentum = state["momentum"]

        # G ← (1-α)G + grad  (Scion-style EMA)
        momentum.mul_(1 - alpha).add_(grad_matrix)

        # Spectral LMO via Newton-Schulz
        direction = (
            _zeropower_via_newtonschulz5(
                momentum.unsqueeze(0),
                ns_steps,
            )
            .squeeze(0)
            .to(grad_matrix.dtype)
        )  # (N, D)

        # Scatter updates back to individual parameters
        for i, p in enumerate(params_with_grad):
            if wd > 0:
                p.mul_(1 - lr * wd)
            p.add_(direction[i], alpha=-lr * rho)

        return loss


class AdEMAMix(Optimizer):
    """AdEMAMix: Adam + a second, slower EMA of the gradient.

    Keeps two gradient EMAs — a fast one (β1 ≈ 0.9) and a slow one (β3 ≈
    0.9999) — and updates with `(m1_hat + α·m2) / (sqrt(v_hat) + eps)`. The
    slow EMA retains signal from older gradients that β1≈0.9 discards. Per
    the paper, m2 is *not* bias-corrected.

    Because β3 near 1 makes early steps unstable, α is linearly warmed up
    0 → alpha_final and β3 follows a log-half-life warmup β1 → β3_final
    over `t_warmup` steps. Set `t_warmup=None` to skip the warmup.

    Args:
        params: Parameters to optimize.
        lr: Learning rate.
        betas: (β1, β2, β3). Paper defaults (0.9, 0.999, 0.9999).
        alpha: Final mixing coefficient for the slow EMA (paper default 5.0).
        eps: Denominator epsilon.
        weight_decay: Decoupled weight decay (AdamW-style).
        t_warmup: Steps over which α and β3 ramp to their final values. The
            paper recommends setting this to the total number of training
            steps. None disables warmup (use α and β3 immediately).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        t_warmup: int | None = None,
    ):
        b1, b2, b3 = betas
        if not 0.0 <= b1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1): {b1}")
        if not 0.0 <= b2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1): {b2}")
        if not 0.0 <= b3 < 1.0:
            raise ValueError(f"beta3 must be in [0, 1): {b3}")
        defaults = dict(
            lr=lr,
            betas=betas,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            t_warmup=t_warmup,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _alpha_at(step: int, alpha_end: float, warmup: int | None) -> float:
        if warmup is None or step >= warmup:
            return alpha_end
        return step / warmup * alpha_end

    @staticmethod
    def _beta3_at(
        step: int, beta1: float, beta3_end: float, warmup: int | None
    ) -> float:
        # Log-half-life interpolation β1 → β3_end (paper's f/f⁻¹ scheme).
        if warmup is None or step >= warmup:
            return beta3_end

        def hl(b: float) -> float:
            return math.log(0.5) / math.log(b + 1e-8) - 1

        def hl_inv(t: float) -> float:
            return math.pow(0.5, 1.0 / (t + 1))

        frac = step / warmup
        return hl_inv(frac * (hl(beta3_end) - hl(beta1)) + hl(beta1))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2, beta3_end = group["betas"]
            alpha_end = group["alpha"]
            eps = group["eps"]
            wd = group["weight_decay"]
            warmup = group["t_warmup"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m1"] = torch.zeros_like(p)
                    state["m2"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                g = p.grad

                alpha = self._alpha_at(t, alpha_end, warmup)
                beta3 = self._beta3_at(t, beta1, beta3_end, warmup)

                m1, m2, v = state["m1"], state["m2"], state["v"]
                m1.mul_(beta1).add_(g, alpha=1 - beta1)
                m2.mul_(beta3).add_(g, alpha=1 - beta3)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                bc1 = 1 - beta1**t
                bc2 = 1 - beta2**t
                denom = (v / bc2).sqrt_().add_(eps)
                update = (m1 / bc1 + alpha * m2) / denom

                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)

        return loss
