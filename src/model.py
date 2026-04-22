"""Minimal Qwen3 transformer with built-in steering vectors and activation injection."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

VALID_INJECTION_POINTS = ("post_attn", "post_mlp")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Match HF: cast normalized back to input dtype, then multiply by weight
        # (weight stays in its param dtype). Doing the multiply in f32 then casting
        # produces different bf16 rounding.
        return self.weight * x.to(dtype)


def build_rope_cache(
    seq_len: int, head_dim: int, base: float, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    theta = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, theta)  # (seq_len, head_dim/2)
    freqs = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
    return freqs.cos().unsqueeze(0), freqs.sin().unsqueeze(0)  # (1, seq_len, head_dim)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, nh, T, hd), cos/sin: (1, T, hd) — pre-unsqueezed from build_rope_cache."""
    hd_half = x.size(-1) // 2
    x1, x2 = x[..., :hd_half], x[..., hd_half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(
            config.n_embd, config.n_head * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.n_embd, config.n_kv_head * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.n_embd, config.n_kv_head * config.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.n_head * config.head_dim, config.n_embd, bias=False
        )

        # Qwen3 applies RMSNorm to Q and K before RoPE
        self.q_norm = RMSNorm(config.head_dim, eps=config.norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, T, D) input embeddings
            cos, sin: (1, T, hd) RoPE for current positions
            attn_mask: Attention mask. Three formats accepted:
                - (B, 1, T_q, T_kv) bool: pre-built 4D mask (prefill, avoids per-block rebuild)
                - (B, T_kv) bool, True=attend: 2D mask (decode, or legacy prefill)
                - None: pure causal, no padding
            kv_cache: cached (k, v) each (B, n_kv_heads, S_prev, hd). None for prefill/training.
            return_kv: if True, return (output, (k, v)) for caching. False for training.
        """
        B, T_q, _ = x.shape

        q = self.q_proj(x).view(B, T_q, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T_q, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T_q, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # KV cache: concat new K,V with cached (before GQA expansion)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)  # (B, n_kv_heads, S, hd)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv = (k, v) if return_kv else None

        # GQA: expand kv heads to match query heads
        if self.n_kv_head != self.n_head:
            reps = self.n_head // self.n_kv_head
            k_exp = k.repeat_interleave(reps, dim=1)
            v_exp = v.repeat_interleave(reps, dim=1)
        else:
            k_exp, v_exp = k, v

        # Build attention mask — 3 paths:
        # 1. Pre-built 4D mask (caller already combined causal + padding)
        # 2. 2D mask + decode (T_q == 1): padding only
        # 3. 2D mask + prefill (T_q > 1): build causal + padding (legacy, used by generate decode)
        # 4. None: pure causal, no padding
        if attn_mask is not None and attn_mask.ndim == 4:
            bool_mask = attn_mask
        elif attn_mask is not None and T_q > 1:
            pad_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            causal = ~torch.triu(
                torch.ones(T_q, T_q, device=x.device, dtype=torch.bool), diagonal=1
            )
            bool_mask = pad_mask & causal  # (B, 1, T_q, T_q)
        elif attn_mask is not None:
            # Decode: L=1, no causal needed
            bool_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, S)
        else:
            bool_mask = None

        if q.dtype == torch.float32:
            # Manual attention for f32 — ROCm 7.2 / PyTorch 2.10 SDPA is completely
            # broken for f32 on MI210: 91/96 configs produce wrong results (zeros,
            # inf, diffs of 1-7x). See scripts/investigate_sdpa_bug.py.
            scale = self.head_dim**-0.5
            attn = torch.matmul(q, k_exp.transpose(2, 3)) * scale  # (B, nh, T_q, T_kv)
            if bool_mask is not None:
                attn = attn.masked_fill(~bool_mask, float("-inf"))
            else:
                # Prefill without padding (collect_activations path)
                assert T_q > 1, (
                    "decode without attn_mask — generate() must always pass kv_mask"
                )
                mask = torch.triu(
                    torch.full((T_q, T_q), float("-inf"), device=q.device), diagonal=1
                )
                attn = attn + mask
            attn = F.softmax(attn, dim=-1, dtype=torch.float32)
            y = torch.matmul(attn, v_exp)
        else:
            # SDPA — is_causal and attn_mask are MUTUALLY EXCLUSIVE in PyTorch
            if bool_mask is not None:
                y = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=bool_mask)
            else:
                # Prefill without padding (collect_activations path)
                assert T_q > 1, (
                    "decode without attn_mask — generate() must always pass kv_mask"
                )
                y = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)

        y = y.transpose(1, 2).reshape(B, T_q, self.n_head * self.head_dim)
        out = self.o_proj(y)
        return (out, new_kv) if return_kv else out


class SwiGLUMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = Attention(config)
        self.norm_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv: bool = False,
        post_attn_steering: torch.Tensor | None = None,
        post_mlp_steering: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out = self.attn(
            self.norm_1(x), cos, sin, attn_mask, kv_cache=kv_cache, return_kv=return_kv
        )
        if return_kv:
            attn_out, new_kv = attn_out
        x = x + attn_out
        if post_attn_steering is not None:
            x = x + post_attn_steering
        x = x + self.mlp(self.norm_2(x))
        if post_mlp_steering is not None:
            x = x + post_mlp_steering
        return (x, new_kv) if return_kv else x


class OracleTransformer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        injection_layer: int = 1,
        steering_coefficient: float = 1.0,
        vector_init_scale: float = 0.0,
        injection_points: list[str] | None = None,
    ):
        super().__init__()
        self.config = config
        self.injection_layer = injection_layer
        self.steering_coefficient = steering_coefficient
        self.injection_points = injection_points or ["post_mlp"]
        for p in self.injection_points:
            assert p in VALID_INJECTION_POINTS, (
                f"Invalid injection point {p!r}, must be one of {VALID_INJECTION_POINTS}"
            )

        # Base model — use meta device to skip weight init (we'll load from checkpoint)
        with torch.device("meta"):
            self.embed = nn.Embedding(config.vocab_size, config.n_embd)
            self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
            self.norm = RMSNorm(config.n_embd, eps=config.norm_eps)
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Steering vectors — keyed by injection point, one ParameterList per point.
        # These are real, not meta (small, need actual init).
        self.steering_vectors = nn.ModuleDict(
            {
                point: nn.ParameterList(
                    [
                        nn.Parameter(torch.randn(config.n_embd) * vector_init_scale)
                        for _ in range(config.n_layer)
                    ]
                )
                for point in self.injection_points
            }
        )

        cos, sin = build_rope_cache(
            config.max_seq_len, config.head_dim, config.rope_base
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _get_rope(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) each shaped (1, seq_len, head_dim)."""
        if seq_len > self.rope_cos.size(1):
            cos, sin = build_rope_cache(
                seq_len,
                self.config.head_dim,
                self.config.rope_base,
                self.rope_cos.device,
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        return self.rope_cos[:, :seq_len], self.rope_sin[:, :seq_len]

    def freeze_base(self):
        for name, param in self.named_parameters():
            if "steering_vectors" not in name:
                param.requires_grad = False

    def _steering_at_layer(
        self,
        layer_idx: int,
        vectors=None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        sv = vectors if vectors is not None else self.steering_vectors
        pa = sv["post_attn"][layer_idx] if "post_attn" in sv else None
        pm = sv["post_mlp"][layer_idx] if "post_mlp" in sv else None
        return pa, pm

    @staticmethod
    def _build_prefill_mask(
        attention_mask: torch.Tensor,
        T: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build 4D causal+padding mask once, to be shared across all blocks.

        Args:
            attention_mask: (B, T) bool, True=attend
        Returns:
            (B, 1, T, T) bool mask combining causal and padding constraints.
        """
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        causal = ~torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )
        return pad_mask & causal  # (B, 1, T, T)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        injected_activations: torch.Tensor | None = None,  # (B, K, D)
        injection_positions: torch.Tensor | None = None,  # (B, K) long
        use_steering: bool = True,
        attention_mask: torch.Tensor | None = None,  # (B, T) bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = input_ids.shape
        x = self.embed(input_ids)
        cos, sin = self._get_rope(T)

        # Pre-build 4D mask once for all blocks (avoids rebuilding triu per block)
        mask_4d = (
            self._build_prefill_mask(attention_mask, T, x.device)
            if attention_mask is not None
            else None
        )

        for i, block in enumerate(self.blocks):
            pa, pm = self._steering_at_layer(i) if use_steering else (None, None)
            x = block(x, cos, sin, mask_4d, post_attn_steering=pa, post_mlp_steering=pm)

            if i == self.injection_layer and injected_activations is not None:
                x = self._inject_activations(
                    x, injected_activations, injection_positions
                )

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Standard next-token prediction: logits[t] (seeing tokens 0..t)
            # predicts targets[t+1]. Without this shift, the model trivially
            # learns to echo the current token (loss→0 but 0% eval accuracy).
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def _inject_activations(
        self,
        x: torch.Tensor,
        activations: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Norm-matched additive injection: h' = h + ||h|| * normalize(v) * coeff

        activations: (B, K, D) collected hidden states
        positions: (B, K) long — which sequence positions to inject into
        """
        B, K, D = activations.shape
        pos_exp = positions.unsqueeze(-1).expand(B, K, D)  # (B, K, D)
        orig = torch.gather(x, 1, pos_exp)  # (B, K, D)
        norms = orig.norm(dim=-1, keepdim=True)  # (B, K, 1)
        v_normed = F.normalize(activations.to(x.dtype), dim=-1)
        v_normed = torch.nan_to_num(
            v_normed, nan=0.0
        )  # zero-padded acts → zero injection
        x = x.clone()
        x.scatter_(1, pos_exp, orig + norms * v_normed * self.steering_coefficient)
        return x

    @torch.no_grad()
    def collect_activations(
        self,
        input_ids: torch.Tensor,
        layer: int,
        positions: list[list[int]],
        attention_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Collect activations from a single layer with steering disabled.

        Returns list of (K_b, D) tensors — activations at the given positions per item.
        """
        results = self.collect_activations_multi(input_ids, [layer], attention_mask)
        x = results[layer]
        return [x[b, pos] for b, pos in enumerate(positions)]

    @torch.no_grad()
    def collect_activations_multi(
        self,
        input_ids: torch.Tensor,
        layers: list[int],
        attention_mask: torch.Tensor | None = None,
    ) -> dict[int, torch.Tensor]:
        """Collect full hidden states at multiple layers in a single forward pass.

        Runs embed + blocks up to max(layers), holding GPU references at each
        requested layer. block() returns a new tensor each call, so earlier
        snapshots stay valid. Callers should extract positions on GPU and use
        non-blocking D2H copies for the small per-example results.

        Returns: {layer: (B, T, D)} tensors on GPU.
        """
        B, T = input_ids.shape
        x = self.embed(input_ids)
        cos, sin = self._get_rope(T)

        mask_4d = (
            self._build_prefill_mask(attention_mask, T, x.device)
            if attention_mask is not None
            else None
        )

        layer_set = set(layers)
        max_layer = max(layers)
        results = {}

        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, attn_mask=mask_4d)
            if i in layer_set:
                results[i] = x
            if i == max_layer:
                break

        return results

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        im_end_id: int,
        pad_token_id: int,
        injected_activations: torch.Tensor | None = None,
        injection_positions: torch.Tensor | None = None,
        use_steering: bool = True,
        steering_vectors=None,
    ) -> torch.Tensor:
        """Greedy decode with KV cache. Returns generated token ids (B, num_generated)."""
        B, T = input_ids.shape
        cos, sin = self._get_rope(T + max_new_tokens)

        # --- Prefill: full prompt, build KV caches ---
        x = self.embed(input_ids)
        prefill_mask = self._build_prefill_mask(attention_mask, T, x.device)
        kv_caches = []
        for i, block in enumerate(self.blocks):
            pa, pm = (
                self._steering_at_layer(i, steering_vectors)
                if use_steering
                else (None, None)
            )
            x, kv = block(
                x,
                cos[:, :T],
                sin[:, :T],
                prefill_mask,
                kv_cache=None,
                return_kv=True,
                post_attn_steering=pa,
                post_mlp_steering=pm,
            )
            kv_caches.append(kv)
            if i == self.injection_layer and injected_activations is not None:
                x = self._inject_activations(
                    x, injected_activations, injection_positions
                )

        logits = self.lm_head(self.norm(x))
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        finished = next_token.squeeze(1) == im_end_id
        next_token.masked_fill_(finished.unsqueeze(1), pad_token_id)
        generated = [next_token]

        # Running KV mask: tracks which KV positions are real vs padding.
        # Padding comes from two sources: (1) left-pad in prompt, (2) pad tokens
        # appended for finished sequences during decode.
        kv_mask = attention_mask  # (B, T), True=real, False=pad

        # --- Decode: one token at a time against cached K, V ---
        for step in range(max_new_tokens - 1):
            if finished.all():
                break
            pos = T + step
            # Extend mask: real tokens for active sequences, pad for finished ones
            kv_mask = torch.cat([kv_mask, ~finished.unsqueeze(1)], dim=1)

            x = self.embed(next_token)
            # Always pass mask — overhead is negligible, avoids bugs from
            # skipping mask when finished sequences create new padding mid-decode
            for i, block in enumerate(self.blocks):
                pa, pm = (
                    self._steering_at_layer(i, steering_vectors)
                    if use_steering
                    else (None, None)
                )
                x, kv = block(
                    x,
                    cos[:, pos : pos + 1],
                    sin[:, pos : pos + 1],
                    kv_mask,
                    kv_cache=kv_caches[i],
                    return_kv=True,
                    post_attn_steering=pa,
                    post_mlp_steering=pm,
                )
                kv_caches[i] = kv
                # No injection during decode (only at prefill positions)

            logits = self.lm_head(self.norm(x))
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            next_token.masked_fill_(finished.unsqueeze(1), pad_token_id)
            generated.append(next_token)
            finished = finished | (next_token.squeeze(1) == im_end_id)

        return torch.cat(generated, dim=1)  # (B, num_generated)
