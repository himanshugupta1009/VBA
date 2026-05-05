import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

# Assumes your repo path setup allows these imports from external/set_transformer
sys.path.append(str(Path(__file__).resolve().parent / "external" / "set_transformer"))
from models import SetTransformer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Normalization + Attention Blocks
# ============================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    Paper: https://arxiv.org/abs/1910.07467
    Used in: Llama-3, Grok, PaLM

    Formula:
        a_bar_i = (a_i / RMS(a)) * g_i
        where RMS(a) = sqrt(mean(a^2) + eps)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scale parameter 'g' (gamma)
        self.scale = nn.Parameter(torch.ones(dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
        Returns:
            Normalized tensor of same shape
        """
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) # permit broadcasting on last dim
        x_normed = x * rms
        x_scaled = x_normed * self.scale

        return x_scaled


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)

    Paper: RoFormer (Su et al., 2021) - https://arxiv.org/abs/2104.09864
    Used in: Llama-3, PaLM-E, GPT-NeoX

    Key Idea: Rotate pairs of dimensions by an angle proportional to position
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE head dimension must be even."
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute rotation frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos and sin for all positions
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Precompute cos and sin values for all positions"""
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Helper function to rotate tensor by swapping and negating half the dimensions"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)
        """
        This does the same thing

        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
        """

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors

        Args:
            q: Query tensor (batch, num_heads, seq_len, head_dim)
            k: Key tensor (batch, num_heads, seq_len, head_dim)
        Returns:
            Rotated (q, k) tensors
        """
        seq_len = q.shape[2]

        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)

        # Get cached cos/sin values
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        # Apply rotation: q_rot = q * cos + rotate_half(q) * sin
        q = (q * cos) + (self.rotate_half(q) * sin)
        k = (k * cos) + (self.rotate_half(k) * sin)

        return q, k


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with RoPE

    Key constraints:
    - Token at position t can only attend to positions <= t
    - Uses RoPE instead of absolute positional embeddings
    """
        
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Rotary embeddings
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        #bsz is batch size, seq_len is sequence length

        # Project input to q, k, v
        # Recall that `reshape` does `view` automatically if it's possible.

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now (q, k, v) should all be of shape (batch, num_heads, seq_len, head_dim)

        # Apply RoPE to Q and K
        q, k = self.rope(q, k)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        # (batch, num_heads, seq_len, seq_len)

        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))

        # Softmax
        attn = F.softmax(scores, dim=-1)
        # Dropout
        attn = self.attn_dropout(attn)
        # Apply values
        out = torch.matmul(attn, v) #(batch, num_heads, seq_len, head_dim)

        # Concatenate heads and apply output projection
        # Heads may or may not be contiguous - reshape should handle both cases
        # out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.dim)
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.dim)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network with SwiGLU activation

    Used in modern LLMs for better performance than standard ReLU
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
        Returns:
            Output tensor (batch, seq_len, dim)
        """

        x = self.fc1(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block

    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + FeedForward(RMSNorm(x))
    """
    
    def __init__(self, dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = CausalSelfAttention(dim, num_heads, dropout)
        self.feed_forward = FeedForward(dim, ff_hidden_dim, dropout)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask
        Returns:
            Output tensor (batch, seq_len, dim)
        """

        # Pre-norm architecture (norm before attention/FF)
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.feed_forward(self.norm2(x))
        return x


# ============================================================
# Prefix Mask
# ============================================================

# def build_prefix_lm_mask(seq_len: int, action_start_idx: int, device: torch.device) -> torch.Tensor:
#     """
#     Prefix tokens: fully visible among themselves.
#     Action tokens: can attend to all prefix tokens + past/current action tokens.
#     """
#     mask = torch.zeros(seq_len, seq_len, device=device)

#     mask[:action_start_idx, :action_start_idx] = 1
#     mask[action_start_idx:, :action_start_idx] = 1

#     action_len = seq_len - action_start_idx
#     mask[action_start_idx:, action_start_idx:] = torch.tril(
#         torch.ones(action_len, action_len, device=device)
#     )
#     return mask


# ============================================================
# Decoder
# ============================================================

class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        ff_hidden_dim: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm_final = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        # action_start_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            x = self.token_embedding(input_ids)
        else:
            x = inputs_embeds

        _, seq_len, _ = x.shape
        device = x.device

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm_final(x)
        logits = self.lm_head(x)

        loss = None            
        if targets is not None:
            if targets.shape[:2] != logits.shape[:2]:
                raise ValueError(
                    f"targets must have shape (batch, seq_len). "
                    f"Got targets {targets.shape}, logits {logits.shape}"
                )

            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-1,
            )

        return logits, loss


# ============================================================
# Full Vision-Language-Belief-Action Model (FLAT VERSION)
# ============================================================

class VLABAModel(nn.Module):
    def __init__(
        self,
        clip_model_name: str,
        particle_dim: int,
        action_vocab_size: int,
        decoder_dim: int = 512,
        num_belief_tokens: int = 1,
        set_num_inds: int = 32,
        set_hidden_dim: int = 128,
        set_num_heads: int = 4,
        num_decoder_layers: int = 4,
        num_decoder_heads: int = 8,
        ff_hidden_dim: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        freeze_clip: bool = True,
        use_vision_cls: bool = False,
    ):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.use_vision_cls = use_vision_cls

        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        if freeze_clip:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        self.vision_proj = nn.Linear(
            self.clip_model.vision_model.config.hidden_size,
            decoder_dim,
            bias=False,
        )
        self.text_proj = nn.Linear(
            self.clip_model.text_model.config.hidden_size,
            decoder_dim,
            bias=False,
        )

        with torch.no_grad():
            self.vision_proj.weight.copy_(self.clip_model.visual_projection.weight)
            self.text_proj.weight.copy_(self.clip_model.text_projection.weight)

        self.belief_encoder = SetTransformer(
            dim_input=particle_dim,
            num_outputs=num_belief_tokens,
            dim_output=decoder_dim,
            num_inds=set_num_inds,
            dim_hidden=set_hidden_dim,
            num_heads=set_num_heads,
            ln=False,
        )

        self.decoder = DecoderOnlyTransformer(
            vocab_size=action_vocab_size,
            dim=decoder_dim,
            num_layers=num_decoder_layers,
            num_heads=num_decoder_heads,
            ff_hidden_dim=ff_hidden_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

    def encode_modalities(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        particles: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        clip_outputs = self.clip_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        vision_tokens = clip_outputs.vision_model_output.last_hidden_state
        if not self.use_vision_cls:
            vision_tokens = vision_tokens[:, 1:, :]
        vision_tokens = self.vision_proj(vision_tokens)

        text_tokens = clip_outputs.text_model_output.last_hidden_state
        text_tokens = self.text_proj(text_tokens)

        belief_tokens = self.belief_encoder(particles)

        prefix_tokens = torch.cat([vision_tokens, text_tokens, belief_tokens], dim=1)

        info = {
            "num_vision_tokens": vision_tokens.shape[1],
            "num_text_tokens": text_tokens.shape[1],
            "num_belief_tokens": belief_tokens.shape[1],
            "prefix_len": prefix_tokens.shape[1],
        }
        return prefix_tokens, info

    def forward(
        self,
        pixel_values: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        particles: torch.Tensor,
        action_input_ids: torch.Tensor,
        action_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        prefix_tokens, info = self.encode_modalities(
            pixel_values=pixel_values,
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            particles=particles,
        )

        action_embeds = self.decoder.token_embedding(action_input_ids)
        inputs_embeds = torch.cat([prefix_tokens, action_embeds], dim=1)
        action_start_idx = prefix_tokens.shape[1]
        full_targets = None
        if action_targets is not None:
            prefix_targets = torch.full(
                (action_targets.shape[0], action_start_idx),
                fill_value=-1,
                dtype=action_targets.dtype,
                device=action_targets.device,
            )
            full_targets = torch.cat([prefix_targets, action_targets], dim=1)

        logits, loss = self.decoder(
            inputs_embeds=inputs_embeds,
            targets=full_targets,
        )

        action_logits = logits[:, action_start_idx:, :]
        return action_logits, loss, info

    @torch.no_grad()
    def generate_actions(
        self,
        pixel_values: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        particles: torch.Tensor,
        bos_token_id: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        prefix_tokens, _ = self.encode_modalities(
            pixel_values=pixel_values,
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            particles=particles,
        )

        batch_size = pixel_values.shape[0]
        generated = torch.full(
            (batch_size, 1),
            fill_value=bos_token_id,
            dtype=torch.long,
            device=pixel_values.device,
        )

        for _ in range(max_new_tokens):
            action_embeds = self.decoder.token_embedding(generated)
            inputs_embeds = torch.cat([prefix_tokens, action_embeds], dim=1)
            logits, _ = self.decoder(
                inputs_embeds=inputs_embeds,
                targets=None,
            )
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.shape[-1]))
                next_logits[next_logits < values[:, [-1]]] = -float("inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated



# ============================================================
# Full Vision-Language-Belief-Action Model (SEQUENTIAL VERSION)
# ============================================================

class VLBAModel(nn.Module):
    def __init__(
        self,
        clip_model_name: str,
        particle_dim: int,
        action_vocab_size: int,
        clip_cache_dir: Optional[str] = None,
        decoder_dim: int = 512,
        num_belief_tokens: int = 1,
        set_num_inds: int = 32,
        set_hidden_dim: int = 128,
        set_num_heads: int = 4,
        num_decoder_layers: int = 4,
        num_decoder_heads: int = 8,
        ff_hidden_dim: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        freeze_clip: bool = True,
        use_vision_cls: bool = False,
    ):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.use_vision_cls = use_vision_cls
        self.num_belief_tokens = num_belief_tokens
        self.action_vocab_size = action_vocab_size

        self.clip_model = CLIPModel.from_pretrained(
            clip_model_name,
            cache_dir=clip_cache_dir,
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            clip_model_name,
            cache_dir=clip_cache_dir,
        )

        if freeze_clip:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        self.vision_proj = nn.Linear(
            self.clip_model.vision_model.config.hidden_size,
            decoder_dim,
            bias=False,
        )
        self.text_proj = nn.Linear(
            self.clip_model.text_model.config.hidden_size,
            decoder_dim,
            bias=False,
        )

        # Safer to let these train rather than copying CLIP pooled projection
        # weights onto token-level hidden states.
        nn.init.normal_(self.vision_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.text_proj.weight, mean=0.0, std=0.02)

        self.belief_encoder = SetTransformer(
            dim_input=particle_dim,
            num_outputs=num_belief_tokens,
            dim_output=decoder_dim,
            num_inds=set_num_inds,
            dim_hidden=set_hidden_dim,
            num_heads=set_num_heads,
            ln=False,
        )

        self.decoder = DecoderOnlyTransformer(
            vocab_size=action_vocab_size,
            dim=decoder_dim,
            num_layers=num_decoder_layers,
            num_heads=num_decoder_heads,
            ff_hidden_dim=ff_hidden_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

    def encode_text(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            text_tokens: (B, Lt, D)
            text_valid_mask: (B, Lt) bool
        """
        text_outputs = self.clip_model.text_model(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
        )
        text_tokens = self.text_proj(text_outputs.last_hidden_state)
        text_valid_mask = text_attention_mask.bool()
        text_tokens = text_tokens * text_valid_mask.unsqueeze(-1)
        return text_tokens, text_valid_mask

    def encode_vision_sequence(
        self,
        pixel_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        Args:
            pixel_values: (B, T, C, H, W)

        Returns:
            vision_tokens: (B, T, Lv, D)
            Lv: number of vision tokens per timestep
        """
        B, T, C, H, W = pixel_values.shape
        flat_pixels = pixel_values.reshape(B * T, C, H, W)

        vision_outputs = self.clip_model.vision_model(pixel_values=flat_pixels)
        vision_tokens = vision_outputs.last_hidden_state  # (B*T, Lv_full, Hdim)

        if not self.use_vision_cls:
            vision_tokens = vision_tokens[:, 1:, :]  # drop CLS if desired

        vision_tokens = self.vision_proj(vision_tokens)  # (B*T, Lv, D)
        Lv = vision_tokens.shape[1]
        vision_tokens = vision_tokens.reshape(B, T, Lv, self.decoder_dim)
        return vision_tokens, Lv

    def encode_belief_sequence(
        self,
        particles: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        Args:
            particles: (B, T, Np, particle_dim)

        Returns:
            belief_tokens: (B, T, Lb, D)
            Lb: number of belief tokens per timestep
        """
        B, T, Np, Pdim = particles.shape
        flat_particles = particles.reshape(B * T, Np, Pdim)

        belief_tokens = self.belief_encoder(flat_particles)  # (B*T, Lb, D)
        Lb = belief_tokens.shape[1]
        belief_tokens = belief_tokens.reshape(B, T, Lb, self.decoder_dim)
        return belief_tokens, Lb

    def build_sequence_and_targets(
        self,
        text_tokens: torch.Tensor,          # (B, Lt, D)
        text_valid_mask: torch.Tensor,      # (B, Lt) bool
        vision_tokens: torch.Tensor,        # (B, T, Lv, D)
        belief_tokens: torch.Tensor,        # (B, T, Lb, D)
        action_input_ids: torch.Tensor,     # (B, T)
        action_targets: Optional[torch.Tensor] = None,   # (B, T)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """
        Constructs:
            [text_once] + [vision_1, belief_1, action_1] + ... + [vision_T, belief_T, action_T]

        Returns:
            inputs_embeds: (B, seq_len, D)
            full_targets: (B, seq_len) with -1 for non-action positions
        """
        B, Lt, D = text_tokens.shape
        _, T, Lv, _ = vision_tokens.shape
        _, _, Lb, _ = belief_tokens.shape

        action_embeds = self.decoder.token_embedding(action_input_ids).unsqueeze(2)  # (B, T, 1, D)

        pieces = []
        target_pieces = []

        # Text once at the beginning
        pieces.append(text_tokens)

        if action_targets is not None:
            text_targets = torch.full(
                (B, Lt),
                fill_value=-1,
                dtype=torch.long,
                device=text_tokens.device,
            )

            # Also ignore padded text positions explicitly
            text_targets = text_targets.masked_fill(~text_valid_mask, -1)
            target_pieces.append(text_targets)

        # Interleave per timestep
        for t in range(T):
            pieces.append(vision_tokens[:, t])   # (B, Lv, D)
            pieces.append(belief_tokens[:, t])   # (B, Lb, D)
            pieces.append(action_embeds[:, t])   # (B, 1, D)

            if action_targets is not None:
                vision_targets = torch.full(
                    (B, Lv),
                    fill_value=-1,
                    dtype=torch.long,
                    device=text_tokens.device,
                )
                belief_targets = torch.full(
                    (B, Lb),
                    fill_value=-1,
                    dtype=torch.long,
                    device=text_tokens.device,
                )
                step_action_targets = action_targets[:, t].unsqueeze(1)  # (B, 1)

                target_pieces.append(vision_targets)
                target_pieces.append(belief_targets)
                target_pieces.append(step_action_targets)

        inputs_embeds = torch.cat(pieces, dim=1) #Shape becomes (B, seq_len, D)
        # seq_len = Lt + T * (Lv + Lb + 1)

        full_targets = None
        if action_targets is not None:
            full_targets = torch.cat(target_pieces, dim=1)
            # full_targets shape: (B, seq_len) with -1 for non-action positions
            # and actual token ids for action positions

        info = {
            "num_text_tokens": Lt,
            "num_vision_tokens_per_step": Lv,
            "num_belief_tokens_per_step": Lb,
            "num_steps": T,
            "seq_len": inputs_embeds.shape[1],
        }

        return inputs_embeds, full_targets, info

    def forward(
        self,
        pixel_values: torch.Tensor,         # (B, T, C, H, W)
        text_input_ids: torch.Tensor,       # (B, Lt)
        text_attention_mask: torch.Tensor,  # (B, Lt)
        particles: torch.Tensor,            # (B, T, Np, particle_dim)
        action_input_ids: torch.Tensor,     # (B, T)
        action_targets: Optional[torch.Tensor] = None,   # (B, T)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        text_tokens, text_valid_mask = self.encode_text(
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
        )

        vision_tokens, Lv = self.encode_vision_sequence(pixel_values)
        belief_tokens, Lb = self.encode_belief_sequence(particles)

        inputs_embeds, full_targets, info = self.build_sequence_and_targets(
            text_tokens=text_tokens,
            text_valid_mask=text_valid_mask,
            vision_tokens=vision_tokens,
            belief_tokens=belief_tokens,
            action_input_ids=action_input_ids,
            action_targets=action_targets,
        )

        logits, loss = self.decoder(
            inputs_embeds=inputs_embeds,
            targets=full_targets,
        )

        return logits, loss, info

# ============================================================
# Example Dataset Stub
# Replace this with your real dataset.
# ============================================================

class VLBADataset(Dataset):
    def __init__(
        self,
        processor: CLIPProcessor,
        episodes,
        image_size: int = 224,
        window_len: Optional[int] = None,
        stride: int = 1,
    ):
        """
        episodes: list of dicts with:
            prompt: str
            image_paths or images: length-T list of image paths/PIL images
            particles: (T, num_particles, particle_dim)
            action_tokens: (T,)

        For autoregressive action prediction, each action position receives
        previous action context as input and predicts the current action:
            input  = [BOS, action_1, ..., action_{T-1}]
            target = [action_1, ..., action_T]
        """
        self.processor = processor
        self.episodes = episodes
        self.image_size = image_size
        self.window_len = window_len
        self.stride = stride
        self.samples = []

        for episode_idx, episode in enumerate(episodes):
            episode_len = len(episode["action_tokens"])
            if window_len is None:
                self.samples.append((episode_idx, 0, episode_len))
            else:
                for start in range(0, episode_len - window_len + 1, stride):
                    self.samples.append((episode_idx, start, start + window_len))

        if not self.samples:
            raise ValueError("No dataset samples were created. Try a shorter window_len.")

    def __len__(self):
        return len(self.samples)

    def _load_images(self, episode):
        if "images" in episode:
            images = episode["images"]
        else:
            images = [
                Image.open(Path(path)).convert("RGB")
                for path in episode["image_paths"]
            ]
        return [
            image.convert("RGB").resize((self.image_size, self.image_size))
            if isinstance(image, Image.Image)
            else Image.open(Path(image)).convert("RGB").resize((self.image_size, self.image_size))
            for image in images
        ]

    def __getitem__(self, idx):
        episode_idx, start, end = self.samples[idx]
        episode = self.episodes[episode_idx]
        episode_slice = {
            "prompt": episode["prompt"],
            "particles": episode["particles"][start:end],
            "action_tokens": episode["action_tokens"][start:end],
        }
        if "images" in episode:
            episode_slice["images"] = episode["images"][start:end]
        else:
            episode_slice["image_paths"] = episode["image_paths"][start:end]

        images = self._load_images(episode_slice)
        prompt = episode_slice["prompt"]

        enc = self.processor(
            text=[prompt],
            images=images,
            return_tensors="pt",
            padding=True,
        )

        particles = torch.as_tensor(episode_slice["particles"], dtype=torch.float32)
        action_tokens = torch.as_tensor(episode_slice["action_tokens"], dtype=torch.long)

        if particles.ndim != 3:
            raise ValueError(f"particles must be (T, Np, particle_dim), got {particles.shape}")
        if action_tokens.ndim != 1:
            raise ValueError(f"action_tokens must be (T,), got {action_tokens.shape}")
        if len(images) != particles.shape[0] or len(images) != action_tokens.shape[0]:
            raise ValueError(
                "episode lengths must match: "
                f"images={len(images)}, particles={particles.shape[0]}, actions={action_tokens.shape[0]}"
            )

        return {
            "pixel_values": enc["pixel_values"],
            "text_input_ids": enc["input_ids"].squeeze(0),
            "text_attention_mask": enc["attention_mask"].squeeze(0),
            "particles": particles,
            "action_tokens": action_tokens,
        }


def vlba_collate_fn(batch):
    lengths = [x["action_tokens"].shape[0] for x in batch]
    if len(set(lengths)) != 1:
        raise ValueError(
            "This collate_fn expects fixed-length episodes/windows. "
            "Crop or pad episodes before batching variable horizons."
        )

    pixel_values = torch.stack([x["pixel_values"] for x in batch], dim=0)

    text_input_ids = nn.utils.rnn.pad_sequence(
        [x["text_input_ids"] for x in batch],
        batch_first=True,
        padding_value=0,
    )
    text_attention_mask = nn.utils.rnn.pad_sequence(
        [x["text_attention_mask"] for x in batch],
        batch_first=True,
        padding_value=0,
    )

    particles = torch.stack([x["particles"] for x in batch], dim=0)
    action_tokens = torch.stack([x["action_tokens"] for x in batch], dim=0)

    return {
        "pixel_values": pixel_values,
        "text_input_ids": text_input_ids,
        "text_attention_mask": text_attention_mask,
        "particles": particles,
        "action_tokens": action_tokens,
    }


def load_collected_vlba_episodes(
    run_dir,
    prompt: str,
    action_tokenizer,
    include_particle_weights: bool = True,
    include_orientation: bool = True,
    successful_only: bool = True,
    success_reward: float = 100.0,
):
    """
    Load episodes written by scripts/run_vts_lightdark.py --collect-dataset.

    action_tokenizer receives one continuous action array and must return one
    integer token id. By default, each particle becomes:
        [x, y, sin(theta), cos(theta), normalized_weight]
    so TrainConfig.particle_dim should be 5.

    If successful_only=True, keep only episodes that terminated in success.
    """
    run_dir = Path(run_dir)
    episode_dirs = sorted((run_dir / "episodes").glob("episode_*"))
    episodes = []

    for episode_dir in episode_dirs:
        data = np.load(episode_dir / "episode_data.npz", allow_pickle=True)
        is_success = bool(data["dones"][-1]) and float(data["rewards"][-1]) >= success_reward
        if successful_only and not is_success:
            continue

        image_paths = [
            str(episode_dir / "images" / image_file)
            for image_file in data["image_files"].tolist()
        ]

        particle_states = data["particle_states"].astype(np.float32)
        particle_features = [particle_states]

        if include_orientation:
            orientations = data["orientations"].astype(np.float32)
            sin_theta = np.sin(orientations)[:, None, None]
            cos_theta = np.cos(orientations)[:, None, None]
            sin_theta = np.repeat(sin_theta, particle_states.shape[1], axis=1)
            cos_theta = np.repeat(cos_theta, particle_states.shape[1], axis=1)
            particle_features.extend([sin_theta.astype(np.float32), cos_theta.astype(np.float32)])

        if include_particle_weights:
            particle_weights = data["particle_weights"].astype(np.float32)[..., None]
            particle_features.append(particle_weights)

        particles = np.concatenate(particle_features, axis=-1)

        action_tokens = np.asarray(
            [action_tokenizer(action) for action in data["actions"]],
            dtype=np.int64,
        )

        episodes.append(
            {
                "prompt": prompt,
                "image_paths": image_paths,
                "particles": particles,
                "action_tokens": action_tokens,
                "episode_dir": str(episode_dir),
                "success": is_success,
                "episode_return": float(data["rewards"].sum()),
            }
        )

    return episodes


LIGHTDARK_ACTIONS = [
    (-0.2, -0.2),
    (0.0, -0.2),
    (0.2, -0.2),
    (-0.2, 0.0),
    (0.2, 0.0),
    (-0.2, 0.2),
    (0.0, 0.2),
    (0.2, 0.2),
]


def tokenize_lightdark_action(action) -> int:
    action_key = tuple(round(float(v), 1) for v in np.asarray(action).tolist())
    try:
        return LIGHTDARK_ACTIONS.index(action_key) + 1
    except ValueError as exc:
        raise ValueError(f"Unknown LightDark action {action}") from exc


# ============================================================
# Training / Evaluation
# ============================================================

@dataclass
class TrainConfig:
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_cache_dir: str = "/home/himanshu/.cache/huggingface"
    dataset_run_dir: str = "data/vlba_dataset_1000/vts_lightdark04-22-15_37_04"
    device: str = "cuda:1"
    output_dir: str = "checkpoints_vlba"
    prompt: str = "Navigate to the goal using the visual observation and particle belief."
    image_size: int = 224
    window_len: int = 8
    window_stride: int = 1
    train_fraction: float = 0.9
    successful_only: bool = True

    particle_dim: int = 5
    action_vocab_size: int = 9
    bos_token_id: int = 0

    decoder_dim: int = 512
    num_belief_tokens: int = 1
    set_num_inds: int = 32
    set_hidden_dim: int = 128
    set_num_heads: int = 4

    num_decoder_layers: int = 4
    num_decoder_heads: int = 8
    ff_hidden_dim: int = 2048
    max_seq_len: int = 1024
    dropout: float = 0.1

    freeze_clip: bool = True
    use_vision_cls: bool = False

    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 50
    grad_clip: float = 1.0
    log_every: int = 10


def train_epoch(model, dataloader, optimizer, device, cfg: TrainConfig, epoch: int):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_actions = 0
    num_batches = 0

    for step, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device)
        text_input_ids = batch["text_input_ids"].to(device)
        text_attention_mask = batch["text_attention_mask"].to(device)
        particles = batch["particles"].to(device)
        action_tokens = batch["action_tokens"].to(device)

        bos = torch.full(
            (action_tokens.shape[0], 1),
            fill_value=cfg.bos_token_id,
            dtype=action_tokens.dtype,
            device=device,
        )
        action_input_ids = torch.cat([bos, action_tokens[:, :-1]], dim=1)
        action_targets = action_tokens

        logits, loss, info = model(
            pixel_values=pixel_values,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            particles=particles,
            action_input_ids=action_input_ids,
            action_targets=action_targets,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        action_logits = extract_action_logits(logits, info, action_targets.shape[1])
        preds = action_logits.argmax(dim=-1)
        total_correct += (preds == action_targets).sum().item()
        total_actions += action_targets.numel()

        total_loss += loss.item()
        num_batches += 1

        if step % cfg.log_every == 0:
            running_acc = total_correct / max(total_actions, 1)
            print(
                f"Epoch {epoch+1} | Step {step}/{len(dataloader)} | "
                f"Loss {loss.item():.4f} | Acc {running_acc:.3f} | Seq {info['seq_len']} "
                f"(text={info['num_text_tokens']}, "
                f"vision/step={info['num_vision_tokens_per_step']}, "
                f"belief/step={info['num_belief_tokens_per_step']}, "
                f"steps={info['num_steps']})"
            )

    return {
        "loss": total_loss / max(num_batches, 1),
        "acc": total_correct / max(total_actions, 1),
    }


@torch.no_grad()
def evaluate(model, dataloader, device, cfg: TrainConfig):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_actions = 0
    num_batches = 0

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        text_input_ids = batch["text_input_ids"].to(device)
        text_attention_mask = batch["text_attention_mask"].to(device)
        particles = batch["particles"].to(device)
        action_tokens = batch["action_tokens"].to(device)

        bos = torch.full(
            (action_tokens.shape[0], 1),
            fill_value=cfg.bos_token_id,
            dtype=action_tokens.dtype,
            device=device,
        )
        action_input_ids = torch.cat([bos, action_tokens[:, :-1]], dim=1)
        action_targets = action_tokens

        logits, loss, info = model(
            pixel_values=pixel_values,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            particles=particles,
            action_input_ids=action_input_ids,
            action_targets=action_targets,
        )

        action_logits = extract_action_logits(logits, info, action_targets.shape[1])
        preds = action_logits.argmax(dim=-1)
        total_correct += (preds == action_targets).sum().item()
        total_actions += action_targets.numel()

        total_loss += loss.item()
        num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "acc": total_correct / max(total_actions, 1),
    }


def extract_action_logits(logits: torch.Tensor, info: dict, num_steps: int) -> torch.Tensor:
    Lt = info["num_text_tokens"]
    Lv = info["num_vision_tokens_per_step"]
    Lb = info["num_belief_tokens_per_step"]
    step_width = Lv + Lb + 1
    action_positions = torch.tensor(
        [Lt + t * step_width + (Lv + Lb) for t in range(num_steps)],
        dtype=torch.long,
        device=logits.device,
    )
    return logits[:, action_positions, :]


def save_learning_curve(history, output_path: str = "vlba_learning_curve.png"):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross entropy loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("action accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    cfg = TrainConfig()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {cfg.device}, but CUDA is not available.")
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    model = VLBAModel(
        clip_model_name=cfg.clip_model_name,
        particle_dim=cfg.particle_dim,
        action_vocab_size=cfg.action_vocab_size,
        clip_cache_dir=cfg.clip_cache_dir,
        decoder_dim=cfg.decoder_dim,
        num_belief_tokens=cfg.num_belief_tokens,
        set_num_inds=cfg.set_num_inds,
        set_hidden_dim=cfg.set_hidden_dim,
        set_num_heads=cfg.set_num_heads,
        num_decoder_layers=cfg.num_decoder_layers,
        num_decoder_heads=cfg.num_decoder_heads,
        ff_hidden_dim=cfg.ff_hidden_dim,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
        freeze_clip=cfg.freeze_clip,
        use_vision_cls=cfg.use_vision_cls,
    ).to(device)

    print("Trainable parameter counts:")
    total_params = 0
    trainable_params = 0
    for _, p in model.named_parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    print(f"  total:     {total_params:,}")
    print(f"  trainable: {trainable_params:,}")

    episodes = load_collected_vlba_episodes(
        run_dir=cfg.dataset_run_dir,
        prompt=cfg.prompt,
        action_tokenizer=tokenize_lightdark_action,
        include_particle_weights=True,
        include_orientation=True,
        successful_only=cfg.successful_only,
    )
    if not episodes:
        raise ValueError(
            "No episodes were loaded. If you want to include failed rollouts, "
            "set TrainConfig.successful_only=False."
        )
    print(f"Loaded {len(episodes)} episodes for training/evaluation.")
    split_idx = max(1, int(len(episodes) * cfg.train_fraction))
    train_episodes = episodes[:split_idx]
    val_episodes = episodes[split_idx:] or episodes[:1]

    train_dataset = VLBADataset(
        processor=model.clip_processor,
        episodes=train_episodes,
        image_size=cfg.image_size,
        window_len=cfg.window_len,
        stride=cfg.window_stride,
    )
    val_dataset = VLBADataset(
        processor=model.clip_processor,
        episodes=val_episodes,
        image_size=cfg.image_size,
        window_len=cfg.window_len,
        stride=cfg.window_stride,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=vlba_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=vlba_collate_fn,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_val = float("inf")
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    for epoch in range(cfg.num_epochs):
        t0 = time.perf_counter()
        train_metrics = train_epoch(model, train_loader, optimizer, device, cfg, epoch)
        val_metrics = evaluate(model, val_loader, device, cfg)
        dt = time.perf_counter() - t0
        train_loss = train_metrics["loss"]
        val_loss = val_metrics["loss"]

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        save_learning_curve(history, str(output_dir / "vlba_learning_curve.png"))

        print(
            f"Epoch {epoch+1}/{cfg.num_epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['acc']:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['acc']:.3f} | "
            f"time={dt:.2f}s"
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(cfg),
            "epoch": epoch,
            "history": history,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
        }
        torch.save(ckpt, output_dir / f"vlba_epoch_{epoch+1}.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, output_dir / "vlba_best.pt")

    print("Done.")


if __name__ == "__main__":
    main()
