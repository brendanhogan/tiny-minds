"""
Brain - A tiny transformer neural network for each creature.

Each creature's brain is a small transformer that:
  1. Maintains a memory of recent observations (context window)
  2. Uses self-attention to attend over past experiences
  3. Outputs action decisions based on current + past context

This means creatures can remember where they saw food, learn patterns,
and make decisions based on history - not just current observation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    This is the core of the transformer - it lets the network
    "look back" at past inputs and decide what's relevant.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (seq_len, embed_dim)

        Returns:
            Output tensor of shape (seq_len, embed_dim)
        """
        seq_len, embed_dim = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (seq_len, embed_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        # (seq_len, embed_dim) -> (num_heads, seq_len, head_dim)
        q = q.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale  # (num_heads, seq_len, seq_len)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (num_heads, seq_len, head_dim)

        # Reshape back
        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, embed_dim)

        # Output projection
        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    """
    A single transformer block with:
      - Multi-head self-attention
      - Feed-forward network
      - Layer normalization
      - Residual connections
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
        super().__init__()

        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attention(self.norm1(x))
        x = x + attn_out

        # Feed-forward with residual
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out

        return x


class Brain(nn.Module):
    """
    A tiny transformer brain for a creature.

    Architecture:
      - Input embedding: projects raw sensory input to embed_dim
      - Positional encoding: so the transformer knows temporal order
      - N transformer blocks with self-attention
      - Output projection: from embed_dim to action space

    The creature maintains a context window of recent inputs.
    Each timestep, the new input is added and the oldest is dropped.
    The transformer attends over this history to make decisions.
    """

    def __init__(self, input_dim: int, output_dim: int, embed_dim: int = 32,
                 num_heads: int = 2, num_layers: int = 1, context_len: int = 8):
        """
        Create a new transformer brain.

        Args:
            input_dim: Size of sensory input each timestep
            output_dim: Size of action output
            embed_dim: Internal embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            context_len: How many past timesteps to remember
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.context_len = context_len

        # Input projection: raw input -> embedding
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(context_len, embed_dim) * 0.1)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 2)
            for _ in range(num_layers)
        ])

        # Output projection: embedding -> actions
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, output_dim),
            nn.Tanh(),  # Keep outputs in [-1, 1]
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for stability and good starting behavior."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization with smaller gain
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    # Small random bias to break symmetry and encourage movement
                    nn.init.uniform_(module.bias, -0.1, 0.1)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        """
        Run the transformer on current input + context.

        Args:
            x: Current input tensor of shape (input_dim,)
            context: Past inputs tensor of shape (context_len-1, input_dim) or None

        Returns:
            output: Action tensor of shape (output_dim,)
            new_context: Updated context including current input
        """
        # Ensure x is 1D
        if x.dim() > 1:
            x = x.squeeze(0)

        # Initialize context if None
        if context is None:
            context = torch.zeros(self.context_len - 1, self.input_dim)

        # Add current input to context (shift old ones out)
        # context shape: (context_len-1, input_dim)
        # new_context shape: (context_len, input_dim)
        new_context = torch.cat([context[1:], x.unsqueeze(0)], dim=0)

        # Add current observation to make full sequence
        full_sequence = torch.cat([context, x.unsqueeze(0)], dim=0)  # (context_len, input_dim)

        # Project inputs to embedding space
        embedded = self.input_proj(full_sequence)  # (context_len, embed_dim)

        # Add positional encoding
        embedded = embedded + self.pos_encoding

        # Run through transformer blocks
        for block in self.transformer_blocks:
            embedded = block(embedded)

        # Take the last position (current timestep) for output
        final_embedding = embedded[-1]  # (embed_dim,)

        # Project to action space
        output = self.output_proj(final_embedding)  # (output_dim,)

        return output, new_context

    def copy_with_noise(self, noise_std: float = 0.05) -> 'Brain':
        """
        Create a copy of this brain with small random mutations.
        Used when a creature reproduces.
        """
        new_brain = Brain(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            context_len=self.context_len
        )

        # Copy weights with noise
        with torch.no_grad():
            for (name1, param1), (name2, param2) in zip(
                self.named_parameters(), new_brain.named_parameters()
            ):
                noise = torch.randn_like(param1) * noise_std
                param2.copy_(param1 + noise)

        return new_brain

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters())


def test_brain_stability(brain: Brain) -> bool:
    """
    Test if a brain is dynamically stable.

    We run random inputs through the brain and check if outputs
    stay bounded. Unstable brains explode to infinity or NaN.
    """
    context = None

    for _ in range(config.STABILITY_TEST_STEPS):
        # Random input
        x = torch.randn(brain.input_dim)

        try:
            output, context = brain(x, context)
        except Exception:
            return False

        # Check for NaN or Inf
        if torch.isnan(output).any() or torch.isinf(output).any():
            return False

        if torch.abs(output).max() > config.STABILITY_THRESHOLD:
            return False

        # Check context
        if context is not None:
            if torch.isnan(context).any() or torch.isinf(context).any():
                return False

    return True
