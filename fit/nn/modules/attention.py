"""
Core Attention Mechanisms for FIT Framework

This module implements the fundamental attention mechanisms that power
modern deep learning: scaled dot-product attention, multi-head attention,
and various attention variants.

The implementation is educational (showing how attention really works)
while being efficient and production-ready.
"""

import numpy as np
from typing import Optional, Tuple, Union
import math

from fit.core.tensor import Tensor
from fit.nn.modules.base import Layer
from fit.nn.modules.linear import Linear
from fit.nn.modules.activation import Softmax, Dropout


class ScaledDotProductAttention(Layer):
    """
    Scaled Dot-Product Attention: the core of all attention mechanisms.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    This is the fundamental building block that makes Transformers work.
    """

    def __init__(self, dropout: float = 0.1, temperature: float = 1.0):
        """
        Initialize scaled dot-product attention.

        Args:
            dropout: Dropout probability for attention weights
            temperature: Temperature scaling factor (higher = more uniform attention)
        """
        super().__init__()
        self.dropout = Dropout(dropout) if dropout > 0 else None
        self.temperature = temperature
        self.softmax = Softmax()

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Apply scaled dot-product attention.

        Args:
            query: Query tensor (batch_size, seq_len_q, d_k)
            key: Key tensor (batch_size, seq_len_k, d_k)
            value: Value tensor (batch_size, seq_len_v, d_v)
            mask: Optional attention mask to prevent attention to certain positions
            return_attention: Whether to return attention weights

        Returns:
            Output tensor (batch_size, seq_len_q, d_v) and optionally attention weights
        """
        batch_size, seq_len_q, d_k = query.data.shape
        seq_len_k = key.data.shape[1]

        # Step 1: Compute attention scores = Q * K^T / sqrt(d_k)
        scores = self._compute_attention_scores(query, key, d_k)

        # Step 2: Apply mask if provided
        if mask is not None:
            scores = self._apply_mask(scores, mask)

        # Step 3: Apply softmax to get attention weights
        attention_weights = self.softmax(scores)

        # Step 4: Apply dropout to attention weights (if training)
        if self.dropout is not None:
            attention_weights = self.dropout(attention_weights)

        # Step 5: Apply attention to values
        output = self._apply_attention_to_values(attention_weights, value)

        if return_attention:
            return output, attention_weights
        else:
            return output

    def _compute_attention_scores(self, query: Tensor, key: Tensor, d_k: int) -> Tensor:
        """Compute raw attention scores."""
        # Q * K^T
        scores = query @ self._transpose_last_two_dims(key)

        # Scale by sqrt(d_k) and temperature
        scale_factor = math.sqrt(d_k) * self.temperature
        scores = scores / scale_factor

        return scores

    def _transpose_last_two_dims(self, tensor: Tensor) -> Tensor:
        """Transpose the last two dimensions of a tensor."""
        # For (batch, seq, dim) -> (batch, dim, seq)
        data = tensor.data
        transposed_data = np.transpose(data, (0, 2, 1))
        return Tensor(transposed_data, requires_grad=tensor.requires_grad)

    def _apply_mask(self, scores: Tensor, mask: Tensor) -> Tensor:
        """Apply attention mask by setting masked positions to large negative values."""
        # Mask should be 1 for positions to attend to, 0 for positions to ignore
        masked_scores = scores.data.copy()
        masked_scores[mask.data == 0] = -1e9  # Large negative value
        return Tensor(masked_scores, requires_grad=scores.requires_grad)

    def _apply_attention_to_values(
        self, attention_weights: Tensor, value: Tensor
    ) -> Tensor:
        """Apply attention weights to values."""
        return attention_weights @ value


class MultiHeadAttention(Layer):
    """
    Multi-Head Attention: the key innovation that makes Transformers so powerful.

    Instead of using a single attention function, we use multiple "heads" that
    can focus on different types of relationships in the data.
    """

    def __init__(
        self, d_model: int, num_heads: int, dropout: float = 0.1, bias: bool = True
    ):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear projections
        """
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V
        self.w_q = Linear(d_model, d_model, bias=bias)
        self.w_k = Linear(d_model, d_model, bias=bias)
        self.w_v = Linear(d_model, d_model, bias=bias)

        # Output projection
        self.w_o = Linear(d_model, d_model, bias=bias)

        # Core attention mechanism
        self.attention = ScaledDotProductAttention(dropout=dropout)

        # Add as children for parameter collection
        self.add_child(self.w_q)
        self.add_child(self.w_k)
        self.add_child(self.w_v)
        self.add_child(self.w_o)
        self.add_child(self.attention)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Apply multi-head attention.

        Args:
            query: Query tensor (batch_size, seq_len, d_model)
            key: Key tensor (batch_size, seq_len, d_model)
            value: Value tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            Output tensor and optionally attention weights
        """
        batch_size, seq_len, d_model = query.data.shape

        # Step 1: Linear projections and reshape for multiple heads
        Q = self._project_and_reshape(self.w_q(query), batch_size, seq_len)
        K = self._project_and_reshape(self.w_k(key), batch_size, seq_len)
        V = self._project_and_reshape(self.w_v(value), batch_size, seq_len)

        # Step 2: Apply attention to each head
        if return_attention:
            attn_output, attention_weights = self.attention(
                Q, K, V, mask=mask, return_attention=True
            )
        else:
            attn_output = self.attention(Q, K, V, mask=mask)
            attention_weights = None

        # Step 3: Concatenate heads and apply output projection
        output = self._concatenate_heads(attn_output, batch_size, seq_len)
        output = self.w_o(output)

        if return_attention:
            return output, attention_weights
        else:
            return output

    def _project_and_reshape(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """Project and reshape tensor for multi-head attention."""
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        # -> (batch_size, num_heads, seq_len, d_k)
        reshaped = x.data.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        transposed = np.transpose(reshaped, (0, 2, 1, 3))

        # Flatten to (batch_size * num_heads, seq_len, d_k) for attention computation
        flattened = transposed.reshape(batch_size * self.num_heads, seq_len, self.d_k)

        return Tensor(flattened, requires_grad=x.requires_grad)

    def _concatenate_heads(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """Concatenate attention heads back together."""
        # (batch_size * num_heads, seq_len, d_k) -> (batch_size, num_heads, seq_len, d_k)
        reshaped = x.data.reshape(batch_size, self.num_heads, seq_len, self.d_k)

        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, num_heads, d_k)
        transposed = np.transpose(reshaped, (0, 2, 1, 3))

        # (batch_size, seq_len, num_heads, d_k) -> (batch_size, seq_len, d_model)
        concatenated = transposed.reshape(batch_size, seq_len, self.d_model)

        return Tensor(concatenated, requires_grad=x.requires_grad)


class SelfAttention(Layer):
    """
    Self-Attention: a special case where query, key, and value are the same.

    This allows the model to relate different positions in a single sequence,
    which is crucial for understanding context and long-range dependencies.
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize self-attention layer.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.multihead_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_child(self.multihead_attn)

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None, return_attention: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Apply self-attention to input sequence.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            Output tensor and optionally attention weights
        """
        # In self-attention, query = key = value = x
        return self.multihead_attn(
            query=x, key=x, value=x, mask=mask, return_attention=return_attention
        )


class CrossAttention(Layer):
    """
    Cross-Attention: attention between two different sequences.

    Used in encoder-decoder architectures where the decoder attends to
    the encoder's output. Query comes from decoder, Key and Value from encoder.
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize cross-attention layer.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.multihead_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_child(self.multihead_attn)

    def forward(
        self,
        query: Tensor,  # From decoder
        key_value: Tensor,  # From encoder
        mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Apply cross-attention between query and key_value sequences.

        Args:
            query: Query tensor from decoder (batch_size, seq_len_q, d_model)
            key_value: Key and value tensor from encoder (batch_size, seq_len_kv, d_model)
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            Output tensor and optionally attention weights
        """
        # In cross-attention, key = value = encoder output, query = decoder input
        return self.multihead_attn(
            query=query,
            key=key_value,
            value=key_value,
            mask=mask,
            return_attention=return_attention,
        )


class CausalSelfAttention(Layer):
    """
    Causal (Masked) Self-Attention: prevents positions from attending to future positions.

    Essential for autoregressive models like GPT, where we want to predict the next
    token without "cheating" by looking at future tokens.
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize causal self-attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.multihead_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_child(self.multihead_attn)

    def forward(
        self, x: Tensor, return_attention: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Apply causal self-attention to input sequence.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            return_attention: Whether to return attention weights

        Returns:
            Output tensor and optionally attention weights
        """
        batch_size, seq_len, d_model = x.data.shape

        # Create causal mask (lower triangular matrix)
        mask = self._create_causal_mask(seq_len)

        return self.multihead_attn(
            query=x, key=x, value=x, mask=mask, return_attention=return_attention
        )

    def _create_causal_mask(self, seq_len: int) -> Tensor:
        """Create causal mask to prevent attention to future positions."""
        # Lower triangular matrix: 1s below and on diagonal, 0s above
        mask = np.tril(np.ones((seq_len, seq_len)))
        # Expand for batch dimension
        mask = np.expand_dims(mask, 0)  # (1, seq_len, seq_len)
        return Tensor(mask, requires_grad=False)


# Utility functions for attention
def create_padding_mask(sequences: Tensor, pad_token_id: int = 0) -> Tensor:
    """
    Create padding mask to ignore padded positions in variable-length sequences.

    Args:
        sequences: Input sequences (batch_size, seq_len)
        pad_token_id: Token ID used for padding

    Returns:
        Padding mask (batch_size, 1, seq_len)
    """
    # Create mask: 1 for real tokens, 0 for padding
    mask = (sequences.data != pad_token_id).astype(np.float32)

    # Add dimension for broadcasting with attention scores
    mask = np.expand_dims(mask, 1)  # (batch_size, 1, seq_len)

    return Tensor(mask, requires_grad=False)


def create_look_ahead_mask(seq_len: int) -> Tensor:
    """
    Create look-ahead mask for causal attention.

    Args:
        seq_len: Sequence length

    Returns:
        Look-ahead mask (1, seq_len, seq_len)
    """
    mask = np.tril(np.ones((seq_len, seq_len)))
    mask = np.expand_dims(mask, 0)  # Add batch dimension
    return Tensor(mask, requires_grad=False)


def attention_visualization_helper(attention_weights: Tensor, tokens: list = None):
    """
    Helper function to visualize attention weights.

    Args:
        attention_weights: Attention weights (batch_size, num_heads, seq_len, seq_len)
        tokens: Optional list of tokens for labeling

    Returns:
        Dictionary with visualization data
    """
    # Take first batch and average over heads for simplicity
    weights = attention_weights.data[0]  # (num_heads, seq_len, seq_len)
    avg_weights = np.mean(weights, axis=0)  # (seq_len, seq_len)

    viz_data = {
        "attention_matrix": avg_weights,
        "tokens": tokens or [f"Token_{i}" for i in range(avg_weights.shape[0])],
        "max_attention": np.max(avg_weights),
        "attention_entropy": -np.sum(avg_weights * np.log(avg_weights + 1e-9), axis=1),
    }

    return viz_data


# Testing and demonstration functions
def demonstrate_attention():
    """Demonstrate how attention mechanisms work with simple examples."""

    print("🔍 Attention Mechanisms Demonstration")
    print("=" * 50)

    # Create simple test data
    batch_size, seq_len, d_model = 2, 4, 8

    # Random input sequences
    np.random.seed(42)
    x = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)

    print(f"Input shape: {x.data.shape}")
    print()

    # Test Self-Attention
    print("🎯 Self-Attention:")
    self_attn = SelfAttention(d_model=d_model, num_heads=2)
    output, attention_weights = self_attn(x, return_attention=True)

    print(f"Output shape: {output.data.shape}")
    print(f"Attention weights shape: {attention_weights.data.shape}")
    print(f"Attention weights (first head, first batch):")
    print(attention_weights.data[0][:seq_len, :seq_len].round(3))
    print()

    # Test Causal Self-Attention
    print("🎯 Causal Self-Attention:")
    causal_attn = CausalSelfAttention(d_model=d_model, num_heads=2)
    causal_output, causal_weights = causal_attn(x, return_attention=True)

    print(f"Causal attention weights (first head, first batch):")
    print(causal_weights.data[0][:seq_len, :seq_len].round(3))
    print("Notice the upper triangular part is nearly zero!")
    print()

    # Test gradient flow
    print("🔄 Testing Gradient Flow:")
    loss = output.sum()
    loss.backward()

    print(f"Input gradient shape: {x.grad.shape}")
    print(f"Gradient magnitude: {np.abs(x.grad).mean():.6f}")
    print("✅ Gradients flow correctly through attention!")


if __name__ == "__main__":
    demonstrate_attention()
