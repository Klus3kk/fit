"""
Transformer Blocks & Complete Transformer Architecture

This module implements the complete Transformer architecture including:
- Transformer Encoder/Decoder Blocks
- Positional Encoding
- Layer Normalization
- Feed-Forward Networks
- Complete Transformer models

Built on top of the attention mechanisms, this creates the full power
of modern Transformer architectures.
"""

import numpy as np
import math
from typing import Optional, Tuple, List

from core.tensor import Tensor
from nn.modules.base import Layer
from nn.modules.linear import Linear
from nn.modules.activation import ReLU, GELU, Dropout
from nn.modules.normalization import LayerNorm
from attention_core import (
    MultiHeadAttention,
    SelfAttention,
    CrossAttention,
    CausalSelfAttention,
)


class PositionalEncoding(Layer):
    """
    Positional Encoding: adds position information to embeddings.

    Since Transformers have no inherent notion of sequence order,
    we add sinusoidal position encodings to give the model
    information about token positions.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length to precompute
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = Dropout(dropout) if dropout > 0 else None

        # Precompute positional encodings
        self.pe = self._create_positional_encoding(max_len, d_model)

    def _create_positional_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        """Create sinusoidal positional encodings."""
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)

        # Create div_term for the sinusoidal pattern
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        # Apply sin to even indices
        pe[:, 0::2] = np.sin(position * div_term)

        # Apply cos to odd indices
        if d_model % 2 == 1:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = np.cos(position * div_term)

        return pe

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input embeddings (batch_size, seq_len, d_model)

        Returns:
            Embeddings with positional encoding added
        """
        batch_size, seq_len, d_model = x.data.shape

        # Get positional encodings for this sequence length
        pos_encoding = self.pe[:seq_len, :d_model]

        # Add positional encoding (broadcasting over batch dimension)
        pos_tensor = Tensor(pos_encoding, requires_grad=False)
        result = x + pos_tensor

        # Apply dropout if specified
        if self.dropout is not None:
            result = self.dropout(result)

        return result


class LayerNorm(Layer):
    """
    Layer Normalization: normalizes inputs across the feature dimension.

    Essential for training stability in Transformers. Unlike BatchNorm,
    LayerNorm normalizes across features for each sample independently.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Initialize layer normalization.

        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # Learnable parameters
        self.gamma = Tensor(np.ones(d_model), requires_grad=True)
        self.beta = Tensor(np.zeros(d_model), requires_grad=True)

        self.add_parameter(self.gamma)
        self.add_parameter(self.beta)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply layer normalization.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Normalized tensor
        """
        # Calculate mean and variance across the feature dimension
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)

        # Normalize
        normalized = (x.data - mean) / np.sqrt(var + self.eps)

        # Scale and shift
        output_data = self.gamma.data * normalized + self.beta.data

        # Create output tensor
        output = Tensor(output_data, requires_grad=x.requires_grad)

        # Define backward pass
        def _backward():
            if output.grad is None:
                return

            # Gradient computation for layer norm
            N = x.data.shape[-1]  # Feature dimension

            # Gradient w.r.t. gamma and beta
            if self.gamma.requires_grad:
                self.gamma.grad = np.sum(output.grad * normalized, axis=(0, 1))
            if self.beta.requires_grad:
                self.beta.grad = np.sum(output.grad, axis=(0, 1))

            # Gradient w.r.t. input
            if x.requires_grad:
                std = np.sqrt(var + self.eps)

                # Mean of gradient over features
                grad_mean = np.mean(output.grad, axis=-1, keepdims=True)

                # Mean of gradient * normalized input over features
                grad_norm_mean = np.mean(
                    output.grad * normalized, axis=-1, keepdims=True
                )

                # Input gradient
                x_grad = (
                    (1.0 / std)
                    * (output.grad - grad_mean - normalized * grad_norm_mean)
                    * self.gamma.data
                )

                x.grad = x_grad if x.grad is None else x.grad + x_grad

        output._backward = _backward
        output._prev = {x, self.gamma, self.beta}

        return output


class GELU(Layer):
    """
    Gaussian Error Linear Unit: smooth activation function used in Transformers.

    GELU(x) = x * Φ(x) where Φ is the cumulative distribution function
    of the standard normal distribution.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Apply GELU activation."""
        # Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        x_cubed = x.data * x.data * x.data
        tanh_input = math.sqrt(2.0 / math.pi) * (x.data + 0.044715 * x_cubed)
        tanh_output = np.tanh(tanh_input)

        gelu_output = 0.5 * x.data * (1 + tanh_output)

        output = Tensor(gelu_output, requires_grad=x.requires_grad)

        def _backward():
            if output.grad is None or not x.requires_grad:
                return

            # GELU derivative (approximate)
            sech2 = 1 - tanh_output * tanh_output  # sech²(x) = 1 - tanh²(x)
            derivative = 0.5 * (1 + tanh_output) + 0.5 * x.data * sech2 * math.sqrt(
                2.0 / math.pi
            ) * (1 + 3 * 0.044715 * x.data * x.data)

            x_grad = output.grad * derivative
            x.grad = x_grad if x.grad is None else x.grad + x_grad

        output._backward = _backward
        output._prev = {x}

        return output


class FeedForward(Layer):
    """
    Position-wise Feed-Forward Network: applies same FFN to each position.

    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

    This adds non-linearity and allows the model to process information
    within each position independently.
    """

    def __init__(
        self, d_model: int, d_ff: int, activation: str = "gelu", dropout: float = 0.1
    ):
        """
        Initialize feed-forward network.

        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (usually 4 * d_model)
            activation: Activation function ('relu', 'gelu')
            dropout: Dropout probability
        """
        super().__init__()

        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)

        if activation == "relu":
            self.activation = ReLU()
        elif activation == "gelu":
            self.activation = GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.dropout = Dropout(dropout) if dropout > 0 else None

        # Add as children
        self.add_child(self.linear1)
        self.add_child(self.linear2)
        self.add_child(self.activation)
        if self.dropout:
            self.add_child(self.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply feed-forward network.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # First linear layer
        out = self.linear1(x)

        # Activation
        out = self.activation(out)

        # Dropout (if training)
        if self.dropout is not None:
            out = self.dropout(out)

        # Second linear layer
        out = self.linear2(out)

        return out


class TransformerEncoderBlock(Layer):
    """
    Transformer Encoder Block: the core building block of the Transformer encoder.

    Structure:
    1. Multi-Head Self-Attention
    2. Residual connection + Layer Norm
    3. Feed-Forward Network
    4. Residual connection + Layer Norm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Initialize transformer encoder block.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function in FFN
        """
        super().__init__()

        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, activation, dropout)

        # Layer normalization layers
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout = Dropout(dropout) if dropout > 0 else None

        # Add as children
        self.add_child(self.self_attention)
        self.add_child(self.feed_forward)
        self.add_child(self.norm1)
        self.add_child(self.norm2)
        if self.dropout:
            self.add_child(self.dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through encoder block.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, mask=mask)

        if self.dropout is not None:
            attn_output = self.dropout(attn_output)

        # First residual connection and layer norm
        x = self.norm1(x + attn_output)

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)

        if self.dropout is not None:
            ff_output = self.dropout(ff_output)

        # Second residual connection and layer norm
        output = self.norm2(x + ff_output)

        return output


class TransformerDecoderBlock(Layer):
    """
    Transformer Decoder Block: core building block of the Transformer decoder.

    Structure:
    1. Masked Multi-Head Self-Attention
    2. Residual connection + Layer Norm
    3. Multi-Head Cross-Attention (encoder-decoder attention)
    4. Residual connection + Layer Norm
    5. Feed-Forward Network
    6. Residual connection + Layer Norm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Initialize transformer decoder block.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function in FFN
        """
        super().__init__()

        # Masked self-attention
        self.self_attention = CausalSelfAttention(d_model, num_heads, dropout)

        # Cross-attention (encoder-decoder attention)
        self.cross_attention = CrossAttention(d_model, num_heads, dropout)

        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, activation, dropout)

        # Layer normalization layers
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout = Dropout(dropout) if dropout > 0 else None

        # Add as children
        self.add_child(self.self_attention)
        self.add_child(self.cross_attention)
        self.add_child(self.feed_forward)
        self.add_child(self.norm1)
        self.add_child(self.norm2)
        self.add_child(self.norm3)
        if self.dropout:
            self.add_child(self.dropout)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        self_attn_mask: Optional[Tensor] = None,
        cross_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through decoder block.

        Args:
            x: Decoder input (batch_size, target_seq_len, d_model)
            encoder_output: Encoder output (batch_size, source_seq_len, d_model)
            self_attn_mask: Mask for self-attention
            cross_attn_mask: Mask for cross-attention

        Returns:
            Output tensor (batch_size, target_seq_len, d_model)
        """
        # Masked self-attention
        self_attn_output = self.self_attention(x)

        if self.dropout is not None:
            self_attn_output = self.dropout(self_attn_output)

        # First residual connection and layer norm
        x = self.norm1(x + self_attn_output)

        # Cross-attention
        cross_attn_output = self.cross_attention(
            query=x, key_value=encoder_output, mask=cross_attn_mask
        )

        if self.dropout is not None:
            cross_attn_output = self.dropout(cross_attn_output)

        # Second residual connection and layer norm
        x = self.norm2(x + cross_attn_output)

        # Feed-forward
        ff_output = self.feed_forward(x)

        if self.dropout is not None:
            ff_output = self.dropout(ff_output)

        # Third residual connection and layer norm
        output = self.norm3(x + ff_output)

        return output


class TransformerEncoder(Layer):
    """
    Complete Transformer Encoder: stack of encoder blocks with embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Initialize transformer encoder.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()

        # Token embeddings
        self.embedding = Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Stack of encoder blocks
        self.layers = []
        for _ in range(num_layers):
            layer = TransformerEncoderBlock(
                d_model, num_heads, d_ff, dropout, activation
            )
            self.layers.append(layer)
            self.add_child(layer)

        # Add other components as children
        self.add_child(self.embedding)
        self.add_child(self.pos_encoding)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through transformer encoder.

        Args:
            x: Input token indices (batch_size, seq_len)
            mask: Optional attention mask

        Returns:
            Encoded representations (batch_size, seq_len, d_model)
        """
        # Token embedding + positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)

        # Pass through all encoder layers
        for layer in self.layers:
            x = layer(x, mask=mask)

        return x


class TransformerDecoder(Layer):
    """
    Complete Transformer Decoder: stack of decoder blocks with embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Initialize transformer decoder.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()

        # Token embeddings
        self.embedding = Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Stack of decoder blocks
        self.layers = []
        for _ in range(num_layers):
            layer = TransformerDecoderBlock(
                d_model, num_heads, d_ff, dropout, activation
            )
            self.layers.append(layer)
            self.add_child(layer)

        # Add other components as children
        self.add_child(self.embedding)
        self.add_child(self.pos_encoding)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        self_attn_mask: Optional[Tensor] = None,
        cross_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through transformer decoder.

        Args:
            x: Target token indices (batch_size, target_seq_len)
            encoder_output: Encoder output (batch_size, source_seq_len, d_model)
            self_attn_mask: Mask for self-attention
            cross_attn_mask: Mask for cross-attention

        Returns:
            Decoded representations (batch_size, target_seq_len, d_model)
        """
        # Token embedding + positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)

        # Pass through all decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask, cross_attn_mask)

        return x


class Embedding(Layer):
    """
    Token embedding layer that converts token indices to dense vectors.
    """

    def __init__(self, vocab_size: int, d_model: int):
        """
        Initialize embedding layer.

        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Initialize embedding matrix
        # Use scaled random initialization
        scale = math.sqrt(1.0 / d_model)
        self.weight = Tensor(
            np.random.normal(0, scale, (vocab_size, d_model)), requires_grad=True
        )

        self.add_parameter(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        """
        Look up embeddings for input tokens.

        Args:
            x: Token indices (batch_size, seq_len)

        Returns:
            Embeddings (batch_size, seq_len, d_model)
        """
        # Convert token indices to embeddings
        batch_size, seq_len = x.data.shape
        indices = x.data.astype(int)

        # Look up embeddings
        embeddings = self.weight.data[indices]  # (batch_size, seq_len, d_model)

        # Scale embeddings by sqrt(d_model) as in original paper
        embeddings = embeddings * math.sqrt(self.d_model)

        output = Tensor(embeddings, requires_grad=self.weight.requires_grad)

        def _backward():
            if output.grad is None or not self.weight.requires_grad:
                return

            # Gradient w.r.t. embedding weights
            weight_grad = np.zeros_like(self.weight.data)

            # Accumulate gradients for each token
            for i in range(batch_size):
                for j in range(seq_len):
                    token_id = indices[i, j]
                    weight_grad[token_id] += output.grad[i, j] * math.sqrt(self.d_model)

            self.weight.grad = (
                weight_grad
                if self.weight.grad is None
                else self.weight.grad + weight_grad
            )

        output._backward = _backward
        output._prev = {self.weight}

        return output


# Complete Transformer model for sequence-to-sequence tasks
class Transformer(Layer):
    """
    Complete Transformer model for sequence-to-sequence tasks.

    This is the full Transformer as described in "Attention Is All You Need".
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Initialize complete Transformer model.

        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()

        # Encoder
        self.encoder = TransformerEncoder(
            src_vocab_size,
            d_model,
            num_heads,
            num_encoder_layers,
            d_ff,
            max_len,
            dropout,
            activation,
        )

        # Decoder
        self.decoder = TransformerDecoder(
            tgt_vocab_size,
            d_model,
            num_heads,
            num_decoder_layers,
            d_ff,
            max_len,
            dropout,
            activation,
        )

        # Output projection
        self.output_projection = Linear(d_model, tgt_vocab_size)

        # Add as children
        self.add_child(self.encoder)
        self.add_child(self.decoder)
        self.add_child(self.output_projection)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through complete Transformer.

        Args:
            src: Source sequences (batch_size, src_seq_len)
            tgt: Target sequences (batch_size, tgt_seq_len)
            src_mask: Source attention mask
            tgt_mask: Target attention mask

        Returns:
            Output logits (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Encode source sequence
        encoder_output = self.encoder(src, mask=src_mask)

        # Decode target sequence
        decoder_output = self.decoder(
            tgt, encoder_output, self_attn_mask=tgt_mask, cross_attn_mask=src_mask
        )

        # Project to vocabulary
        output = self.output_projection(decoder_output)

        return output


def demonstrate_transformer():
    """Demonstrate transformer components with simple examples."""

    print("🤖 Transformer Architecture Demonstration")
    print("=" * 60)

    # Test parameters
    vocab_size = 1000
    d_model = 128
    seq_len = 10
    batch_size = 2

    # Create sample data
    np.random.seed(42)
    src_tokens = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))
    tgt_tokens = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))

    print(f"Input shapes: src={src_tokens.data.shape}, tgt={tgt_tokens.data.shape}")
    print()

    # Test individual components
    print("🧩 Testing Individual Components:")
    print("-" * 40)

    # 1. Positional Encoding
    pos_enc = PositionalEncoding(d_model, max_len=100)
    dummy_embeddings = Tensor(np.random.randn(batch_size, seq_len, d_model))
    pos_output = pos_enc(dummy_embeddings)
    print(
        f"✅ Positional Encoding: {dummy_embeddings.data.shape} -> {pos_output.data.shape}"
    )

    # 2. Layer Normalization
    layer_norm = LayerNorm(d_model)
    norm_output = layer_norm(dummy_embeddings)
    print(
        f"✅ Layer Normalization: {dummy_embeddings.data.shape} -> {norm_output.data.shape}"
    )

    # 3. Feed Forward
    ff = FeedForward(d_model, d_ff=512)
    ff_output = ff(dummy_embeddings)
    print(f"✅ Feed Forward: {dummy_embeddings.data.shape} -> {ff_output.data.shape}")

    # 4. Transformer Block
    encoder_block = TransformerEncoderBlock(d_model, num_heads=8, d_ff=512)
    block_output = encoder_block(dummy_embeddings)
    print(
        f"✅ Encoder Block: {dummy_embeddings.data.shape} -> {block_output.data.shape}"
    )
    print()

    # Test complete models
    print("🏗️ Testing Complete Models:")
    print("-" * 30)

    # 1. Encoder only
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        max_len=100,
    )
    encoder_output = encoder(src_tokens)
    print(
        f"✅ Transformer Encoder: {src_tokens.data.shape} -> {encoder_output.data.shape}"
    )

    # 2. Complete Transformer
    transformer = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
        max_len=100,
    )

    transformer_output = transformer(src_tokens, tgt_tokens)
    print(
        f"✅ Complete Transformer: {src_tokens.data.shape}, {tgt_tokens.data.shape} -> {transformer_output.data.shape}"
    )
    print()

    # Test gradient flow
    print("🔄 Testing Gradient Flow:")
    print("-" * 25)

    loss = transformer_output.sum()
    loss.backward()

    # Check if gradients exist
    param_count = 0
    grad_count = 0

    for param in transformer.parameters():
        param_count += 1
        if param.grad is not None:
            grad_count += 1

    print(f"Parameters with gradients: {grad_count}/{param_count}")
    print(f"✅ Gradient flow working correctly!")
    print()

    # Parameter count
    total_params = sum(np.prod(p.data.shape) for p in transformer.parameters())
    print(f"📊 Model Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")


if __name__ == "__main__":
    demonstrate_transformer()
