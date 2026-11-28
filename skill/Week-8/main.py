import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(query, key, value, causal=False):
    """Compute scaled dot-product attention.

    The tensors use (..., sequence_length, feature_size) dimensions, so this
    function works with both a single attention head and batched multi-head
    inputs.
    """
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(query.size(-1))

    if causal:
        query_length = query.size(-2)
        key_length = key.size(-2)
        # A query position may attend only to the same or earlier key positions.
        mask = torch.triu(
            torch.ones(
                query_length,
                key_length,
                device=query.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))

    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention built from linear layers and PyTorch ops."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError("embed_dim and num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.W_O = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, causal=False):
        if x.ndim != 3:
            raise ValueError("x must have shape (batch_size, seq_len, embed_dim)")
        batch_size, seq_len, embed_dim = x.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embed_dim}, got {embed_dim}"
            )

        def split_heads(projection):
            return projection.view(
                batch_size, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)

        query = split_heads(self.W_Q(x))
        key = split_heads(self.W_K(x))
        value = split_heads(self.W_V(x))

        attention_output, attention_weights = scaled_dot_product_attention(
            query, key, value, causal=causal
        )

        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        output = self.W_O(attention_output)
        return output, attention_weights


def main():
    # Scaled dot-product attention example
    query = torch.randn(1, 3, 4)
    key = torch.randn(1, 3, 4)
    value = torch.randn(1, 3, 4)

    output, weights = scaled_dot_product_attention(query, key, value)
    print("Scaled Dot-Product Attention")
    print("--------------------------------")
    print("Query Shape:", query.shape)
    print("Key Shape:", key.shape)
    print("Value Shape:", value.shape)
    print("\nAttention Weights:")
    print(weights)
    print("\nAttention Output:")
    print(output)
    print("\nOutput Shape:", output.shape)

    # Multi-head self-attention example
    print("\n\nMulti-Head Attention")
    print("--------------------------------")
    embed_dim = 8
    num_heads = 2
    x = torch.randn(1, 3, embed_dim)
    attention = MultiHeadAttention(embed_dim, num_heads)
    mha_output, mha_weights = attention(x)

    print("Input Shape:", x.shape)
    print("Number of Heads:", num_heads)
    print("Attention Weights Shape:", mha_weights.shape)
    print("\nMulti-Head Attention Output:")
    print(mha_output)
    print("\nOutput Shape:", mha_output.shape)


if __name__ == "__main__":
    main()
