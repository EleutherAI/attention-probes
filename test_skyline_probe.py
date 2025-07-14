#!/usr/bin/env python3
"""
Simple test script for the SkylineProbe.
"""

import torch
import numpy as np
from attention_probe.skyline_probe import SkylineProbe
from attention_probe.trainer import MulticlassTrainConfig, TrainingData


def test_skyline_probe():
    """Test that the SkylineProbe works correctly."""
    
    # Create dummy data
    batch_size = 4
    seq_len = 10
    hidden_dim = 64
    n_classes = 2
    
    # Create random input data
    x = torch.randn(batch_size, seq_len, hidden_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    position = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    y = torch.randint(0, n_classes, (batch_size,))
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Create training data
    train_data = TrainingData(
        x=x,
        mask=mask,
        position=position,
        y=y,
        input_ids=input_ids,
        n_classes=n_classes
    )
    
    # Create config
    config = MulticlassTrainConfig(
        train_skyline=True,
        n_heads=4,
        hidden_dim=32,
        learning_rate=1e-4,
        train_iterations=10,  # Small number for testing
        batch_size=batch_size
    )
    
    # Create skyline probe
    probe = SkylineProbe(
        d_in=hidden_dim,
        n_heads=config.n_heads,
        output_dim=n_classes,
        config=config
    )
    
    print(f"SkylineProbe created successfully:")
    print(f"  - Input dimension: {hidden_dim}")
    print(f"  - Number of heads: {config.n_heads}")
    print(f"  - Output dimension: {n_classes}")
    print(f"  - Number of layers: {probe.n_layers}")
    print(f"  - MLP width: {probe.d_ff}")
    print(f"  - Number of parameters: {sum(p.numel() for p in probe.parameters())}")
    
    # Test forward pass
    probe.eval()
    with torch.no_grad():
        output = probe(x, mask, position)
        print(f"  - Output shape: {output.shape}")
        print(f"  - Expected shape: ({batch_size}, {n_classes})")
        assert output.shape == (batch_size, n_classes), f"Expected shape ({batch_size}, {n_classes}), got {output.shape}"
    
    # Test training compatibility
    device = torch.device("cpu")
    probe = probe.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.learning_rate)
    
    # Test training step
    probe.train()
    optimizer.zero_grad()
    output = probe(x, mask, position)
    loss = torch.nn.functional.cross_entropy(output, y)
    loss.backward()
    optimizer.step()
    
    print(f"  - Training step completed successfully")
    print(f"  - Loss: {loss.item():.4f}")
    
    print("✅ All tests passed!")


if __name__ == "__main__":
    test_skyline_probe() 