from __future__ import annotations

import torch

from tank_ai.src.models.cnn_dqn import CNNDQN


def test_cnn_output_shape():
    model = CNNDQN(input_channels=6, output_size=6, rows=15, cols=20)
    x = torch.zeros(4, 6, 15, 20)
    y = model(x)
    assert y.shape == (4, 6)
