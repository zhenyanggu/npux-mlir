import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_common import export_quantized_model, print_generated_summary


class UnifiedMatrixAddModel(nn.Module):
    """Generate a unified Add case with a fixed residual on the shared 3D shape."""

    def __init__(self, seed=2026):
        """Create a deterministic residual tensor so every run stays reproducible."""
        super().__init__()
        rng = np.random.default_rng(seed)
        residual = rng.standard_normal(size=(1, 128, 768)).astype(np.float32) * 0.125
        self.register_buffer("residual", torch.from_numpy(residual))

    def forward(self, x):
        """Add the frozen residual tensor to the runtime input."""
        return x + self.residual


def main():
    """Export, quantize, and serialize the unified matrix Add test artifacts."""
    model_name = "unified_matrix_add"
    input_shape = (1, 128, 768)
    workdir = os.path.dirname(os.path.abspath(__file__))
    export_quantized_model(
        model=UnifiedMatrixAddModel(),
        input_shape=input_shape,
        model_name=model_name,
        workdir=workdir,
        op_types_to_quantize=["Add"],
    )
    print_generated_summary(model_name, input_shape, ["Target op: Add"])


if __name__ == "__main__":
    main()
