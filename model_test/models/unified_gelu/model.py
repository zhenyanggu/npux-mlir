import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_common import export_quantized_model, print_generated_summary


class UnifiedGeluModel(nn.Module):
    """Generate a unified GELU case with the shared transformer-style shape."""

    def __init__(self):
        """Instantiate the single GELU operator under test."""
        super().__init__()
        self.gelu = nn.GELU()

    def forward(self, x):
        """Run the wrapped GELU path used by the single-op tests."""
        x = x + 1e-3
        x = self.gelu(x)
        x = x - 1e-3
        return x


def main():
    """Export, quantize, and serialize the unified GELU test artifacts."""
    model_name = "unified_gelu"
    input_shape = (1, 128, 768)
    workdir = os.path.dirname(os.path.abspath(__file__))
    export_quantized_model(
        model=UnifiedGeluModel(),
        input_shape=input_shape,
        model_name=model_name,
        workdir=workdir,
        op_types_to_quantize=["Gelu"],
    )
    print_generated_summary(model_name, input_shape, ["Target op: Gelu"])


if __name__ == "__main__":
    main()
