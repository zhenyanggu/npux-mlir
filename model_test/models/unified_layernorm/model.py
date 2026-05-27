import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_common import export_quantized_model, print_generated_summary


class UnifiedLayerNormModel(nn.Module):
    """Generate a unified LayerNorm case over the shared hidden size 768."""

    def __init__(self):
        """Instantiate one LayerNorm operator with the benchmark hidden width."""
        super().__init__()
        self.layer_norm = nn.LayerNorm(768, eps=1e-5)

    def forward(self, x):
        """Run the wrapped LayerNorm path used by the single-op test."""
        x = x + 1e-3
        x = self.layer_norm(x)
        x = x - 1e-3
        return x


def main():
    """Export, quantize, and serialize the unified LayerNorm test artifacts."""
    model_name = "unified_layernorm"
    input_shape = (1, 128, 768)
    workdir = os.path.dirname(os.path.abspath(__file__))
    export_quantized_model(
        model=UnifiedLayerNormModel(),
        input_shape=input_shape,
        model_name=model_name,
        workdir=workdir,
        op_types_to_quantize=["LayerNormalization"],
    )
    print_generated_summary(model_name, input_shape, ["Target op: LayerNormalization"])


if __name__ == "__main__":
    main()
