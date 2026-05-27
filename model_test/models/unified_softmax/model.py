import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_common import export_quantized_model, print_generated_summary


class UnifiedSoftmaxModel(nn.Module):
    """Generate a unified Softmax case that keeps the original benchmark shape."""

    def __init__(self):
        """Instantiate the target Softmax operator on the last dimension."""
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """Apply the wrapped Softmax path on the shared transformer-style input."""
        x = x + 1e-3
        x = self.softmax(x)
        x = x - 1e-3
        return x


def main():
    """Export, quantize, and serialize the unified Softmax test artifacts."""
    model_name = "unified_softmax"
    input_shape = (1, 128, 768)
    workdir = os.path.dirname(os.path.abspath(__file__))
    export_quantized_model(
        model=UnifiedSoftmaxModel(),
        input_shape=input_shape,
        model_name=model_name,
        workdir=workdir,
        op_types_to_quantize=["Softmax"],
    )
    print_generated_summary(model_name, input_shape, ["Target op: Softmax"])


if __name__ == "__main__":
    main()
