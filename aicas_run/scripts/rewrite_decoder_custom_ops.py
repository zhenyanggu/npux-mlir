import argparse
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import helper, numpy_helper


DEFAULT_INPUT = "aicas_run/models/decoder_model_merged.onnx"
DEFAULT_OUTPUT = "aicas_run/models/decoder_model_merged_fp32_rewritten.onnx"


def get_attr_map(node: onnx.NodeProto) -> Dict[str, object]:
    return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}


class DecoderCustomOpRewriter:
    def __init__(self, model: onnx.ModelProto):
        self.model = model
        self.graph = model.graph
        self.initializers: Dict[str, np.ndarray] = {
            init.name: numpy_helper.to_array(init) for init in self.graph.initializer
        }
        self.producers: Dict[str, onnx.NodeProto] = {}
        self.value_shapes: Dict[str, List[object]] = {}
        self.value_dtypes: Dict[str, np.dtype] = {}
        self.const_cache: Dict[Tuple[str, bytes, Tuple[int, ...]], str] = {}
        self.counter = 0
        self.rewrite_counts: Counter[str] = Counter()

        self._build_producer_map()
        self._build_value_shape_map()
        self._build_value_dtype_map()

    def _build_producer_map(self) -> None:
        for node in self.graph.node:
            for output_name in node.output:
                if output_name:
                    self.producers[output_name] = node

    def _build_value_shape_map(self) -> None:
        values = (
            list(self.graph.input)
            + list(self.graph.output)
            + list(self.graph.value_info)
        )
        for value in values:
            tensor_type = value.type.tensor_type
            if not tensor_type.HasField("shape"):
                continue
            dims: List[object] = []
            for dim in tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    dims.append(int(dim.dim_value))
                elif dim.HasField("dim_param"):
                    dims.append(dim.dim_param)
                else:
                    dims.append("?")
            self.value_shapes[value.name] = dims

    def _build_value_dtype_map(self) -> None:
        values = (
            list(self.graph.input)
            + list(self.graph.output)
            + list(self.graph.value_info)
        )
        for value in values:
            tensor_type = value.type.tensor_type
            if not tensor_type.HasField("elem_type"):
                continue
            elem_type = int(tensor_type.elem_type)
            if elem_type == 0:
                continue
            try:
                self.value_dtypes[value.name] = np.dtype(
                    helper.tensor_dtype_to_np_dtype(elem_type)
                )
            except ValueError:
                continue

    def unique(self, stem: str) -> str:
        self.counter += 1
        return f"__rewrite_{self.counter}_{stem}"

    def add_initializer(self, array: np.ndarray, stem: str) -> str:
        array = np.asarray(array)
        key = (array.dtype.str, array.tobytes(), tuple(array.shape))
        if key in self.const_cache:
            return self.const_cache[key]
        name = self.unique(stem)
        self.graph.initializer.append(numpy_helper.from_array(array, name))
        self.initializers[name] = array
        self.const_cache[key] = name
        return name

    def make_scalar(self, value: object, dtype: np.dtype, stem: str) -> str:
        return self.add_initializer(np.array(value, dtype=dtype), stem)

    def infer_last_dim(self, value_name: str) -> int:
        if value_name in self.initializers:
            shape = self.initializers[value_name].shape
            if not shape:
                raise RuntimeError(f"cannot infer scalar dim from initializer: {value_name}")
            return int(shape[-1])

        if value_name in self.value_shapes:
            dims = self.value_shapes[value_name]
            last_dim = dims[-1]
            if isinstance(last_dim, int) and last_dim > 0:
                return last_dim

        producer = self.producers.get(value_name)
        if producer is None:
            raise RuntimeError(f"cannot infer last dim for value: {value_name}")

        if producer.op_type == "MatMul":
            rhs = producer.input[1]
            if rhs in self.initializers and self.initializers[rhs].ndim >= 2:
                return int(self.initializers[rhs].shape[-1])

        if producer.op_type == "Reshape":
            shape_name = producer.input[1]
            if shape_name in self.initializers:
                shape = self.initializers[shape_name].astype(np.int64).tolist()
                if shape[-1] > 0:
                    return int(shape[-1])
                if shape[-1] == 0:
                    return self.infer_last_dim(producer.input[0])

        if producer.op_type == "SimplifiedLayerNormalization":
            return int(self.initializers[producer.input[1]].shape[0])

        if producer.op_type == "SkipSimplifiedLayerNormalization":
            return int(self.initializers[producer.input[2]].shape[0])

        if producer.op_type == "RotaryEmbedding":
            return self.infer_last_dim(producer.input[0])

        if producer.op_type == "MultiHeadAttention":
            if producer.input[2]:
                return self.infer_last_dim(producer.input[2])
            return self.infer_last_dim(producer.input[0])

        raise RuntimeError(
            f"cannot infer last dim for value {value_name} from producer "
            f"{producer.op_type}:{producer.name}"
        )

    def infer_tensor_dtype(self, value_name: str) -> np.dtype:
        if value_name in self.initializers:
            return self.initializers[value_name].dtype

        if value_name in self.value_dtypes:
            return self.value_dtypes[value_name]

        producer = self.producers.get(value_name)
        if producer is None:
            raise RuntimeError(f"cannot infer dtype for value: {value_name}")

        same_as_input_ops = {
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Pow",
            "Sqrt",
            "MatMul",
            "Transpose",
            "Reshape",
            "Slice",
            "Concat",
            "Gather",
            "Softmax",
            "Identity",
            "SimplifiedLayerNormalization",
            "SkipSimplifiedLayerNormalization",
            "RotaryEmbedding",
            "MultiHeadAttention",
        }
        if producer.op_type in same_as_input_ops and producer.input and producer.input[0]:
            return self.infer_tensor_dtype(producer.input[0])

        if producer.op_type == "Where":
            return self.infer_tensor_dtype(producer.input[1])

        if producer.op_type == "Cast":
            attrs = get_attr_map(producer)
            to = int(attrs.get("to", 0))
            if to == 0:
                raise RuntimeError(
                    f"cannot infer Cast target dtype for value: {value_name}"
                )
            return np.dtype(helper.tensor_dtype_to_np_dtype(to))

        raise RuntimeError(
            f"cannot infer dtype for value {value_name} from producer "
            f"{producer.op_type}:{producer.name}"
        )

    def replace_model(self) -> None:
        new_nodes: List[onnx.NodeProto] = []
        for node in self.graph.node:
            replaced = self.rewrite_node(node)
            if replaced is None:
                new_nodes.append(node)
            else:
                new_nodes.extend(replaced)

        del self.graph.node[:]
        self.graph.node.extend(new_nodes)
        self._prune_unused_opsets()

    def _prune_unused_opsets(self) -> None:
        used_domains = {node.domain for node in self.graph.node if node.domain}
        if not used_domains:
            used_domains = {""}

        keep = []
        for opset in self.model.opset_import:
            if opset.domain == "" or opset.domain in used_domains:
                keep.append((opset.domain, opset.version))

        del self.model.opset_import[:]
        for domain, version in keep:
            self.model.opset_import.append(helper.make_opsetid(domain, version))

    def rewrite_node(self, node: onnx.NodeProto) -> Optional[List[onnx.NodeProto]]:
        if node.domain == "" and node.op_type == "SimplifiedLayerNormalization":
            self.rewrite_counts["SimplifiedLayerNormalization"] += 1
            return self.rewrite_simplified_layer_norm(node)
        if (
            node.domain == "com.microsoft"
            and node.op_type == "SkipSimplifiedLayerNormalization"
        ):
            self.rewrite_counts["SkipSimplifiedLayerNormalization"] += 1
            return self.rewrite_skip_simplified_layer_norm(node)
        if node.domain == "com.microsoft" and node.op_type == "RotaryEmbedding":
            self.rewrite_counts["RotaryEmbedding"] += 1
            return self.rewrite_rotary_embedding(node)
        if node.domain == "com.microsoft" and node.op_type == "MultiHeadAttention":
            self.rewrite_counts["MultiHeadAttention"] += 1
            return self.rewrite_multi_head_attention(node)
        return None

    def rewrite_simplified_layer_norm(self, node: onnx.NodeProto) -> List[onnx.NodeProto]:
        attrs = get_attr_map(node)
        axis = int(attrs.get("axis", -1))
        if axis != -1:
            raise RuntimeError(f"only axis=-1 is supported, got {axis} for {node.name}")

        x_name, scale_name = node.input
        y_name = node.output[0]
        eps_name = self.make_scalar(
            attrs.get("epsilon", 1.0e-5),
            self.initializers[scale_name].dtype,
            "ln_eps",
        )

        x2_name = self.unique("ln_x2")
        mean_name = self.unique("ln_mean")
        var_eps_name = self.unique("ln_var_eps")
        std_name = self.unique("ln_std")
        norm_name = self.unique("ln_norm")

        return [
            helper.make_node("Mul", [x_name, x_name], [x2_name], name=self.unique("Mul")),
            helper.make_node(
                "ReduceMean",
                [x2_name],
                [mean_name],
                axes=[-1],
                keepdims=1,
                name=self.unique("ReduceMean"),
            ),
            helper.make_node(
                "Add",
                [mean_name, eps_name],
                [var_eps_name],
                name=self.unique("Add"),
            ),
            helper.make_node(
                "Sqrt", [var_eps_name], [std_name], name=self.unique("Sqrt")
            ),
            helper.make_node("Div", [x_name, std_name], [norm_name], name=self.unique("Div")),
            helper.make_node(
                "Mul", [norm_name, scale_name], [y_name], name=self.unique("Mul")
            ),
        ]

    def rewrite_skip_simplified_layer_norm(
        self, node: onnx.NodeProto
    ) -> List[onnx.NodeProto]:
        attrs = get_attr_map(node)
        if len(node.input) not in (3, 4):
            raise RuntimeError(
                f"unexpected input count for SkipSimplifiedLayerNormalization: {len(node.input)}"
            )

        input_name, skip_name, gamma_name = node.input[:3]
        bias_name = node.input[3] if len(node.input) == 4 and node.input[3] else ""
        output_name = node.output[0]
        sum_output_name = (
            node.output[3] if len(node.output) >= 4 and node.output[3] else self.unique("sum")
        )

        eps_name = self.make_scalar(
            attrs.get("epsilon", 1.0e-5),
            self.initializers[gamma_name].dtype,
            "skip_ln_eps",
        )

        x2_name = self.unique("skip_ln_x2")
        mean_name = self.unique("skip_ln_mean")
        var_eps_name = self.unique("skip_ln_var_eps")
        std_name = self.unique("skip_ln_std")
        norm_name = self.unique("skip_ln_norm")

        nodes: List[onnx.NodeProto] = [
            helper.make_node(
                "Add", [input_name, skip_name], [sum_output_name], name=self.unique("Add")
            )
        ]
        norm_input_name = sum_output_name
        if bias_name:
            with_bias_name = self.unique("skip_ln_sum_bias")
            nodes.append(
                helper.make_node(
                    "Add",
                    [sum_output_name, bias_name],
                    [with_bias_name],
                    name=self.unique("Add"),
                )
            )
            norm_input_name = with_bias_name

        nodes.extend(
            [
                helper.make_node(
                    "Mul",
                    [norm_input_name, norm_input_name],
                    [x2_name],
                    name=self.unique("Mul"),
                ),
                helper.make_node(
                    "ReduceMean",
                    [x2_name],
                    [mean_name],
                    axes=[-1],
                    keepdims=1,
                    name=self.unique("ReduceMean"),
                ),
                helper.make_node(
                    "Add",
                    [mean_name, eps_name],
                    [var_eps_name],
                    name=self.unique("Add"),
                ),
                helper.make_node(
                    "Sqrt", [var_eps_name], [std_name], name=self.unique("Sqrt")
                ),
                helper.make_node(
                    "Div",
                    [norm_input_name, std_name],
                    [norm_name],
                    name=self.unique("Div"),
                ),
                helper.make_node(
                    "Mul",
                    [norm_name, gamma_name],
                    [output_name],
                    name=self.unique("Mul"),
                ),
            ]
        )
        return nodes

    def rewrite_rotary_embedding(self, node: onnx.NodeProto) -> List[onnx.NodeProto]:
        attrs = get_attr_map(node)
        if int(attrs.get("interleaved", 0)) != 0:
            raise RuntimeError(f"interleaved rotary is not supported: {node.name}")
        if int(attrs.get("rotary_embedding_dim", 0)) != 0:
            raise RuntimeError(
                f"partial rotary is not supported for this rewrite: {node.name}"
            )
        if int(attrs.get("is_packed_batching", 0)) != 0:
            raise RuntimeError(f"packed batching rotary is not supported: {node.name}")
        if float(attrs.get("scale", 1.0)) != 1.0:
            raise RuntimeError(f"scaled rotary is not supported: {node.name}")

        input_name, position_ids_name, cos_cache_name, sin_cache_name = node.input
        output_name = node.output[0]
        hidden_size = self.infer_last_dim(input_name)
        cos_cache = self.initializers.get(cos_cache_name)
        if cos_cache is None or cos_cache.ndim != 2:
            raise RuntimeError(f"cos_cache must be a 2D initializer: {cos_cache_name}")

        half_dim = int(cos_cache.shape[1])
        head_size = half_dim * 2
        num_heads = int(attrs.get("num_heads", 0)) or (hidden_size // head_size)
        if num_heads * head_size != hidden_size:
            raise RuntimeError(
                f"cannot infer num_heads/head_size for rotary node {node.name}: "
                f"hidden={hidden_size}, head_size={head_size}, num_heads={num_heads}"
            )

        shape4_name = self.add_initializer(
            np.array([0, 0, num_heads, head_size], dtype=np.int64), "rope_shape4"
        )
        shape_cos_name = self.add_initializer(
            np.array([0, 0, 1, half_dim], dtype=np.int64), "rope_shape_cos"
        )
        shape3_name = self.add_initializer(
            np.array([0, 0, hidden_size], dtype=np.int64), "rope_shape3"
        )
        starts1_name = self.add_initializer(np.array([0], dtype=np.int64), "rope_starts1")
        ends1_name = self.add_initializer(
            np.array([half_dim], dtype=np.int64), "rope_ends1"
        )
        starts2_name = self.add_initializer(
            np.array([half_dim], dtype=np.int64), "rope_starts2"
        )
        ends2_name = self.add_initializer(
            np.array([head_size], dtype=np.int64), "rope_ends2"
        )
        axes_name = self.add_initializer(np.array([-1], dtype=np.int64), "rope_axes")

        x4_name = self.unique("rope_x4")
        x1_name = self.unique("rope_x1")
        x2_name = self.unique("rope_x2")
        cos_gather_name = self.unique("rope_cos_gather")
        sin_gather_name = self.unique("rope_sin_gather")
        cos4_name = self.unique("rope_cos4")
        sin4_name = self.unique("rope_sin4")
        cx1_name = self.unique("rope_cx1")
        sx2_name = self.unique("rope_sx2")
        real_name = self.unique("rope_real")
        sx1_name = self.unique("rope_sx1")
        cx2_name = self.unique("rope_cx2")
        imag_name = self.unique("rope_imag")
        y4_name = self.unique("rope_y4")

        return [
            helper.make_node(
                "Reshape",
                [input_name, shape4_name],
                [x4_name],
                allowzero=0,
                name=self.unique("Reshape"),
            ),
            helper.make_node(
                "Slice",
                [x4_name, starts1_name, ends1_name, axes_name],
                [x1_name],
                name=self.unique("Slice"),
            ),
            helper.make_node(
                "Slice",
                [x4_name, starts2_name, ends2_name, axes_name],
                [x2_name],
                name=self.unique("Slice"),
            ),
            helper.make_node(
                "Gather",
                [cos_cache_name, position_ids_name],
                [cos_gather_name],
                axis=0,
                name=self.unique("Gather"),
            ),
            helper.make_node(
                "Gather",
                [sin_cache_name, position_ids_name],
                [sin_gather_name],
                axis=0,
                name=self.unique("Gather"),
            ),
            helper.make_node(
                "Reshape",
                [cos_gather_name, shape_cos_name],
                [cos4_name],
                allowzero=0,
                name=self.unique("Reshape"),
            ),
            helper.make_node(
                "Reshape",
                [sin_gather_name, shape_cos_name],
                [sin4_name],
                allowzero=0,
                name=self.unique("Reshape"),
            ),
            helper.make_node(
                "Mul", [cos4_name, x1_name], [cx1_name], name=self.unique("Mul")
            ),
            helper.make_node(
                "Mul", [sin4_name, x2_name], [sx2_name], name=self.unique("Mul")
            ),
            helper.make_node(
                "Sub", [cx1_name, sx2_name], [real_name], name=self.unique("Sub")
            ),
            helper.make_node(
                "Mul", [sin4_name, x1_name], [sx1_name], name=self.unique("Mul")
            ),
            helper.make_node(
                "Mul", [cos4_name, x2_name], [cx2_name], name=self.unique("Mul")
            ),
            helper.make_node(
                "Add", [sx1_name, cx2_name], [imag_name], name=self.unique("Add")
            ),
            helper.make_node(
                "Concat",
                [real_name, imag_name],
                [y4_name],
                axis=-1,
                name=self.unique("Concat"),
            ),
            helper.make_node(
                "Reshape",
                [y4_name, shape3_name],
                [output_name],
                allowzero=0,
                name=self.unique("Reshape"),
            ),
        ]

    def rewrite_multi_head_attention(self, node: onnx.NodeProto) -> List[onnx.NodeProto]:
        attrs = get_attr_map(node)
        if len(node.input) < 6:
            raise RuntimeError(f"unexpected MHA input count: {len(node.input)}")

        query_name = node.input[0]
        key_name = node.input[1]
        value_name = node.input[2]
        bias_name = node.input[3] if len(node.input) > 3 else ""
        key_padding_mask_name = node.input[4] if len(node.input) > 4 else ""
        attention_bias_name = node.input[5] if len(node.input) > 5 else ""
        past_key_name = node.input[6] if len(node.input) > 6 else ""
        past_value_name = node.input[7] if len(node.input) > 7 else ""
        past_sequence_length_name = node.input[8] if len(node.input) > 8 else ""
        cache_indirection_name = node.input[9] if len(node.input) > 9 else ""

        unsupported_inputs = {
            "bias": bias_name,
            "key_padding_mask": key_padding_mask_name,
            "past_key": past_key_name,
            "past_value": past_value_name,
            "past_sequence_length": past_sequence_length_name,
            "cache_indirection": cache_indirection_name,
        }
        for field, name in unsupported_inputs.items():
            if name:
                raise RuntimeError(f"MHA {field} is not supported in rewrite: {node.name}")

        if len(node.output) > 1 and any(output_name for output_name in node.output[1:]):
            raise RuntimeError(f"MHA present/qk outputs are not supported: {node.name}")

        output_name = node.output[0]
        num_heads = int(attrs["num_heads"])
        hidden_size = self.infer_last_dim(query_name)
        if hidden_size % num_heads != 0:
            raise RuntimeError(
                f"invalid head layout for MHA {node.name}: hidden={hidden_size}, "
                f"num_heads={num_heads}"
            )
        head_size = hidden_size // num_heads
        scale = float(attrs.get("scale", 1.0 / np.sqrt(head_size)))

        shape4_name = self.add_initializer(
            np.array([0, 0, num_heads, head_size], dtype=np.int64), "mha_shape4"
        )
        shape3_name = self.add_initializer(
            np.array([0, 0, hidden_size], dtype=np.int64), "mha_shape3"
        )
        scale_name = self.make_scalar(scale, self.infer_tensor_dtype(query_name), "mha_scale")

        q_reshape_name = self.unique("mha_qr")
        k_reshape_name = self.unique("mha_kr")
        v_reshape_name = self.unique("mha_vr")
        q_transpose_name = self.unique("mha_qt")
        k_transpose_name = self.unique("mha_kt")
        v_transpose_name = self.unique("mha_vt")
        k_transpose_t_name = self.unique("mha_ktt")
        score_name = self.unique("mha_scores")
        scaled_name = self.unique("mha_scaled")
        masked_name = self.unique("mha_masked")
        prob_name = self.unique("mha_prob")
        ctx_name = self.unique("mha_ctx")
        ctx_transpose_name = self.unique("mha_ctx_t")

        nodes = [
            helper.make_node(
                "Reshape",
                [query_name, shape4_name],
                [q_reshape_name],
                allowzero=0,
                name=self.unique("Reshape"),
            ),
            helper.make_node(
                "Reshape",
                [key_name, shape4_name],
                [k_reshape_name],
                allowzero=0,
                name=self.unique("Reshape"),
            ),
            helper.make_node(
                "Reshape",
                [value_name, shape4_name],
                [v_reshape_name],
                allowzero=0,
                name=self.unique("Reshape"),
            ),
            helper.make_node(
                "Transpose",
                [q_reshape_name],
                [q_transpose_name],
                perm=[0, 2, 1, 3],
                name=self.unique("Transpose"),
            ),
            helper.make_node(
                "Transpose",
                [k_reshape_name],
                [k_transpose_name],
                perm=[0, 2, 1, 3],
                name=self.unique("Transpose"),
            ),
            helper.make_node(
                "Transpose",
                [v_reshape_name],
                [v_transpose_name],
                perm=[0, 2, 1, 3],
                name=self.unique("Transpose"),
            ),
            helper.make_node(
                "Transpose",
                [k_transpose_name],
                [k_transpose_t_name],
                perm=[0, 1, 3, 2],
                name=self.unique("Transpose"),
            ),
            helper.make_node(
                "MatMul",
                [q_transpose_name, k_transpose_t_name],
                [score_name],
                name=self.unique("MatMul"),
            ),
            helper.make_node(
                "Mul",
                [score_name, scale_name],
                [scaled_name],
                name=self.unique("Mul"),
            ),
        ]

        softmax_input_name = scaled_name
        if attention_bias_name:
            nodes.append(
                helper.make_node(
                    "Add",
                    [scaled_name, attention_bias_name],
                    [masked_name],
                    name=self.unique("Add"),
                )
            )
            softmax_input_name = masked_name

        nodes.extend(
            [
                helper.make_node(
                    "Softmax",
                    [softmax_input_name],
                    [prob_name],
                    axis=-1,
                    name=self.unique("Softmax"),
                ),
                helper.make_node(
                    "MatMul",
                    [prob_name, v_transpose_name],
                    [ctx_name],
                    name=self.unique("MatMul"),
                ),
                helper.make_node(
                    "Transpose",
                    [ctx_name],
                    [ctx_transpose_name],
                    perm=[0, 2, 1, 3],
                    name=self.unique("Transpose"),
                ),
                helper.make_node(
                    "Reshape",
                    [ctx_transpose_name, shape3_name],
                    [output_name],
                    allowzero=0,
                    name=self.unique("Reshape"),
                ),
            ]
        )
        return nodes


def count_custom_nodes(model: onnx.ModelProto) -> Counter[str]:
    counter: Counter[str] = Counter()
    for node in model.graph.node:
        if node.domain == "com.microsoft" and node.op_type in {
            "SkipSimplifiedLayerNormalization",
            "RotaryEmbedding",
            "MultiHeadAttention",
        }:
            counter[node.op_type] += 1
        if node.domain == "" and node.op_type == "SimplifiedLayerNormalization":
            counter[node.op_type] += 1
    return counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite decoder custom ops into standard ONNX subgraphs."
    )
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip onnx.checker.check_model on the rewritten graph.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = onnx.load(args.input)
    before = count_custom_nodes(model)
    print(f"rewrite 前 custom 统计: {dict(before)}")

    rewriter = DecoderCustomOpRewriter(model)
    rewriter.replace_model()

    after = count_custom_nodes(model)
    print(f"rewrite 后 custom 统计: {dict(after)}")
    print(f"本次重写数量: {dict(rewriter.rewrite_counts)}")

    if not args.skip_check:
        onnx.checker.check_model(model)

    onnx.save(model, args.output)
    print(f"已保存重写后的模型: {args.output}")


if __name__ == "__main__":
    main()
