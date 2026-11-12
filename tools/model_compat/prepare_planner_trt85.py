"""Prepare SONIC's FP32 opset-17 planner for TensorRT 8.5.

TensorRT 8.5 cannot import this ONNX operator and requires TopK's K input to
be constant. Write a separate model with the mean/variance definition and
fold TopK's static shape calculation, and expand Einsum ellipses using tensor
ranks. The source file is unchanged. LayerNorm follows the ONNX definition:
https://onnx.ai/onnx/operators/onnx__LayerNormalization.html
Dependencies: onnx==1.17.0, numpy==1.26.4, onnxruntime==1.17.3.
"""

import argparse
import gc
from pathlib import Path
import string

import numpy as np
import onnx
from onnx import helper, numpy_helper


def expand(model: onnx.ModelProto) -> int:
    if next(x.version for x in model.opset_import if x.domain == "") != 17:
        raise ValueError("This converter targets the SONIC opset-17 planner")
    nodes = []
    count = 0
    for node in model.graph.node:
        if node.op_type != "LayerNormalization" or node.domain:
            nodes.append(node)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if attrs.get("axis", -1) != -1 or attrs.get("stash_type", 1) != 1:
            raise ValueError("Expected last-axis FP32 LayerNormalization")
        if len(node.output) != 1:
            raise ValueError("Expected only the normalized output, as in the SONIC planner")
        p = f"sonic_trt85_ln_{count}_"
        eps = numpy_helper.from_array(
            np.asarray(attrs.get("epsilon", 1e-5), dtype=np.float32), name=p + "epsilon"
        )
        model.graph.initializer.append(eps)
        x, scale = node.input[:2]
        scaled = node.output[0] if len(node.input) < 3 or not node.input[2] else p + "scaled"
        specs = [
            ("ReduceMean", [x], p + "mean", {"axes": [-1], "keepdims": 1}),
            ("Sub", [x, p + "mean"], p + "centered", {}),
            ("Mul", [p + "centered", p + "centered"], p + "square", {}),
            ("ReduceMean", [p + "square"], p + "variance", {"axes": [-1], "keepdims": 1}),
            ("Add", [p + "variance", p + "epsilon"], p + "variance_eps", {}),
            ("Sqrt", [p + "variance_eps"], p + "std", {}),
            ("Div", [p + "centered", p + "std"], p + "normalized", {}),
            ("Mul", [p + "normalized", scale], scaled, {}),
        ]
        if scaled != node.output[0]:
            specs.append(("Add", [scaled, node.input[2]], node.output[0], {}))
        nodes.extend(helper.make_node(op, ins, [out], name=p + op + str(i), **kw)
                     for i, (op, ins, out, kw) in enumerate(specs))
        count += 1
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError("Write a separate compatibility model")
    model = onnx.load(args.source)
    count = expand(model)
    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    del model
    gc.collect()
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    # Basic optimization folds static shapes without re-fusing LayerNorm.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    folded_path = args.output.with_suffix(".folding.onnx")
    options.optimized_model_filepath = str(folded_path)
    session = ort.InferenceSession(str(args.output), options, providers=["CPUExecutionProvider"])
    del session
    gc.collect()
    model = onnx.shape_inference.infer_shapes(onnx.load(folded_path))
    ranks = {v.name: len(v.type.tensor_type.shape.dim)
             for v in [*model.graph.input, *model.graph.value_info]}
    ranks.update({t.name: len(t.dims) for t in model.graph.initializer})
    einsums = 0
    for node in model.graph.node:
        if node.op_type != "Einsum":
            continue
        attr = next(a for a in node.attribute if a.name == "equation")
        equation = attr.s.decode().replace(" ", "")
        if "..." not in equation:
            continue
        inputs, output = equation.split("->")
        terms = inputs.split(",")
        lengths = [ranks[name] - len(term.replace("...", ""))
                   for name, term in zip(node.input, terms)]
        size = max(lengths)
        labels = "".join(c for c in string.ascii_letters if c not in equation)[:size]
        expanded = [term.replace("...", labels[size - n:] if n else "")
                    for term, n in zip(terms, lengths)]
        attr.s = (",".join(expanded) + "->" + output.replace("...", labels)).encode()
        einsums += 1
    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    folded_path.unlink()
    print(f"Expanded {count} LayerNormalization nodes, folded static shapes, "
          f"and expanded {einsums} Einsum equations: {args.output}")


if __name__ == "__main__":
    main()
