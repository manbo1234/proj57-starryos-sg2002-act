from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper

root = Path("/workspace/proj57")
src = root / "proj57_relu_tpu_fold.onnx"
dst = root / "proj57_relu_tpu_fold2.onnx"
data = root / "proj57_relu_tpu_fold2.onnx.data"

model = onnx.load(str(src), load_external_data=True)

consts = {}

for init in model.graph.initializer:
    consts[init.name] = numpy_helper.to_array(init)

for node in model.graph.node:
    if node.op_type == "Constant" and len(node.output) == 1:
        for a in node.attribute:
            if a.name == "value":
                consts[node.output[0]] = numpy_helper.to_array(a.t)

def get_attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default

new_nodes = []
folded = 0

for node in model.graph.node:
    if node.op_type == "Split" and node.input and node.input[0] in consts:
        arr = consts[node.input[0]]
        axis = int(get_attr(node, "axis", 0))

        split = None
        if len(node.input) >= 2 and node.input[1] in consts:
            split = consts[node.input[1]].astype(np.int64).tolist()

        if split is None:
            split_attr = get_attr(node, "split", None)
            if split_attr is not None:
                split = list(split_attr)

        if split is None:
            if arr.shape[axis] % len(node.output) != 0:
                raise RuntimeError(f"cannot evenly split {node.name}: shape={arr.shape}, axis={axis}, outputs={len(node.output)}")
            each = arr.shape[axis] // len(node.output)
            split = [each] * len(node.output)

        cut = np.cumsum(split)[:-1]
        parts = np.split(arr, cut, axis=axis)

        for out_name, part in zip(node.output, parts):
            tensor = numpy_helper.from_array(part.astype(arr.dtype), name=out_name + "_const")
            new_nodes.append(helper.make_node(
                "Constant",
                [],
                [out_name],
                name=(node.name or "split") + "_" + out_name + "_folded",
                value=tensor,
            ))
            consts[out_name] = part

        print("folded Split", node.name, "input", node.input[0], "shape", arr.shape, "axis", axis, "split", split)
        folded += 1
    else:
        new_nodes.append(node)

del model.graph.node[:]
model.graph.node.extend(new_nodes)

onnx.save_model(
    model,
    str(dst),
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=data.name,
    size_threshold=1024,
)

print("folded const Split:", folded)
print("saved:", dst)
