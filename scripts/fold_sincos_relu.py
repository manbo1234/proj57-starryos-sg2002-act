from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

root = Path("/workspace/proj57")
src = root / "proj57_relu_tpu.onnx"
dst = root / "proj57_relu_tpu_fold.onnx"
data = root / "proj57_relu_tpu_fold.onnx.data"

model = onnx.load(str(src), load_external_data=True)

producer = {}
consts = {}

for init in model.graph.initializer:
    consts[init.name] = numpy_helper.to_array(init)

for node in model.graph.node:
    for out in node.output:
        producer[out] = node

def get_attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default

def to_dtype(onnx_type):
    mp = {
        TensorProto.FLOAT: np.float32,
        TensorProto.DOUBLE: np.float64,
        TensorProto.INT64: np.int64,
        TensorProto.INT32: np.int32,
        TensorProto.BOOL: np.bool_,
    }
    return mp.get(onnx_type, np.float32)

def eval_value(name):
    if name in consts:
        return consts[name]

    node = producer.get(name)
    if node is None:
        raise RuntimeError(f"not constant: {name}")

    vals = [eval_value(x) for x in node.input if x]

    if node.op_type == "Constant":
        v = get_attr(node, "value")
        arr = numpy_helper.to_array(v)
    elif node.op_type == "Cast":
        arr = vals[0].astype(to_dtype(get_attr(node, "to")))
    elif node.op_type == "Add":
        arr = vals[0] + vals[1]
    elif node.op_type == "Sub":
        arr = vals[0] - vals[1]
    elif node.op_type == "Mul":
        arr = vals[0] * vals[1]
    elif node.op_type == "Div":
        arr = vals[0] / vals[1]
    elif node.op_type == "Pow":
        arr = vals[0] ** vals[1]
    elif node.op_type == "Sin":
        arr = np.sin(vals[0]).astype(np.float32)
    elif node.op_type == "Cos":
        arr = np.cos(vals[0]).astype(np.float32)
    elif node.op_type == "Slice":
        data_in = vals[0]
        starts = vals[1].astype(np.int64).tolist()
        ends = vals[2].astype(np.int64).tolist()
        axes = vals[3].astype(np.int64).tolist() if len(vals) >= 4 else list(range(len(starts)))
        steps = vals[4].astype(np.int64).tolist() if len(vals) >= 5 else [1] * len(starts)
        sl = [slice(None)] * data_in.ndim
        for s, e, ax, st in zip(starts, ends, axes, steps):
            if e > 9223372036854770000:
                e = None
            sl[ax] = slice(int(s), None if e is None else int(e), int(st))
        arr = data_in[tuple(sl)]
    elif node.op_type == "Unsqueeze":
        arr = vals[0]
        axes = vals[1].astype(np.int64).tolist() if len(vals) > 1 else get_attr(node, "axes")
        for ax in sorted([int(x) for x in axes]):
            arr = np.expand_dims(arr, ax)
    elif node.op_type == "Concat":
        axis = int(get_attr(node, "axis", 0))
        arr = np.concatenate(vals, axis=axis)
    elif node.op_type == "Reshape":
        arr = np.reshape(vals[0], vals[1].astype(np.int64))
    else:
        raise RuntimeError(f"unsupported const op {node.op_type} for {name}")

    consts[name] = arr
    return arr

new_nodes = []
folded = 0

for node in model.graph.node:
    if node.op_type in ("Sin", "Cos"):
        out = node.output[0]
        arr = eval_value(out).astype(np.float32)
        tensor = numpy_helper.from_array(arr, name=out + "_const")
        new_nodes.append(helper.make_node("Constant", [], [out], name=node.name + "_folded", value=tensor))
        print("folded", node.op_type, node.name, out, arr.shape)
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

print("folded Sin/Cos:", folded)
print("saved:", dst)
