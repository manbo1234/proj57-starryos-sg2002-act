from pathlib import Path
import onnx
from onnx import helper

root = Path("/workspace/proj57")
src = root / "proj57_relu.onnx"
dst = root / "proj57_relu_tpu.onnx"
data = root / "proj57_relu_tpu.onnx.data"

if dst.exists():
    dst.unlink()
if data.exists():
    data.unlink()

model = onnx.load(str(src), load_external_data=True)
initializers = {x.name: x for x in model.graph.initializer}

patched = 0

for node in model.graph.node:
    if node.op_type != "Conv":
        continue

    attrs = {a.name for a in node.attribute}

    if len(node.input) < 2 or node.input[1] not in initializers:
        print("skip Conv without static weight:", node.name)
        continue

    weight = initializers[node.input[1]]
    spatial_rank = len(weight.dims) - 2

    if "kernel_shape" not in attrs:
        node.attribute.append(helper.make_attribute("kernel_shape", list(weight.dims[2:])))
        patched += 1

    if "strides" not in attrs:
        node.attribute.append(helper.make_attribute("strides", [1] * spatial_rank))

    if "dilations" not in attrs:
        node.attribute.append(helper.make_attribute("dilations", [1] * spatial_rank))

    if "pads" not in attrs:
        node.attribute.append(helper.make_attribute("pads", [0] * (2 * spatial_rank)))

    if "group" not in attrs:
        node.attribute.append(helper.make_attribute("group", 1))

onnx.save_model(
    model,
    str(dst),
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=data.name,
    size_threshold=1024,
)

print("patched Conv kernel_shape attrs:", patched)
print("saved:", dst)
print("saved external data:", data)
