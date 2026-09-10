"""Export the project's fixed-batch ONNX model without replacing existing files."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from scripts.provenance import environment, sha256


def sort_nodes(model):
    """Order conversion-inserted casts before their consumers; preserve all ops."""
    available = {v.name for v in model.graph.input} | {v.name for v in model.graph.initializer} | {''}
    pending, ordered = list(model.graph.node), []
    while pending:
        ready = [node for node in pending if all(name in available for name in node.input)]
        if not ready:
            raise ValueError('Graph contains unresolved dependencies or a cycle')
        remaining = []
        for node in pending:
            if all(name in available for name in node.input):
                ordered.append(node)
                available.update(node.output)
            else:
                remaining.append(node)
        pending = remaining
    del model.graph.node[:]
    model.graph.node.extend(ordered)


def inspect_model(path):
    import onnx
    model = onnx.load(str(path))
    # Old deployed exports are usable in ORT despite unsorted conversion casts.
    # Validate a sorted in-memory copy without touching the deployed file.
    original_order = [node.name for node in model.graph.node]
    sort_nodes(model)
    onnx.checker.check_model(model)
    def interface(value):
        tensor = value.type.tensor_type
        return {'name': value.name, 'dtype': tensor.elem_type,
                'shape': [d.dim_value if d.HasField('dim_value') else d.dim_param for d in tensor.shape.dim]}
    inputs, outputs = list(map(interface, model.graph.input)), list(map(interface, model.graph.output))
    if len(inputs) != 1 or inputs[0]['shape'] != [4, 3, 640, 640] or inputs[0]['dtype'] != 1:
        raise ValueError(f'Unexpected model input: {inputs}')
    if len(outputs) != 1 or outputs[0]['shape'] != [4, 300, 6] or outputs[0]['dtype'] != 1:
        raise ValueError(f'Unexpected model output: {outputs}')
    if not any(t.data_type == onnx.TensorProto.FLOAT16 for t in model.graph.initializer):
        raise ValueError('Model lacks FP16 weights')
    metadata = {p.key: p.value for p in model.metadata_props}
    if metadata.get('end2end') != 'True':
        raise ValueError('Expected native end-to-end detector output')
    return {'inputs': inputs, 'outputs': outputs, 'opset': [v.version for v in model.opset_import],
            'checker_required_sort': original_order != [node.name for node in model.graph.node],
            'sha256': sha256(path), 'metadata': metadata}


def export(weights, output):
    if not weights.is_file():
        raise FileNotFoundError('Supply local YOLO26m weights; automatic downloads are disabled')
    if output.exists():
        raise FileExistsError(f'Refusing to replace {output}; choose a new output path')
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    from ultralytics import YOLO
    output.parent.mkdir(parents=True, exist_ok=True)
    args = dict(format='onnx', imgsz=640, batch=4, dynamic=False, quantize=16,
                simplify=True, opset=20, nms=False, device='cpu')
    with tempfile.TemporaryDirectory(prefix='traffic-export-') as temp:
        copied = Path(temp) / weights.name
        shutil.copy2(weights, copied)
        result = Path(YOLO(str(copied)).export(**args))
        import onnx
        graph = onnx.load(result)
        sort_nodes(graph)
        onnx.checker.check_model(graph)
        onnx.save(graph, result)
        contract = inspect_model(result)
        shutil.copy2(result, output)
    report = {'source_weights': weights.name, 'weights_sha256': sha256(weights),
              'export_args': args, 'model': contract, 'environment': environment()}
    output.with_suffix('.export.json').write_text(json.dumps(report, indent=2))
    print(f'Exported and checked {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.weights, args.output)


if __name__ == '__main__':
    main()
