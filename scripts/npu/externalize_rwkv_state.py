"""Replace RWKV ReadValue/Assign state with explicit host-managed I/O.

Usage: python scripts/npu/externalize_rwkv_state.py stateful.xml stateless.xml
The resulting .states.json describes how to feed outputs into the next call.
"""

import argparse
import json
from pathlib import Path

import openvino as ov


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model = ov.Core().read_model(str(args.model.resolve()))
    reads = {}
    writes = {}
    for node in model.get_ordered_ops():
        kind = node.get_type_name()
        if kind in ("ReadValue", "Assign"):
            variable_id = node.get_attributes()["variable_id"]
            target = reads if kind == "ReadValue" else writes
            if variable_id in target:
                raise ValueError(f"Duplicate {kind} for {variable_id}")
            target[variable_id] = node
    if not reads or set(reads) != set(writes):
        raise ValueError("Expected matching ReadValue/Assign pairs")
    params = list(model.get_parameters())
    results = list(model.get_results())
    state_io = []
    for name in sorted(reads):
        read = reads[name]
        if read.get_input_size():
            raise ValueError("Only RWKV states with implicit zero initialization are supported")
        parameter = ov.opset13.parameter(read.get_output_shape(0), read.get_output_element_type(0), f"state_in.{name}")
        read.output(0).replace(parameter.output(0))
        result = ov.opset13.result(writes[name].input_value(0), name=f"state_out.{name}")
        result.output(0).get_tensor().set_names({f"state_out.{name}"})
        state_io.append({"name": name, "input_index": len(params), "output_index": len(results)})
        params.append(parameter)
        results.append(result)
    external = ov.Model(results, params, model.get_friendly_name() + "_external_state")
    if any(node.get_type_name() in ("ReadValue", "Assign") for node in external.get_ordered_ops()):
        raise RuntimeError("State operations survived externalization")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ov.serialize(external, str(args.output))
    metadata = {"source": str(args.model.resolve()), "original_output_count": len(model.outputs), "states": state_io}
    args.output.with_suffix(".states.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Exported {args.output}: {len(params)} inputs, {len(results)} outputs, {len(state_io)} external states")


if __name__ == "__main__":
    main()
