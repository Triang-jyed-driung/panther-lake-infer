"""Export a static RWKV IR through NPUW, then verify it in a fresh process.

Example (run export and verify as separate commands):
  python scripts/npu/npuw_weightless.py export --model model.xml --blob model.blob --emb emb.f16.bin
  python scripts/npu/npuw_weightless.py verify --model model.xml --blob model.blob --emb emb.f16.bin

WEIGHTS_PATH refers to the OpenVINO IR .bin, not the CPU engine's weight file.
The NPUW properties are internal APIs; this workflow was tested with 2026.3.1.
"""

import argparse
import gc
import io
import json
import shutil
import time
from pathlib import Path

import numpy as np
import openvino as ov


def run_windows(compiled, embedding_path, state_io):
    if len(compiled.inputs) != 1 + len(state_io):
        raise RuntimeError("Input count does not match the embedding and external-state mapping")
    request = compiled.create_infer_request()
    states = request.query_state()
    for state in states:
        state.reset()
    window, dim = map(int, compiled.input(0).shape)
    embeddings = np.memmap(embedding_path, dtype=np.float16, mode="r").reshape(-1, dim)
    arrays = {}
    timings = []
    external_states = {state["input_index"]: np.zeros(compiled.input(state["input_index"]).shape, np.float32) for state in state_io}
    for step in range(2):
        tokens = np.arange(1 + step * window, 1 + (step + 1) * window)
        inputs = np.asarray(embeddings[tokens], dtype=np.float32)
        start = time.monotonic()
        request.infer({0: inputs, **external_states})
        timings.append(time.monotonic() - start)
        for index in range(len(compiled.outputs)):
            value = np.array(request.get_output_tensor(index).data, copy=True)
            if not np.isfinite(value).all():
                raise RuntimeError(f"Non-finite output: window={step}, output={index}")
            arrays[f"window_{step}_output_{index}"] = value
        for state in states:
            value = np.array(state.state.data, copy=True)
            if not np.isfinite(value).all():
                raise RuntimeError(f"Non-finite state: {state.name}")
            arrays[f"window_{step}_state_{state.name}"] = value
        for state in state_io:
            external_states[state["input_index"]] = np.array(request.get_output_tensor(state["output_index"]).data, copy=True)
    info = {
        "state_count": len(states),
        "external_state_count": len(state_io),
        "output_count": len(compiled.outputs),
        "window": window,
        "dim": dim,
        "inference_seconds": timings,
        "top1_tokens": [np.argmax(arrays[f"window_{s}_output_0"], axis=-1).tolist() for s in range(2)],
    }
    del states, request
    return arrays, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export", "verify", "check-missing-weights", "native-reference"])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--blob", type=Path, required=True)
    parser.add_argument("--emb", type=Path, required=True)
    parser.add_argument("--pipeline", default="NONE", help="NPUW partitioning pipeline")
    args = parser.parse_args()
    model_path = args.model.resolve()
    blob_path = args.blob.resolve()
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path = blob_path.with_suffix(".reference.npz")
    report_path = blob_path.with_suffix(f".{args.action}.json")
    core = ov.Core()
    state_map_path = model_path.with_suffix(".states.json")
    state_io = json.loads(state_map_path.read_text())["states"] if state_map_path.exists() else []
    report = {"action": args.action, "openvino": ov.__version__, "model": str(model_path), "blob": str(blob_path)}
    try:
        if args.action == "check-missing-weights":
            try:
                compiled = core.import_model(io.BytesIO(blob_path.read_bytes()), "NPU")
            except Exception as exc:
                message = str(exc)
                if "WEIGHTS_PATH" not in message and "MODEL_PTR" not in message:
                    raise
                report.update(passed=True, rejected_without_weights=True, diagnostic=message)
            else:
                del compiled
                raise RuntimeError("Blob imported without external weights; weightless export was not established")
        else:
            start = time.monotonic()
            if args.action == "export":
                config = {
                    "NPU_USE_NPUW": "YES",
                    "NPUW_DEVICES": "NPU",
                    "NPUW_FALLBACK_EXEC": "NO",
                    "NPUW_ONLINE_PIPELINE": args.pipeline,
                    "NPUW_FOLD": "YES",
                    "NPUW_FUNCALL_FOR_ALL": "YES",
                    "CACHE_MODE": "OPTIMIZE_SIZE",
                }
                report["config"] = config
                print("Compiling:", json.dumps(config), flush=True)
                compiled = core.compile_model(str(model_path), "NPU", config)
                report["compile_seconds"] = time.monotonic() - start
                temporary = blob_path.with_name(blob_path.name + ".tmp")
                with temporary.open("wb") as stream:
                    shutil.copyfileobj(compiled.export_model(), stream)
                temporary.replace(blob_path)
                report["blob_bytes"] = blob_path.stat().st_size
                report["ir_weights_bytes"] = model_path.with_suffix(".bin").stat().st_size
                print("Exported", report["blob_bytes"], "bytes", flush=True)
            elif args.action == "native-reference":
                compiled = core.compile_model(str(model_path), "NPU")
                report["compile_seconds"] = time.monotonic() - start
            else:
                compiled = core.import_model(io.BytesIO(blob_path.read_bytes()), "NPU", {"WEIGHTS_PATH": str(model_path.with_suffix(".bin"))})
                report["import_seconds"] = time.monotonic() - start
            arrays, info = run_windows(compiled, args.emb.resolve(), state_io)
            report.update(info)
            del compiled
            gc.collect()
            if args.action == "export":
                np.savez(reference_path, **arrays)
            else:
                comparisons = {}
                with np.load(reference_path) as reference:
                    if set(reference.files) != set(arrays):
                        raise RuntimeError("Exported/imported output or recurrent-state names differ")
                    for name, actual in arrays.items():
                        expected = reference[name]
                        same_shape = actual.shape == expected.shape and actual.dtype == expected.dtype
                        comparisons[name] = {
                            "exact": same_shape and bool(np.array_equal(actual, expected)),
                            "max_abs_error": float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))) if same_shape else None,
                        }
                report["comparisons"] = comparisons
                report["all_exact"] = all(item["exact"] for item in comparisons.values())
                if args.action == "verify" and not report["all_exact"]:
                    raise RuntimeError("Imported results differ from the exported model reference")
            report["passed"] = True
    except Exception as exc:
        report.update(passed=False, error=str(exc))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
