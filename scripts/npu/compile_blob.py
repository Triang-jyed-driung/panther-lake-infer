"""Compile OpenVINO IR: compile_blob.py model.xml [NPU|GPU] [output.blob]."""
import argparse
import os
from pathlib import Path
import shutil
import time
import openvino as ov

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('xml', type=Path)
    parser.add_argument('device', nargs='?', default='NPU', choices=['NPU', 'GPU'])
    parser.add_argument('output', nargs='?', type=Path)
    parser.add_argument('--precision', choices=['f16', 'f32'], default='f16',
                        help='GPU inference precision (default: f16)')
    args = parser.parse_args()
    xml = args.xml.resolve()
    output = args.output or xml.with_suffix('.gpu.blob' if args.device == 'GPU' else '.blob')
    os.environ.setdefault('ZE_INTEL_NPU_CACHE_SIZE', '0')
    properties = {}
    if args.device == 'GPU':
        # Match the fp16 weights; f32 can double GPU weight memory.
        properties = {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': args.precision}
    start = time.monotonic()
    print(f'Compiling {xml} for {args.device}...', flush=True)
    cm = ov.Core().compile_model(str(xml), args.device, properties)
    print(f'Compiled in {time.monotonic() - start:.2f}s; exporting {output}', flush=True)
    temporary = output.with_name(output.name + '.tmp')
    with temporary.open('wb') as f:
        shutil.copyfileobj(cm.export_model(), f)
    temporary.replace(output)
    print(f'{output}: {output.stat().st_size} bytes', flush=True)


if __name__ == "__main__":
    main()
