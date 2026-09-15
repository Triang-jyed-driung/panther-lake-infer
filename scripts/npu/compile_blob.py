"""compile_blob.py — 编译 xml 并导出 blob
用法: python compile_blob.py <xml> [device=NPU]"""
import os, shutil, sys, openvino as ov
os.environ.setdefault("ZE_INTEL_NPU_CACHE_SIZE", "0")  # 关驱动缓存：默认写 ~/.cache/ze_intel_npu_cache
xml = os.path.realpath(sys.argv[1])
cm = ov.Core().compile_model(xml, sys.argv[2] if len(sys.argv) > 2 else "NPU")
with open(xml[:-4] + ".blob", "wb") as f:
    shutil.copyfileobj(cm.export_model(), f)  # 分块搬运，避免 getvalue() 的 2×blob 峰值
print(xml[:-4] + ".blob", os.path.getsize(xml[:-4] + ".blob"))
