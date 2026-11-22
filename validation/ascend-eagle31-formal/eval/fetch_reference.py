"""Fetch the published matched-target EAGLE3 serving reference with immutable content checks."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
REPO="lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex"
REV="d1ac703a537d2b8a5b748d4f5f8ca7e97efe9214"
FILES={"config.json":"ea074ee6751dbf97acebcc8da5240c3b3f44c81371f06463b09a6934f52055e0",
"model.safetensors":"2615281348ed0b331ca14406c48cb4a05f372052d9ede286a86906543cd13972",
"README.md":"b029a80a77b920e0256211c43671d4271fd8fa31f9a60cec786cc4aeb1e26a00"}
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda:f.read(8<<20),b""):h.update(buf)
    return h.hexdigest()
def fetch(out):
    out.mkdir(parents=True,exist_ok=True)
    for name,expected in FILES.items():
        path=out/name
        if path.exists() and digest(path)==expected:continue
        partial=out/(name+".partial")
        urls=[f"https://www.modelscope.cn/models/{REPO}/resolve/master/{name}",
              f"https://huggingface.co/{REPO}/resolve/{REV}/{name}", f"https://hf-mirror.com/{REPO}/resolve/{REV}/{name}"]
        for url in urls:
            completed=subprocess.run(["curl","--fail","--silent","--show-error","--location","--connect-timeout","8","--max-time","170","--output",str(partial),url])
            if completed.returncode==0 and digest(partial)==expected:
                partial.replace(path);break
        else: raise RuntimeError(f"missing checksum-verified {name}")
    (out/"reference-provenance.json").write_text(json.dumps(dict(repo=REPO,revision=REV,sha256=FILES),indent=2)+"\n")
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("output",type=Path);fetch(p.parse_args().output)
