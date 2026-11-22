"""Capture real, pretokenized ShareGPT examples with their original assistant masks."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path
from threading import Lock
import argparse
import hashlib
import json
import os
import time
import uuid

import requests
import torch
from mooncake.store import MooncakeDistributedStore


SOURCE_SHA = "d88c14ff9aaf4cbfe77ea3dbba794f4f5e815a4cfea747865c7c9bc8e73df83a"
VOCAB_SHA = "bb23e71ff5bccec5b91cbf18b9c2c8a4022d606cbfd7a251be21ed54b9dd17f7"
FEATURES = {"aux": "hidden_states", "last_hidden": "target_last_hidden_states"}


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.numpy().astype("<i8", copy=False).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--heldout", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    source_manifest = json.loads((args.corpus / "manifest.json").read_text())
    assert file_hash(args.corpus / "vocab.pt") == source_manifest["mapping_sha256"] == VOCAB_SHA
    assert file_hash(args.target / "tokenizer.json") == source_manifest["tokenizer_sha256"]
    data_dir = args.corpus.parent
    if args.heldout:
        source = data_dir / "heldout-first0128.jsonl"
        details = data_dir / "heldout-first0128.audit.jsonl"
        source_digest = "ec3f1edb952eca5981fcd36e13ebe3f860174391b7f0165a68f536ef3ef85b81"
        expected_audit = "fee33ed16b87154ae511b9a8ee917e580da7efa3db89c738d218a40ed734eb25"
        stage_audit = json.loads((data_dir / "evidence/heldout-first0128.check.json").read_text())
        assert stage_audit["passed"] and stage_audit["data_sha256"] == source_digest
        formal_audit = json.loads((data_dir / "formal-split-audit.json").read_text())
        assert formal_audit["passed"]
        assert not any(formal_audit["cross_split_overlap"]["original_sharegpt__teacher_heldout"].values())
        population = "sealed independent teacher-response held-out partition"
    else:
        source = args.corpus / "train.jsonl"
        details = args.corpus / "families.jsonl"
        source_digest = source_manifest["train_sha256"]
        assert source_digest == SOURCE_SHA
        expected_audit = source_manifest["family_manifest_sha256"]
        population = "sealed original ShareGPT-answer training partition"
    assert file_hash(source) == source_digest
    assert file_hash(details) == expected_audit
    rows = []
    with source.open() as f, details.open() as audit:
        for raw, detail in islice(zip(f, audit), args.offset, args.offset + args.count):
            row, family = json.loads(raw), json.loads(detail)
            n = family["prompt_tokens"] if args.heldout else row["prompt_len"]
            length = len(row["input_ids"])
            assert family["id"] == row["id"]
            assert n == (family["prompt_tokens"] if args.heldout else family["prompt_len"])
            assert (family["answer_tokens"] if args.heldout else family["answer_len"]) == length - n >= 8
            assert length <= 1536 and row["loss_mask"] == [0] * n + [1] * (length - n)
            rows.append(row)
    assert len(rows) == len({r["id"] for r in rows}) == args.count
    assert len({tuple(r["input_ids"]) for r in rows}) == args.count

    run = "cp100-" + uuid.uuid4().hex
    stage = args.root / ".features-staging"
    stage.mkdir(exist_ok=True)
    assert not (args.root / "features").exists()
    deadline = time.monotonic() + 400
    while time.monotonic() < deadline:
        try:
            if requests.get("http://127.0.0.1:30186/health", timeout=2).ok:
                break
        except requests.RequestException:
            pass
        time.sleep(2)
    else:
        raise RuntimeError("Capture endpoint failed its readiness check")
    store = MooncakeDistributedStore()
    rc = store.setup(local_hostname="127.0.0.1", metadata_server="http://127.0.0.1:35880/metadata",
                     global_segment_size=0, local_buffer_size=160 << 20, protocol="tcp",
                     rdma_devices="", master_server_addr="127.0.0.1:35551")
    assert rc in (None, 0), rc
    lock = Lock()

    def capture(index, row):
        stem = f"{index:06d}"
        file = stage / (stem + ".ckpt")
        sidecar = stage / (stem + ".json")
        ids = torch.tensor(row["input_ids"], dtype=torch.int64)
        mask = torch.tensor(row["loss_mask"], dtype=torch.int64)
        info = {"filename": file.name, "source_index": index,
                "source_id_sha256": hashlib.sha256(row["id"].encode()).hexdigest(),
                "input_ids_sha256": tensor_hash(ids), "loss_mask_sha256": tensor_hash(mask),
                "input_tokens": ids.numel(), "supervised_tokens": int(mask.sum())}
        if file.exists() and sidecar.exists():
            prior = json.loads(sidecar.read_text())
            if all(prior[k] == v for k, v in info.items()) and file_hash(file) == prior["file_sha256"]:
                return prior
        spec = dict(store_id=run, sample_id=stem, gen=1, replace=False, features=FEATURES, passthrough=[])
        keys = [f"{run}/{stem}/g1/{name}" for name in FEATURES.values()]
        try:
            response = requests.post("http://127.0.0.1:30186/generate", json={
                "input_ids":row["input_ids"], "extra_key":run + "-" + stem,
                "sampling_params":{"temperature":0, "max_new_tokens":0,
                                   "custom_params":{"spec_capture":json.dumps(spec)}}}, timeout=180)
            response.raise_for_status()
            result, = response.json()["meta_info"]["spec_capture"]
            assert (result["sample_id"], result["store_id"], result["gen"]) == (stem, run, 1)
            assert result["aux_layer_ids"] == [1, 23, 44]
            record = {"input_ids":ids, "loss_mask":mask}
            for key, (name, width) in zip(keys, (("hidden_states", 6144), ("target_last_hidden_states", 2048))):
                meta = result["features"][name]
                assert meta["dtype"] == "bfloat16" and meta["shape"] == [1, ids.numel(), width], meta
                with lock:
                    data = bytearray(store.get(key))
                assert len(data) == 2 * ids.numel() * width
                value = torch.frombuffer(data, dtype=torch.bfloat16).reshape(ids.numel(), width)
                assert torch.isfinite(value).all(), (index, name)
                record[name] = value
            tmp = file.with_suffix(".part")
            with tmp.open("wb") as out:
                torch.save(record, out)
            info["file_sha256"] = file_hash(tmp)
            os.replace(tmp, file)
            sidecar.write_text(json.dumps(info))
            return info
        finally:
            with lock:
                for key in keys:
                    store.remove(key)

    records = []
    try:
        with ThreadPoolExecutor(args.workers) as pool:
            work = [pool.submit(capture, i, r) for i, r in enumerate(rows, start=args.offset)]
            for future in as_completed(work):
                try:
                    records.append(future.result())
                except BaseException:
                    for pending in work:
                        pending.cancel()
                    raise
                if len(records) % 32 == 0:
                    print(f"Captured {len(records)}/{args.count}", flush=True)
    finally:
        store.close()
    records.sort(key=lambda r:r["source_index"])
    manifest = {
        "source":population, "source_file":source.name, "source_sha256":source_digest,
        "source_family_audit_sha256":expected_audit,
        "pretokenized_source_sha256":source_digest, "vocab_mapping_sha256":VOCAB_SHA,
        "target":"Qwen/Qwen3-30B-A3B-Instruct-2507", "target_path":str(args.target),
        "target_revision":"0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
        "target_config_sha256":file_hash(args.target / "config.json"),
        "tokenizer_sha256":source_manifest["tokenizer_sha256"],
        "aux_layer_ids":[1,23,44], "feature_dtype":"bfloat16", "feature_widths":[6144,2048],
        "selection":f"source rows [{args.offset}, {args.offset + args.count}) of {population}",
        "mask":"source per-token assistant response loss mask copied verbatim",
        "unique_sources":len({r["source_id_sha256"] for r in records}),
        "unique_sequences":len({r["input_ids_sha256"] for r in records}),
        "input_tokens":sum(r["input_tokens"] for r in records),
        "supervised_tokens":sum(r["supervised_tokens"] for r in records),
        "max_input_tokens":max(r["input_tokens"] for r in records),
        "min_input_tokens":min(r["input_tokens"] for r in records),
        "records":records,
    }
    assert len(list(stage.glob("*.ckpt"))) == len(records) == args.count
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for r in records:
        (stage / (Path(r["filename"]).stem + ".json")).unlink()
    for path in stage.iterdir():
        path.chmod(0o444)
    stage.chmod(0o555)
    stage.rename(args.root / "features")
    print(json.dumps({k:v for k,v in manifest.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
