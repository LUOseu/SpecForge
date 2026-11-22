import concurrent.futures, hashlib, json, pathlib, subprocess, sys

REV = "8049631c405ae6576f93f445c6b8166f76f5505a"
REPO = "HuggingFaceH4/ultrachat_200k"

def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(exist_ok=True, parents=True)
    tree = json.loads(subprocess.check_output(["curl", "-fsSL", "--max-time", "60", f"https://huggingface.co/api/datasets/{REPO}/tree/{REV}/data"]))
    files = [x for x in tree if x["path"].split("/")[-1].startswith("train_sft-")]
    assert len(files) == 3
    def fetch(row):
        name = pathlib.Path(row["path"]).name
        target = out/name
        if not target.exists():
            partial = target.with_suffix(".part")
            subprocess.run(["curl", "-fLsS", "--retry", "5", "--retry-delay", "1", "--connect-timeout", "12", "--speed-time", "45", "--speed-limit", "4096", "-C", "-", "-o", str(partial), f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{row['path']}?download=true"], check=True)
        else:
            partial = target
        with partial.open("rb") as f: digest = hashlib.file_digest(f, "sha256").hexdigest()
        assert digest == row["lfs"]["oid"] and partial.stat().st_size == row["size"], name
        if partial != target: partial.replace(target)
        print(name, row["size"], digest, flush=True)
        return {"path": name, "sha256": digest, "bytes": row["size"]}
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        verified = list(pool.map(fetch, files))
    (out/"manifest.json").write_text(json.dumps({"repo": REPO, "revision": REV, "split": "train_sft", "files": verified}, indent=2)+"\n")

if __name__ == "__main__":
    main()
