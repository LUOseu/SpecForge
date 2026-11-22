#!/usr/bin/env python3
"""Create a tiny Qwen3.5 MMFlash CPU fixture from a pinned public processor.

Usage from the repository root: python scripts/gates/make_mmflash_cpu_fixture.py
--output-dir /tmp/mmflash-test. Only tokenizer and processor files are downloaded.
The local random model is intentionally sharpened to exercise nonzero gated
weights with Qwen's 248k vocabulary. The generated data tests software behavior.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PROCESSOR_FILES = [
    "chat_template.jinja",
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
]


def create(root: Path):
    if root.exists():
        raise FileExistsError(f"fixture destination already exists: {root}")
    snapshot = Path(
        snapshot_download(
            "Qwen/Qwen3.5-4B", revision=REVISION, allow_patterns=PROCESSOR_FILES
        )
    )
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".mmflash-fixture-", dir=root.parent
    ) as stage_dir:
        stage = Path(stage_dir)
        target = stage / "target"
        target.mkdir()
        for name in PROCESSOR_FILES:
            shutil.copyfile(snapshot / name, target / name)
        config = Qwen3_5Config.from_pretrained(target)
        config.text_config.update(
            dict(
                dtype=torch.float32,
                num_hidden_layers=4,
                hidden_size=32,
                intermediate_size=64,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
                linear_key_head_dim=8,
                linear_value_head_dim=8,
                linear_num_key_heads=2,
                linear_num_value_heads=2,
                layer_types=["linear_attention", "full_attention"] * 2,
                rope_parameters=dict(
                    rope_type="default",
                    rope_theta=10000000,
                    partial_rotary_factor=1.0,
                    mrope_section=[2, 3, 3],
                    mrope_interleaved=True,
                ),
            )
        )
        config.vision_config.update(
            dict(
                depth=1,
                hidden_size=32,
                intermediate_size=64,
                num_heads=2,
                out_hidden_size=32,
            )
        )
        torch.manual_seed(41)
        target_model = Qwen3_5ForConditionalGeneration(config).eval()
        target_model.config.tie_word_embeddings = False
        target_model.config.text_config.tie_word_embeddings = False
        # A uniform random teacher gives exp(-H) ~ 1/vocab, zero after int64
        # wire quantization. Sharpen logits so the visual branch is exercised.
        target_model.lm_head.weight = torch.nn.Parameter(
            target_model.lm_head.weight.detach().clone() * 40
        )
        target_model.save_pretrained(target)

        draft = json.loads((ROOT / "configs/qwen3.5-4b-dflash.json").read_text())
        draft.update(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_target_layers=4,
            head_dim=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            block_size=4,
            layer_types=["full_attention"] * 2,
            dtype="float32",
        )
        draft["dflash_config"]["target_layer_ids"] = [0, 2]
        (stage / "draft_config.json").write_text(json.dumps(draft, indent=2) + "\n")
        with (stage / "data.jsonl").open("w") as sink:
            for color in ("red", "blue"):
                name = color + ".png"
                Image.new("RGB", (64, 64), color).save(stage / name)
                turns = [
                    {
                        "role": "user",
                        "content": "<image>\nWhat is the color of the square?",
                    },
                    {
                        "role": "assistant",
                        "content": f"The square looks {color}. Its color is uniform across the image from top to bottom.",
                    },
                ]
                sink.write(
                    json.dumps(
                        dict(id=color, image=str(root / name), conversations=turns)
                    )
                    + "\n"
                )
        cfg = yaml.safe_load(
            (
                ROOT / "examples/configs/offline/colocated/qwen3.5-4b-mmflash-a100.yaml"
            ).read_text()
        )
        cfg["model"].update(
            target_model_path=str(root / "target"),
            draft_model_config=str(root / "draft_config.json"),
            torch_dtype="float32",
        )
        cfg["data"].update(
            hidden_states_path=str(root / "features"),
            max_length=512,
            dataloader_num_workers=0,
        )
        cfg["training"].update(
            max_steps=2,
            save_interval=2,
            num_anchors=4,
            attention_backend="sdpa",
            fsdp_sharding="NO_SHARD",
        )
        cfg.update(run_id="tiny-mmflash", output_dir=str(root / "training"))
        (stage / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        stage.replace(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    create(destination)
    print(f"Created the CPU fixture in {destination}")
