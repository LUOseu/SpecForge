"""Scored offline VLM feature transport and a CPU draft optimizer step."""

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.common.dflash_family_model import ScoredVlmDFlashModel
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.training.strategies.base import DFlashTrainStrategy


def test_scored_vlm_offline_train_step(tmp_path):
    torch.manual_seed(42)
    for i, length in enumerate((12, 9)):
        raw = {
            "input_ids": torch.arange(length),
            "hidden_states": torch.randn(length, 8),
            "loss_mask": torch.ones(length, dtype=torch.long),
            "visual_score": torch.full((length,), 2500 if i == 0 else -1),
        }
        torch.save(raw, tmp_path / f"{i}.ckpt")

    provider = (
        builtin_algorithm_registry()
        .resolve("dflash")
        .providers.offline_for("multimodal")
    )
    refs = provider.build_reader(
        str(tmp_path), run_id="smoke", ttt_length=1, max_len=11
    ).read()
    loader = FeatureDataLoader(
        LocalFeatureStore("smoke"),
        refs=refs,
        strategy="dflash",
        batch_size=2,
        collate_fn=provider.build_collator(),
        per_sample_transform=provider.build_normalizer(11),
    )
    batch = next(iter(loader))
    scores = batch.tensors["visual_score"]
    torch.testing.assert_close(scores[0], torch.full((11,), 0.25))
    torch.testing.assert_close(scores[1], torch.full((11,), -1.0))

    cfg = Qwen3Config(
        architectures=["DFlashDraftModel"],
        block_size=4,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        num_target_layers=6,
        head_dim=4,
        max_position_embeddings=64,
        vocab_size=32,
        layer_types=["full_attention"],
    )
    cfg._attn_implementation = "eager"
    draft = DFlashDraftModel(cfg)
    wrapper = ScoredVlmDFlashModel(
        draft,
        nn.Linear(8, 32, bias=False).requires_grad_(False),
        nn.Embedding(32, 8).requires_grad_(False),
        mask_token_id=31,
        block_size=4,
        attention_backend="eager",
        num_anchors=3,
        objective_chunk_blocks=1,
        loss_type="mmflash",
    )
    anchors, keep = wrapper._sample_anchor_positions(
        11, batch.tensors["loss_mask"], "cpu"
    )
    assert keep.all() and (anchors <= 11 - wrapper.block_size).all()
    strategy = DFlashTrainStrategy(wrapper)
    before = draft.fc.weight.detach().clone()
    optimizer = torch.optim.AdamW(draft.parameters(), lr=1e-3)
    out = strategy.forward_loss(batch)
    assert torch.isfinite(out.loss) and out.loss_terms[1].item() == 2
    out.loss.backward()
    assert (
        torch.isfinite(draft.fc.weight.grad).all()
        and draft.fc.weight.grad.abs().sum() > 0
    )
    optimizer.step()
    assert not torch.equal(before, draft.fc.weight)
