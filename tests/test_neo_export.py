"""Text-only round trip: full multimodal checkpoint -> text model -> (train) ->
written back into the full layout with vision and MTP tensors untouched."""
from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _tiny_full(path: str) -> None:
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    text = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                linear_key_head_dim=16, linear_value_head_dim=16, vocab_size=248320, mtp_num_hidden_layers=0,
                layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                tie_word_embeddings=True)
    vision = dict(depth=1, hidden_size=32, num_heads=2, intermediate_size=64, out_hidden_size=64, patch_size=16,
                  spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=64)
    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=text, vision_config=vision,
                                                          tie_word_embeddings=True))
    model.save_pretrained(path)
    try:
        AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B").save_pretrained(path)
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen3.5 tokenizer not available offline")
    # real checkpoints also carry an MTP head the HF class ignores
    shard = [n for n in os.listdir(path) if n.endswith(".safetensors")][0]
    tensors = load_file(os.path.join(path, shard))
    tensors["mtp.fc.weight"] = torch.randn(8, 8)
    save_file(tensors, os.path.join(path, shard), metadata={"format": "pt"})


def test_round_trip(tmp_path):
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    from neo.export import export_full, make_text_checkpoint

    full, text, trained, out = (str(tmp_path / d) for d in ("full", "text", "trained", "out"))
    _tiny_full(full)
    make_text_checkpoint(full, text, dtype="float32")

    model = AutoModelForCausalLM.from_pretrained(text, dtype=torch.float32)
    with torch.no_grad():
        model.model.layers[0].linear_attn.in_proj_qkv.weight.add_(0.5)   # "training"
        model.model.layers[3].self_attn.q_proj.weight.mul_(1.1)
    model.save_pretrained(trained)

    report = export_full(trained, full, out)
    assert report["replaced"] > 0

    orig = {k: v for n in os.listdir(full) if n.endswith(".safetensors") for k, v in load_file(os.path.join(full, n)).items()}
    new = {k: v for n in os.listdir(out) if n.endswith(".safetensors") for k, v in load_file(os.path.join(out, n)).items()}
    assert set(orig) == set(new)
    for k in orig:
        if k.startswith("model.language_model."):
            continue
        assert torch.equal(orig[k], new[k]), k                      # vision + MTP untouched
    changed = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    assert not torch.equal(orig[changed], new[changed])

    reloaded = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
    x = torch.randint(0, 1000, (1, 12))
    with torch.no_grad():
        assert torch.allclose(reloaded(x).logits, model(x).logits, atol=1e-5)
