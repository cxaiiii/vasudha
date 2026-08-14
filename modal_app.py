"""
Modal entrypoint for Vasudha training.

Modal containers are ephemeral: anything outside a Volume vanishes when the
function returns. So the HF cache, the converted checkpoints, the prepared
dataset and the run outputs all live on one mounted Volume — otherwise every
invocation re-downloads 8GB of Qwen3 and loses its own results.

    modal run modal_app.py::convert          # one-time, ~10 min
    modal run modal_app.py::prepare_data     # one-time, ~10 min
    modal run modal_app.py::train --attn sdpa --steps 400
    modal shell modal_app.py::train          # interactive debugging

Costs are per-second, so the shell is for poking at things, not for thinking.
"""

import modal

app = modal.App("vasudha-sft")

# One Volume for everything durable. /vol/hf holds the HF cache so Qwen3 is
# downloaded once; /vol/ckpt holds converted models, data and run outputs.
vol = modal.Volume.from_name("vasudha-vol", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers>=4.51",
        "accelerate",
        "peft",
        "trl",
        "bitsandbytes",
        "datasets",
        "safetensors",
        "hydra-core",
        "omegaconf",
        "rich",
        # SFTConfig sets report_to="tensorboard"; the Trainer's callback
        # constructor raises without it, after tokenization but before step 1.
        "tensorboard",
        "sentencepiece",
        "protobuf",
        "fastapi[standard]",
    )
)

# Modal requires all package/build steps before add_local_dir.  Keep local
# source injection last in both images so source edits do not invalidate the
# dependency layer and Unsloth can build cleanly.
#
# Every large local artifact must be listed here, not just vasudha-model —
# add_local_dir uploads the whole tree on every single `modal run`, and a
# missed multi-GB directory (ollama/'s downloaded GGUF, or the stray
# checkpoint-250/checkpoint-50 dirs that ended up nested inside the vasudha/
# package folder) turns every invocation into a multi-GB upload over
# whatever this machine's connection is doing that day — almost certainly
# what was behind the repeated "connection lost" failures tonight, not a
# problem with the conversion script itself.
_local_files = {
    "remote_path": "/root/vasudha",
    "ignore": [
        "*.pyc", "__pycache__", ".git",
        "vasudha-model",       # 9.7GB merged model download
        "ollama",              # 2.6GB downloaded GGUF + Modelfile
        "checkpoint-250",      # 502MB stray LoRA checkpoint under vasudha/
        "checkpoint-50",       # 489MB stray LoRA checkpoint under vasudha/
        "*.gguf", "*.safetensors", "*.bin",  # belt-and-suspenders: no
        # model weight files belong in the uploaded source tree at all
    ],
}
image = base_image.add_local_dir(".", **_local_files)
# flash-attn/causal-conv1d are installed from their prebuilt GitHub-release
# wheels below, not compiled from source — a source build (even with a fixed
# compiler toolchain) burns 30-45 min of billed GPU-tier build time per image
# edit, which is real money on an H100 builder.  Their published wheel matrix
# tops out at CUDA 12 (no cu13 wheel exists for flash-attn 2.8.3.post1 at
# cp311 — verified against the actual GitHub release assets, not assumed),
# so the base image is CUDA 12.4, not 13.0.
unsloth_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("build-essential")
    .env({"CC": "gcc", "CXX": "g++"})
    .pip_install("packaging", "ninja")
    # torch==2.8.0 is pinned to exactly the version both prebuilt wheels below
    # were built against (cu12torch2.8). Re-asserting the same pin in the
    # unsloth/trl install below is required, not decorative: unsloth pulls in
    # xformers/torchvision as transitive deps, and an unpinned resolution of
    # those against "whatever torch is newest" silently uninstalled 2.8.0 and
    # reinstalled 2.12.1+cu13 here once already — pip reports that as a normal
    # upgrade, not an error, so it only surfaces later as an ABI mismatch when
    # flash-attn tries to import against a torch it wasn't built for. Listing
    # the pin in the SAME pip_install call as unsloth forces the resolver to
    # pick xformers/torchvision versions compatible with 2.8.0 from the start,
    # instead of resolving against latest-torch and only fixing the version
    # afterward (which would leave those two mismatched against the rollback).
    .pip_install("torch==2.8.0")
    .pip_install(
        "torch==2.8.0",
        # Pinned to the exact release recorded in vasudha-model/vasudha's
        # config.json ("unsloth_version": "2026.7.6") — that's the version
        # that actually produced the merged Qwen3.5-4B checkpoint, so it's
        # known to handle the qwen3_5 architecture. An unpinned "unsloth"
        # resolved to 2025.11.1 here once already (older, despite being
        # "latest" through Modal's pip mirror) and failed with
        # `KeyError: 'qwen3_5'` inside transformers' CONFIG_MAPPING —
        # unsloth's own patch for that gap isn't in the older release.
        "unsloth==2026.7.6",
        "transformers>=4.51",
        "trl",
        "bitsandbytes",
        "unsloth_zoo",
        "fastapi[standard]",
        # SFTConfig sets report_to="tensorboard"; the Trainer's callback
        # constructor raises without it, after tokenization but before step 1
        # — base_image already carries this for the same reason, unsloth_image
        # never did.
        "tensorboard",
    )
    # cxx11abiTRUE matches PyPI torch>=2.7's default build ABI. Verify rather
    # than trust that going forward — a silent ABI mismatch fails at import
    # time inside the training container, not at build time, and wastes a
    # full GPU allocation before anyone notices.
    .run_commands(
        "python -c \"import torch, sys; "
        "sys.exit(0 if torch.compiled_with_cxx11_abi() else 1)\" "
        "|| (echo 'torch is NOT cxx11abiTRUE — the pinned wheel URLs below "
        "no longer match; re-check the release assets.' && exit 1)"
    )
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/"
        "v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-"
        "cp311-cp311-linux_x86_64.whl"
    )
    .pip_install(
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/"
        "v1.6.2.post1/causal_conv1d-1.6.2.post1+cu12torch2.8cxx11abiTRUE-"
        "cp311-cp311-linux_x86_64.whl"
    )
    # This checkpoint's gated-deltanet linear-attention layers (24 of 32 —
    # see layer_types in config.json) need this specifically, separate from
    # flash-attn/causal-conv1d above which only cover the 8 full-attention
    # layers. Without it Unsloth falls back to a slow pure-PyTorch path for
    # the majority of the model's layers, understating any throughput number
    # measured off this image.
    .pip_install("flash-linear-attention")
    # fla's own backward kernel for gated-deltanet refuses to run on Hopper
    # (H100) with Triton in [3.4.0, 3.7.1) — it produces silently-wrong
    # gradients there, so fla raises instead of training on bad data. torch
    # 2.8.0 pins triton==3.4.0, squarely in that broken range, and bumping
    # triton independently risks breaking the torch/triton pairing that's
    # otherwise finally working. tilelang is the alternate kernel backend
    # fla's own error message names as the fix that doesn't require that.
    .pip_install("tilelang")
    .add_local_dir(".", **_local_files)
)
# HF_HOME on the Volume is what makes the 8GB Qwen3 download a one-time cost.
ENV = {"HF_HOME": "/vol/hf", "PYTHONPATH": "/root/vasudha", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

# L40S is frequently queued on smaller plans. A10G and L4 are 24GB and usually
# schedule immediately; the training function drops batch size and turns
# checkpointing back on for those so the smaller card still fits.
import os

GPU = os.environ.get("VASUDHA_GPU", "L40S")
BIG_GPU = GPU.startswith(("L40S", "A100", "H100"))

GPU_CONFIG = dict(
    gpu=GPU,
    cpu=8.0,
    memory=32768,          # 32GB — the streaming shuffle buffer never gets near this
    timeout=86400,         # 24h; the default 300s would kill training mid-run
    volumes={"/vol": vol},
    image=image,
    env=ENV,
    secrets=[hf_secret],   # HF_TOKEN injected as env var — bypasses anonymous rate limits
)

# prepare_data does nothing but stream HF datasets and shuffle text on CPU —
# no tensor ever touches a GPU. It ran under GPU_CONFIG for a while and burned
# real H100-per-hour billing on a step that's pure network/CPU wait, the
# opposite of what a GPU rents for. No `gpu` key at all, not gpu=None —
# Modal bills by allocated resources, and an idle GPU still costs.
CPU_CONFIG = dict(
    cpu=8.0,
    memory=32768,
    timeout=86400,
    volumes={"/vol": vol},
    image=image,
    env=ENV,
    secrets=[hf_secret],   # HF_TOKEN injected as env var — bypasses anonymous rate limits
)

UNSLOTH_CONFIG = {**GPU_CONFIG, "image": unsloth_image}

# GGUF conversion/quantization is CPU-only work with its own toolchain
# (cmake-built llama.cpp) — kept as a separate image so training runs never
# rebuild this and vice versa.
gguf_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake")
    .pip_install("numpy", "sentencepiece", "protobuf", "gguf", "torch", "transformers", "safetensors")
    .run_commands(
        # master, not a pinned release tag: Gated DeltaNet support (this
        # model's linear-attention layers) landed recently, and an older
        # pinned release would silently lack it and produce a broken/
        # incomplete conversion rather than a clear error.
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /root/llama.cpp",
        "cmake -B /root/llama.cpp/build -S /root/llama.cpp -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF",
        "cmake --build /root/llama.cpp/build --target llama-quantize --parallel 8",
    )
    .add_local_dir(".", **_local_files)
)

GGUF_CONFIG = dict(
    cpu=8.0,
    memory=32768,
    timeout=86400,
    volumes={"/vol": vol},
    image=gguf_image,
)


def _run(cmd: str) -> None:
    """Run a shell command in the repo, streaming output, failing loudly."""
    import os
    import subprocess

    # The child's stdout isn't a tty inside the container, so plain print()
    # in the script fully-buffers instead of line-buffering — output (e.g.
    # per-sample eval progress) sits invisible until the buffer fills or the
    # process exits, which reads as a hang even when the job is running fine.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    print(f"\n$ {cmd}\n", flush=True)
    result = subprocess.run(cmd, shell=True, cwd="/root/vasudha", env=env)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit {result.returncode}: {cmd}")


# Thinking-2507 is Qwen's dedicated reasoning-tuned 4B release — already
# fluent in the <think>...</think> format before our SFT touches it, unlike
# plain Qwen3-4B where that format has to be learned from our data alone.
# Kept as its own checkpoint name (vasudha-4b-thinking-*) so it never collides
# with the earlier plain-Qwen3-4B runs on the same Volume.
BASE_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
BASE_TAG = "thinking"

# "" as a CLI default reads fine in Python but is exactly what broke twice in
# a row: PowerShell drops an empty-string arg entirely (`--base-tag ""`) or
# the harness's own equals-form (`--base-tag=`) gets silently ignored, and in
# both cases the function falls back to BASE_TAG ("thinking") instead of no
# tag — wrong checkpoint, no error, no warning. "none" is an ordinary token
# no shell can eat, so this class of bug can't recur.
NO_TAG = "none"


def _tag(base_tag: str) -> str:
    """'-thinking' style segment, or '' when base_tag is NO_TAG (the original
    untagged plain-Qwen3-4B runs, kept working for the dense/hybrid harness)."""
    return "" if base_tag == NO_TAG else f"-{base_tag}"


def _run_dir(size_tag: str, base_tag: str, attn: str) -> str:
    """'sdpa-bf16' for the original size_tag='4b'/base_tag=NO_TAG runs — that
    exact string is already on the Volume from the completed dense model, so
    it can't change. Any other size_tag or base_tag gets prefixed so it can't
    collide with — or be silently mistaken for — that existing checkpoint."""
    prefix = "" if size_tag == "4b" else f"{size_tag}-"
    prefix += "" if base_tag == NO_TAG else f"{base_tag}-"
    return f"{prefix}{attn}-bf16"


@app.function(**GPU_CONFIG)
def convert(
    attn: str = "sdpa", moe: bool = False, model: str = BASE_MODEL, base_tag: str = BASE_TAG, size_tag: str = "4b"
) -> None:
    """Convert a Qwen3 checkpoint into a Vasudha checkpoint on the Volume."""
    out = f"/vol/ckpt/vasudha-{size_tag}{_tag(base_tag)}-{attn}{'-moe' if moe else ''}"
    flags = "" if moe else "--no-moe"
    _run(
        f"python scripts/convert_qwen3_to_vasudha.py --model {model} "
        f"--out {out} --attention-type {attn} {flags} --dtype bfloat16"
    )
    vol.commit()


@app.function(**GPU_CONFIG)
def diagnose(attn: str = "sdpa", base_tag: str = BASE_TAG, model: str = None, size_tag: str = "4b") -> None:
    """Verify a converted checkpoint matches its source Qwen3 layer by layer.

    model defaults to BASE_MODEL when base_tag is the current tag, and to
    plain Qwen3-4B for the untagged legacy conversions — get this wrong and
    the diff is meaningless (comparing against the wrong reference weights),
    not merely imprecise. Any other base_tag needs --model spelled out
    explicitly; guessing would risk the same silent-wrong-reference failure.
    """
    if model:
        ref = model
    elif base_tag == BASE_TAG:
        ref = BASE_MODEL
    elif base_tag == NO_TAG:
        ref = "Qwen/Qwen3-4B"
    else:
        raise SystemExit(f"--model is required to diagnose base_tag={base_tag!r} — no default reference known.")
    _run(f"python scripts/diagnose_conversion.py /vol/ckpt/vasudha-{size_tag}{_tag(base_tag)}-{attn} --qwen {ref}")


@app.function(**CPU_CONFIG)
def prepare_data(samples: int = 20000) -> None:
    """Materialize a resumable mixture on CPU; retries never rent a GPU."""
    _run(
        f"python scripts/prepare_dataset.py --out /vol/ckpt/data/sft-{samples} "
        f"--max-samples {samples}"
    )
    vol.commit()


@app.function(**CPU_CONFIG)
def prepare_engineering_data(samples: int = 20000) -> None:
    """Build a CAD/EDA/code/simulation add-on without repeating math data."""
    _run(
        f"python scripts/prepare_dataset.py --profile engineering "
        f"--out /vol/ckpt/data/sft-engineering-{samples} --max-samples {samples}"
    )
    vol.commit()


@app.function(**CPU_CONFIG)
def prepare_bio_sim_data(samples: int = 20000) -> None:
    """Build exactly half biology and half diverse simulation/tool-use rows."""
    _run(
        f"python scripts/prepare_dataset.py --profile bio_sim "
        f"--out /vol/ckpt/data/sft-biosim-{samples} --max-samples {samples}"
    )
    vol.commit()


@app.function(**CPU_CONFIG)
def prepare_instruction_following_data(samples: int = 25000) -> None:
    """Build a direct-answer, no-CoT add-on (OpenHermes-2.5) for stage-2
    behavioral correction. Combine with the reasoning mix via combine_data —
    training on this alone would overcorrect into never reasoning."""
    _run(
        f"python scripts/prepare_dataset.py --profile instruction_following "
        f"--out /vol/ckpt/data/sft-instruct-{samples} --max-samples {samples}"
    )
    vol.commit()


@app.function(**CPU_CONFIG)
def prepare_tool_use_data(samples: int = 20600) -> None:
    """Build a real-execution-grounded tool-use add-on (code_act +
    hermes-function-calling + execution-filtered code) for stage-2. Combine
    with the reasoning + instruction-following mixes via combine_data."""
    _run(
        f"python scripts/prepare_dataset.py --profile tool_use "
        f"--out /vol/ckpt/data/sft-tooluse-{samples} --max-samples {samples}"
    )
    vol.commit()


@app.function(**CPU_CONFIG)
def combine_data(
    base_dataset: str = "sft-20000",
    addon_dataset: str = "sft-engineering-20000",
    out_dataset: str = "sft-combined-40000",
    seed: int = 42,
) -> None:
    """Shuffle-combine two prepared Arrow datasets on CPU, with no Hub access."""
    from datasets import concatenate_datasets, load_from_disk

    def arrow_path(name: str) -> str:
        root = f"/vol/ckpt/data/{name}"
        return f"{root}/arrow" if os.path.exists(f"{root}/arrow/state.json") else root

    base = load_from_disk(arrow_path(base_dataset))
    addon = load_from_disk(arrow_path(addon_dataset))
    combined = concatenate_datasets([base, addon]).shuffle(seed=seed)
    destination = f"/vol/ckpt/data/{out_dataset}"
    combined.save_to_disk(destination)
    print(f"Combined {len(base):,} + {len(addon):,} = {len(combined):,} rows -> {destination}")
    vol.commit()


@app.function(**GPU_CONFIG)
def train(
    attn: str = "sdpa",
    steps: int = 3750,
    samples: int = 60000,
    max_seq_length: int = 4096,
    lora_r: int = 64,
    lora_alpha: int = 128,
    base_tag: str = BASE_TAG,
    size_tag: str = "4b",
    dataset_name: str = "",
    attention_override: str = "sdpa",
    packing: bool = True,
) -> None:
    """
    LoRA SFT in bf16, sized for whichever card VASUDHA_GPU points at.

    Defaults target ~2 epochs over 60k reasoning examples:
      - max_seq_length 4096 (up from 2048) so long OpenThoughts3/OpenR1
        chain-of-thought answers aren't truncated before reaching a final
        answer — training on a cut-off chain teaches the model to stop
        mid-reasoning, the same failure we saw truncation cause at eval time.
      - lora_r 64 / alpha 128 (up from 16/32, same alpha:r=2 ratio) for more
        capacity to actually shift a 4B model's behavior in ~2 epochs rather
        than just nudge it.
    Effective batch is held at 32 across GPU tiers by trading per-device
    batch for accumulation, so runs on different cards stay comparable.
    """
    ckpt = f"/vol/ckpt/vasudha-{size_tag}{_tag(base_tag)}-{attn}"
    data_root = f"/vol/ckpt/data/{dataset_name or f'sft-{samples}'}"
    # Older successful preparations store Arrow directly at data_root; the
    # resumable collector writes it below data_root/arrow.  Supporting both
    # avoids re-downloading a perfectly usable existing dataset.
    import os
    prepared_path = f"{data_root}/arrow" if os.path.exists(f"{data_root}/arrow/state.json") else data_root

    if GPU.startswith("H100") or "80GB" in GPU:
        # At 2k context the H100 has enough headroom to retain activations;
        # avoiding checkpoint recomputation is a large throughput win.
        bsz = 16 if max_seq_length <= 2048 else 8
    elif GPU.startswith(("A100", "L40S")):
        bsz = 4
    else:
        bsz = 1

    # Per-token activation memory scales roughly with hidden_size, which
    # scales roughly with param count. Halving batch per doubling of size
    # is a conservative guess, not a measurement — an OOM here just means
    # retry with a smaller size_scale, it isn't silent or expensive to see.
    import re as _re
    m = _re.match(r"(\d+)", size_tag)
    size_b = int(m.group(1)) if m else 4
    size_scale = max(1, size_b // 4)
    bsz = max(1, bsz // size_scale)

    accum = max(1, 32 // bsz)
    ckpting = str(not ((GPU.startswith("H100") or "80GB" in GPU) and max_seq_length <= 2048)).lower()

    # GLA's g_proj is freshly random-initialized (see lora_utils.py) — a LoRA
    # delta on top of noise trains nothing, so it must be fully trainable
    # rather than adapter-only. Only hybrid layers have a g_proj at all.
    modules_to_save = "training.lora.modules_to_save=[g_proj] " if attn == "hybrid" else ""

    print(
        f"GPU={GPU} → batch={bsz} accum={accum} seq_len={max_seq_length} "
        f"lora_r={lora_r} checkpoint_attn={attn} train_attn={attention_override} "
        f"packing={packing} size={size_tag}"
    )
    _run(
        "python scripts/train_sft.py "
        # The checkpoint's config is authoritative at from_pretrained time,
        # but select the matching config here too so the tokenizer/reference
        # name and run logs never claim this is a 4B job.
        f"model=qwen3_{size_tag} "
        f"+model.vasudha_path={ckpt} "
        f"+model.attention_override={attention_override} "
        f"output_dir=/vol/ckpt/runs/{_run_dir(size_tag, base_tag, attn)} "
        # '+' because Hydra runs in struct mode and prepared_path is not a key
        # in configs/data/math_reasoning.yaml — a plain override would be
        # rejected as "not in struct".
        f"+data.prepared_path={prepared_path} "
        "training.quantization.load_in_4bit=false "
        "hardware.compute_dtype=bf16 "
        f"training.args.packing={str(packing).lower()} "
        f"training.args.gradient_checkpointing={ckpting} "
        f"training.args.per_device_train_batch_size={bsz} "
        f"training.args.gradient_accumulation_steps={accum} "
        f"training.args.max_seq_length={max_seq_length} "
        "training.args.optim=adamw_torch_fused "
        f"training.args.max_steps={steps} "
        "training.args.save_steps=250 "
        "training.args.logging_steps=10 "
        f"training.lora.r={lora_r} "
        f"training.lora.lora_alpha={lora_alpha} "
        f"{modules_to_save}"
    )
    vol.commit()


@app.function(**UNSLOTH_CONFIG)
def train_unsloth(
    samples: int = 60000,
    dataset_name: str = "sft-combined-60000",
    steps: int = 1250,
    max_seq_length: int = 2048,
) -> None:
    """Fast practical Vasudha Engineering branch: plain Qwen3-8B + QLoRA."""
    data_root = f"/vol/ckpt/data/{dataset_name}"
    prepared = f"{data_root}/arrow" if os.path.exists(f"{data_root}/arrow/state.json") else data_root
    out = "/vol/ckpt/runs/unsloth-qwen3-8b"
    _run(
        "python scripts/train_unsloth.py "
        f"--data {prepared} --out {out} --steps {steps} --max-seq-length {max_seq_length}"
    )
    vol.commit()


@app.function(**UNSLOTH_CONFIG)
def train_unsloth_4b(
    steps: int = 650,
    dataset_name: str = "sft-combined-60000",
    max_seq_length: int = 2048,
    batch_size: int = 4,
    grad_accum: int = 8,
    no_gradient_checkpointing: bool = True,
    out_name: str = "vasudha-leaprunning-4b",
    base_model: str = "Qwen/Qwen3.5-4B",
) -> None:
    """Train and export the compact standalone Vasudha Engineering model.

    out_name defaults to the existing deployed checkpoint's name so old
    callers are unaffected, but a new training run should always pass a
    distinct out_name (e.g. "vasudha-leaprunning-4b-v2") — this used to be
    hardcoded, which meant any re-run silently overwrote the current
    deployed model with an unproven result and no way back.

    base_model was ALSO hardcoded to stock Qwen3.5-4B until caught here —
    a "continue from our v2 checkpoint" run would have silently retrained
    from scratch instead, discarding the previous pass's learning entirely
    (and burning real budget doing it). Pass a local merged checkpoint path
    (e.g. "/vol/ckpt/models/vasudha-leaprunning-4b-v2") to actually build on
    a prior run instead of the stock base."""
    data_root = f"/vol/ckpt/data/{dataset_name}"
    prepared = f"{data_root}/arrow" if os.path.exists(f"{data_root}/arrow/state.json") else data_root
    merged_out = f"/vol/ckpt/models/{out_name}"
    _run(
        "python scripts/train_unsloth.py "
        f"--model {base_model} --data {prepared} --out {merged_out} "
        f"--steps {steps} --max-seq-length {max_seq_length} --merge "
        f"--batch-size {batch_size} --grad-accum {grad_accum} "
        + ("--no-gradient-checkpointing" if no_gradient_checkpointing else "")
    )
    vol.commit()


@app.function(**CPU_CONFIG)
def stage2_pipeline(
    base_dataset: str = "sft-combined-60000",
    instruct_samples: int = 25000,
    tooluse_samples: int = 20600,
    max_seq_length: int = 1024,
    batch_size: int = 16,
    grad_accum: int = 2,
) -> None:
    """One push: build the OpenHermes instruction-following add-on and the
    tool-use add-on, combine both into the existing reasoning mix, then train
    a full epoch over the result. Set VASUDHA_GPU=H100 before calling — the
    prep/combine stages run here on cheap CPU_CONFIG, and only the final
    train_unsloth_4b dispatch (a .remote() call, not .local(): it needs
    unsloth_image, a different image than this function's own) spins up the
    billed GPU container, and only for the actual training time.
    """
    import math

    prepare_instruction_following_data.local(samples=instruct_samples)
    prepare_tool_use_data.local(samples=tooluse_samples)

    intermediate_dataset = "sft-stage2-intermediate"
    combine_data.local(
        base_dataset=base_dataset,
        addon_dataset=f"sft-instruct-{instruct_samples}",
        out_dataset=intermediate_dataset,
    )
    final_dataset = "sft-stage2-final"
    combine_data.local(
        base_dataset=intermediate_dataset,
        addon_dataset=f"sft-tooluse-{tooluse_samples}",
        out_dataset=final_dataset,
    )

    from datasets import load_from_disk

    final_path = f"/vol/ckpt/data/{final_dataset}"
    arrow_path = f"{final_path}/arrow" if os.path.exists(f"{final_path}/arrow/state.json") else final_path
    total_rows = len(load_from_disk(arrow_path))
    effective_batch = batch_size * grad_accum
    steps = math.ceil(total_rows / effective_batch)
    print(
        f"Final combined dataset: {total_rows:,} rows -> {steps:,} steps "
        f"for 1 epoch at effective batch {effective_batch}"
    )

    train_unsloth_4b.remote(
        steps=steps,
        dataset_name=final_dataset,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        grad_accum=grad_accum,
        no_gradient_checkpointing=True,
    )


@app.function(**CPU_CONFIG)
def stage3_pipeline(
    base_dataset: str = "sft-stage2-final",
    reasoning_addon: str = "reasoning-v2",
    out_dataset: str = "sft-stage3-final",
    out_name: str = "vasudha-leaprunning-4b-v2",
    base_model: str = "/vol/ckpt/models/vasudha-leaprunning-4b-v2",
    max_seq_length: int = 2048,
    batch_size: int = 4,
    grad_accum: int = 8,
) -> None:
    """Folds the locally-distilled reasoning-trace + tool-use corpus
    (vasudha/datasets/reasoning_*.py + scripts/generate_reasoning_dataset.py
    and scripts/generate_tool_dataset.py, merged by scripts/merge_datasets.py
    and uploaded via `modal volume put`) into the existing stage-2 mix.

    reasoning_addon is ~375 rows against a 60k+ base dataset — under 1% of
    the final mix, so it dilutes itself naturally. stage2_pipeline's
    explicit instruction-following re-mix (a hard requirement there, since
    that add-on was large enough to swing the "answer directly by default"
    balance on its own) isn't repeated here for the same reason: at this
    scale it wouldn't measurably change anything. That reasoning would need
    revisiting if reasoning_addon is ever grown into a meaningful fraction
    of the total mix.

    out_name defaults to a distinct "-v2" checkpoint, not the currently
    deployed vasudha-leaprunning-4b — this is a new candidate to evaluate,
    not a silent replacement of the working model.

    base_model defaults to the v2 merged checkpoint, not stock Qwen3.5-4B —
    this pipeline exists specifically to build on a prior pass, and an
    earlier version of this function silently dropped that (train_unsloth_4b
    hardcoded the stock base), which would have retrained from scratch and
    discarded whatever the previous run learned without any error or
    warning. Caught via manual CLI dispatch before it could actually run;
    fixed here so the pipeline function itself can't repeat it.
    """
    import math

    combine_data.local(
        base_dataset=base_dataset,
        addon_dataset=reasoning_addon,
        out_dataset=out_dataset,
    )

    from datasets import load_from_disk

    final_path = f"/vol/ckpt/data/{out_dataset}"
    arrow_path = f"{final_path}/arrow" if os.path.exists(f"{final_path}/arrow/state.json") else final_path
    total_rows = len(load_from_disk(arrow_path))
    effective_batch = batch_size * grad_accum
    steps = math.ceil(total_rows / effective_batch)
    print(
        f"Final combined dataset: {total_rows:,} rows -> {steps:,} steps "
        f"for 1 epoch at effective batch {effective_batch}"
    )

    train_unsloth_4b.remote(
        steps=steps,
        dataset_name=out_dataset,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        grad_accum=grad_accum,
        no_gradient_checkpointing=True,
        base_model=base_model,
        out_name=out_name,
    )


@app.function(**GPU_CONFIG)
def compare_engineering(
    vasudha_run: str = "8b-hybrid-bf16",
    qwen_run: str = "unsloth-qwen3-8b",
) -> None:
    """Run identical engineering prompts through both trained branches."""
    vasudha_adapter = f"/vol/ckpt/runs/{vasudha_run}/checkpoints/final_model"
    qwen_adapter = f"/vol/ckpt/runs/{qwen_run}"
    _run(
        "python scripts/compare_engineering.py "
        f"--qwen-adapter {qwen_adapter} "
        f"--vasudha-base /vol/ckpt/vasudha-8b-hybrid "
        f"--vasudha-adapter {vasudha_adapter} "
        "--out /vol/ckpt/runs/engineering-comparison.json"
    )
    vol.commit()


@app.function(**GGUF_CONFIG)
def convert_to_gguf(
    model_dir: str = "vasudha-leaprunning-4b",
    quant_types: str = "Q4_K_M,Q5_K_M",
) -> None:
    """Convert the merged standalone model to GGUF (f16 intermediate), then
    quantize to each requested type. CPU-only — no GPU needed for format
    conversion or quantization, so this stays off the billed GPU tier.

    --no-mtp drops the multi-token-prediction head: this model's 32 main
    layers plus 1 trailing MTP layer got GGUF-converted as if all 33 were
    regular decoder blocks, and older llama.cpp/Ollama loaders (anything
    before Ollama ~0.31) require attn_qkv/attn_gate on every block — they
    don't yet know to treat a trailing block as an MTP head instead, so they
    fail with "layer 32 missing attn_qkv/attn_gate projections". The MTP
    head is only useful for speculative decoding, not needed for normal
    generation, so dropping it entirely is simpler and more portable than
    requiring a specific bleeding-edge Ollama version.
    """
    import os

    model_path = f"/vol/ckpt/models/{model_dir}"
    out_dir = f"/vol/ckpt/gguf/{model_dir}"
    os.makedirs(out_dir, exist_ok=True)

    f16_path = f"{out_dir}/{model_dir}-f16.gguf"
    _run(
        f"python /root/llama.cpp/convert_hf_to_gguf.py {model_path} "
        f"--outfile {f16_path} --outtype f16 --no-mtp"
    )
    for quant in quant_types.split(","):
        quant = quant.strip()
        quant_path = f"{out_dir}/{model_dir}-{quant}.gguf"
        _run(f"/root/llama.cpp/build/bin/llama-quantize {f16_path} {quant_path} {quant}")
    vol.commit()
    print(f"GGUF files ready under {out_dir}")


@app.function(**UNSLOTH_CONFIG)
def chat_once(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> None:
    """Windows-friendly one-prompt chat entrypoint; model stays on Modal GPU."""
    import shlex
    adapter = "/vol/ckpt/runs/unsloth-qwen3-8b/checkpoint-250"
    command = (
        "python scripts/chat_bench.py --base Qwen/Qwen3-8B "
        f"--adapter {adapter} --load-in-4bit --max-new-tokens {max_new_tokens} "
        f"--temperature {temperature} --prompt {shlex.quote(prompt)}"
    )
    _run(command)


# Lowest-cost practical serving target with comfortable memory for QLoRA Qwen3-8B.
# Set VASUDHA_GPU=L4 only for training; this endpoint intentionally stays cheap.
ENDPOINT_CONFIG = {**UNSLOTH_CONFIG, "gpu": "L4", "scaledown_window": 300}
_endpoint_model = None
_endpoint_tokenizer = None


@app.function(**ENDPOINT_CONFIG)
@modal.fastapi_endpoint(method="POST", label="vasudha-chat", docs=True, requires_proxy_auth=True)
def vasudha_chat(item: dict) -> dict:
    """HTTP chat endpoint for the local HTML bench (development endpoint)."""
    global _endpoint_model, _endpoint_tokenizer
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if _endpoint_model is None:
        _endpoint_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
        _endpoint_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-8B",
            dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=__import__("transformers").BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            ),
        )
        _endpoint_model = PeftModel.from_pretrained(
            _endpoint_model, "/vol/ckpt/runs/unsloth-qwen3-8b/checkpoint-250"
        ).merge_and_unload()
        _endpoint_model.eval()

    messages = item.get("messages") or [{"role": "user", "content": item.get("prompt", "")}]
    encoded = _endpoint_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    input_ids = (encoded["input_ids"] if hasattr(encoded, "__getitem__") and not hasattr(encoded, "shape") else encoded).to(_endpoint_model.device)
    with torch.inference_mode():
        output = _endpoint_model.generate(
            input_ids=input_ids, max_new_tokens=min(int(item.get("max_new_tokens", 512)), 1024),
            do_sample=float(item.get("temperature", 0.7)) > 0,
            temperature=max(float(item.get("temperature", 0.7)), 1e-5), top_p=0.9, use_cache=True,
        )
    text = _endpoint_tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()
    thinking, final = text, text
    if "<think>" in text:
        thinking = text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
        final = text.split("</think>", 1)[-1].strip()
    return {"thinking": thinking, "answer": final, "raw": text}


@app.function(**GPU_CONFIG)
def pipeline(steps: int = 400, samples: int = 20000, eval_samples: int = 200) -> None:
    """
    Everything end to end in one container: one GPU acquisition, no requeueing
    between stages.

    Gated on the parity check — if the dense conversion does not reproduce stock
    Qwen3, every downstream number is meaningless and the run aborts rather than
    burning an hour to produce one.
    """
    def stage(name: str) -> None:
        print(f"\n{'=' * 70}\n  {name}\n{'=' * 70}", flush=True)

    stage("1/7  Parity check: dense conversion vs stock Qwen3")
    _run("python scripts/diagnose_conversion.py /vol/ckpt/vasudha-4b-sdpa")

    # Stages are skipped when their output already exists on the Volume, so a
    # rerun after a mid-pipeline failure resumes instead of repeating work.
    import os

    stage("2/7  Prepare dataset")
    data_dir = f"/vol/ckpt/data/sft-{samples}"
    if os.path.exists(f"{data_dir}/arrow/state.json"):
        print(f"{data_dir}/arrow already exists — skipping")
    else:
        _run(
            f"python scripts/prepare_dataset.py --out {data_dir} "
            f"--max-samples {samples}"
        )
        vol.commit()

    # base_tag=NO_TAG throughout: this harness compares dense vs hybrid on the
    # original plain-Qwen3-4B conversion, kept separate from the tagged
    # Thinking-2507 checkpoints the `train`/`evaluate`/`merge` defaults build.
    stage("3/7  Train dense (control)")
    train.local(attn="sdpa", steps=steps, samples=samples, base_tag=NO_TAG)

    stage("4/7  Evaluate dense")
    evaluate.local(attn="sdpa", samples=eval_samples, base_tag=NO_TAG)

    stage("5/7  Convert hybrid (GLA)")
    if os.path.exists("/vol/ckpt/vasudha-4b-hybrid"):
        print("hybrid checkpoint already exists — skipping")
    else:
        convert.local(attn="hybrid", moe=False, base_tag=NO_TAG)

    stage("6/7  Train hybrid")
    train.local(attn="hybrid", steps=steps, samples=samples, base_tag=NO_TAG)

    stage("7/7  Evaluate hybrid")
    evaluate.local(attn="hybrid", samples=eval_samples, base_tag=NO_TAG)

    vol.commit()
    print("\nDone. Compare the two GSM8K numbers: that difference is the GLA cost.")


@app.function(**GPU_CONFIG)
def evaluate(
    attn: str = "sdpa", samples: int = 200, base_tag: str = BASE_TAG, size_tag: str = "4b", attention_override: str = "sdpa"
) -> None:
    """GSM8K on a trained checkpoint."""
    run_dir = _run_dir(size_tag, base_tag, attn)
    _run(
        # final_model, not the checkpoints root — the root holds checkpoint-N
        # subdirectories and no adapter of its own.
        f"python scripts/evaluate.py "
        f"--model_path /vol/ckpt/runs/{run_dir}/checkpoints/final_model "
        f"--base_path /vol/ckpt/vasudha-{size_tag}{_tag(base_tag)}-{attn} "
        f"--attention-override {attention_override} "
        f"--benchmarks gsm8k --max_samples {samples}"
    )


@app.function(**GPU_CONFIG)
def merge(attn: str = "sdpa", base_tag: str = BASE_TAG, size_tag: str = "4b", attention_override: str = "sdpa") -> None:
    """Fold the trained LoRA adapter into the base weights — one standalone
    checkpoint directory instead of adapter + base kept separately."""
    run_dir = _run_dir(size_tag, base_tag, attn)
    _run(
        f"python scripts/merge_adapter.py "
        f"--adapter_path /vol/ckpt/runs/{run_dir}/checkpoints/final_model "
        f"--base_path /vol/ckpt/vasudha-{size_tag}{_tag(base_tag)}-{attn} "
        f"--attention-override {attention_override} "
        f"--out /vol/ckpt/vasudha-{size_tag}{_tag(base_tag)}-merged"
    )
    vol.commit()
