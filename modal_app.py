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

image = (
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
    )
    # Ship the repo into the image. Rebuilds on change, so edits land without
    # a manual upload step.
    .add_local_dir(".", remote_path="/root/vasudha", ignore=["*.pyc", "__pycache__", ".git"])
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
)


def _run(cmd: str) -> None:
    """Run a shell command in the repo, streaming output, failing loudly."""
    import subprocess

    print(f"\n$ {cmd}\n", flush=True)
    result = subprocess.run(cmd, shell=True, cwd="/root/vasudha")
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit {result.returncode}: {cmd}")


@app.function(**GPU_CONFIG)
def convert(attn: str = "sdpa", moe: bool = False) -> None:
    """Convert Qwen3-4B into a Vasudha checkpoint on the Volume."""
    out = f"/vol/ckpt/vasudha-4b-{attn}{'-moe' if moe else ''}"
    flags = "" if moe else "--no-moe"
    _run(
        f"python scripts/convert_qwen3_to_vasudha.py --model Qwen/Qwen3-4B "
        f"--out {out} --attention-type {attn} {flags} --dtype bfloat16"
    )
    vol.commit()


@app.function(**GPU_CONFIG)
def diagnose(attn: str = "sdpa") -> None:
    """Verify a converted checkpoint matches stock Qwen3 layer by layer."""
    _run(f"python scripts/diagnose_conversion.py /vol/ckpt/vasudha-4b-{attn}")


@app.function(**GPU_CONFIG)
def prepare_data(samples: int = 20000) -> None:
    """Materialize the streaming mixture to the Volume, once."""
    _run(
        f"python scripts/prepare_dataset.py --out /vol/ckpt/data/sft-{samples} "
        f"--max-samples {samples}"
    )
    vol.commit()


@app.function(**GPU_CONFIG)
def train(attn: str = "sdpa", steps: int = 400, samples: int = 20000) -> None:
    """LoRA SFT in bf16. No quantization — an L40S has the VRAM to skip it."""
    ckpt = f"/vol/ckpt/vasudha-4b-{attn}"
    # Effective batch stays 16 either way, so runs on different cards remain
    # directly comparable — only the memory/recompute tradeoff changes.
    #
    # Gradient checkpointing stays ON even on 48GB. Retaining activations for
    # 36 layers at 2048 tokens costs ~35GB at batch 8 (the SwiGLU intermediate
    # is 9728 wide, three tensors per layer), which OOMs an L40S before step 1.
    bsz, accum = (4, 4) if BIG_GPU else (2, 8)
    ckpting = "true"
    print(f"GPU={GPU} → batch={bsz} accum={accum} checkpointing={ckpting}")
    _run(
        "python scripts/train_sft.py "
        f"model.vasudha_path={ckpt} "
        f"output_dir=/vol/ckpt/runs/{attn}-bf16 "
        # '+' because Hydra runs in struct mode and prepared_path is not a key
        # in configs/data/math_reasoning.yaml — a plain override would be
        # rejected as "not in struct".
        f"+data.prepared_path=/vol/ckpt/data/sft-{samples} "
        "training.quantization.load_in_4bit=false "
        "hardware.compute_dtype=bf16 "
        "training.args.packing=false "
        f"training.args.gradient_checkpointing={ckpting} "
        f"training.args.per_device_train_batch_size={bsz} "
        f"training.args.gradient_accumulation_steps={accum} "
        "training.args.max_seq_length=2048 "
        "training.args.optim=adamw_torch_fused "
        f"training.args.max_steps={steps} "
        "training.args.save_steps=100 "
        "training.args.logging_steps=10"
    )
    vol.commit()


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
    if os.path.exists(data_dir):
        print(f"{data_dir} already exists — skipping")
    else:
        _run(
            f"python scripts/prepare_dataset.py --out {data_dir} "
            f"--max-samples {samples}"
        )
        vol.commit()

    stage("3/7  Train dense (control)")
    train.local(attn="sdpa", steps=steps, samples=samples)

    stage("4/7  Evaluate dense")
    evaluate.local(attn="sdpa", samples=eval_samples)

    stage("5/7  Convert hybrid (GLA)")
    if os.path.exists("/vol/ckpt/vasudha-4b-hybrid"):
        print("hybrid checkpoint already exists — skipping")
    else:
        convert.local(attn="hybrid", moe=False)

    stage("6/7  Train hybrid")
    train.local(attn="hybrid", steps=steps, samples=samples)

    stage("7/7  Evaluate hybrid")
    evaluate.local(attn="hybrid", samples=eval_samples)

    vol.commit()
    print("\nDone. Compare the two GSM8K numbers: that difference is the GLA cost.")


@app.function(**GPU_CONFIG)
def evaluate(attn: str = "sdpa", samples: int = 200) -> None:
    """GSM8K on a trained checkpoint."""
    _run(
        # final_model, not the checkpoints root — the root holds checkpoint-N
        # subdirectories and no adapter of its own.
        f"python scripts/evaluate.py "
        f"--model_path /vol/ckpt/runs/{attn}-bf16/checkpoints/final_model "
        f"--base_path /vol/ckpt/vasudha-4b-{attn} "
        f"--benchmarks gsm8k --max_samples {samples}"
    )
