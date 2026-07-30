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
        "sentencepiece",
        "protobuf",
    )
    # Ship the repo into the image. Rebuilds on change, so edits land without
    # a manual upload step.
    .add_local_dir(".", remote_path="/root/vasudha", ignore=["*.pyc", "__pycache__", ".git"])
)

# HF_HOME on the Volume is what makes the 8GB Qwen3 download a one-time cost.
ENV = {"HF_HOME": "/vol/hf", "PYTHONPATH": "/root/vasudha", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

GPU_CONFIG = dict(
    gpu="L40S",
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
    _run(
        "python scripts/train_sft.py "
        f"model.vasudha_path={ckpt} "
        f"output_dir=/vol/ckpt/runs/{attn}-bf16 "
        f"data.prepared_path=/vol/ckpt/data/sft-{samples} "
        "training.quantization.load_in_4bit=false "
        "hardware.compute_dtype=bf16 "
        "training.args.packing=false "
        "training.args.gradient_checkpointing=false "
        "training.args.per_device_train_batch_size=8 "
        "training.args.gradient_accumulation_steps=2 "
        "training.args.max_seq_length=2048 "
        "training.args.optim=adamw_torch_fused "
        f"training.args.max_steps={steps} "
        "training.args.save_steps=100 "
        "training.args.logging_steps=10"
    )
    vol.commit()


@app.function(**GPU_CONFIG)
def evaluate(attn: str = "sdpa", samples: int = 200) -> None:
    """GSM8K on a trained checkpoint."""
    _run(
        f"python scripts/evaluate.py --model_path /vol/ckpt/runs/{attn}-bf16/checkpoints "
        f"--benchmarks gsm8k --max_samples {samples}"
    )
