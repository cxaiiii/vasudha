"""Vasudha Neo — on-policy post-training for the Vasudha engineering assistant.

Every gradient in this package comes from the model's own samples. There is no
supervised fine-tuning step anywhere, deliberately:

    "RL's Razor: Why Online Reinforcement Learning Forgets Less"
    (Shenfeld, Pari, Agrawal — ICLR 2026)

showed that forgetting is predicted by how far fine-tuning moves the model from
its own distribution (KL measured on the new task), and that on-policy methods
are implicitly biased toward the KL-minimal way of solving a task, while SFT on
someone else's text can drag the model arbitrarily far. v1-v3 of Vasudha were
built exactly that way (LoRA SFT on OpenThoughts / NuminaMath / DeepSeek
traces, each version stacked on the last), which is the textbook setup for
erasing what the base model already knew.

Neo restarts from the stock base and uses only two on-policy signals:

  1. On-policy distillation (stage 1) — the student samples, a bigger model of
     the same family scores every token the student wrote (reverse KL).
     Dense signal, no labels needed, no off-policy text.
  2. RL with verifiable rewards in a real Python sandbox (stage 2) — GRPO with
     rewards that check the committed number and its unit, run code against
     hidden tests, and match tool calls exactly.

Nothing in this package imports the legacy ``vasudha`` model stack: that
package eagerly imports its own model code on import, and none of it is
needed here.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
