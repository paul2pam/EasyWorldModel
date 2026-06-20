# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An in-progress, simplified implementation of [DreamerV3](https://arxiv.org/abs/2301.04104) — a model-based RL world model. The goal is to keep it as readable and minimal as possible.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: PyTorch 2.2.2, Gymnasium, NumPy.

## Architecture

The world model is a **Recurrent State Space Model (RSSM)** with three state variables:

- `e` — encoded observation (output of `Encoder`)
- `h` — deterministic recurrent state (output of `SequenceModel` / GRU)
- `z` — stochastic discrete latent state (output of `Prior` or `Posterior`)

### Components (`rssm/`)

**[encoder.py](rssm/encoder.py)** — `Encoder(embed_dim)`: 4-layer CNN (3→32→64→128→256 channels, stride-2 convolutions) + flatten + linear. Takes raw pixel observations `(B, C, H, W)` and outputs a flat embedding vector of size `embed_dim`. Hardcoded for 64×64 RGB input (linear layer expects 4096 features after convolutions).

**[rssm.py](rssm/rssm.py)** — contains four classes:

- `SequenceModel(deter_dim, discrete_dim, action_dim)` — GRUCell-based deterministic recurrence. Concatenates `z` and `a`, projects through a linear layer, then runs a GRUCell to produce `h_new`. Note: uses a plain GRU, not the block-diagonal variant from the paper.
- `Posterior(embed_dim, deter_dim)` — MLP that maps `(e, h)` → `z`. Used during training when observations are available.
- `Prior(deter_dim)` — MLP that maps `h` → `z`. Used during imagination rollouts (no observation).
- `RSSM` — top-level module wiring the above. Exposes:
  - `observation_step(h, z, a, e)` → `(h_new, z_new)` — one step with real observation
  - `imagination_step(h, z, a)` → `(h_new, z_new)` — one step without observation (pure prediction)
  - `initial(batch_size)` — stub, not yet implemented

### Known TODOs in the code

- `RSSM.__init__` calls `SequenceModel()`, `Prior()`, `Posterior()` with no arguments — these constructors require dims, so the class is not yet functional.
- `initial()` is not implemented.
- Several dims are hardcoded (`hidden_dim=512`, `stoch_dim=512`) with `#TODO: un-hardcode` comments.
- `z` should be a categorical/discrete distribution in DreamerV3, but currently the MLP outputs a raw tensor (no discretization or straight-through gradient).
- `in_features=4096` in the encoder is hardcoded for 64×64 input.
