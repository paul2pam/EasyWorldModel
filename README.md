# EasyWorldModel
A dreaming world model bassed off [DreamerV3](https://arxiv.org/abs/2301.04104). 
Stripped of everything except the basic fucntional code. So simple your grandma could use it. 

## Setup
```bash
brew install swig
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python train.py    # collect CarRacing-v2 data and train the world model
python render.py   # load checkpoint and save a 100-frame imagination rollout to imagination.gif

#that's it!!
```

## Architecture
The world model is an RSSM with three state variables:
- `e` — encoded observation (CNN encoder)
- `h` — deterministic recurrent state (GRU)
- `z` — stochastic latent state (MLP, prior or posterior)

Currently only supports the CarRacingV2 environment as a proof of concept. Working on expanding general support for all environements.
