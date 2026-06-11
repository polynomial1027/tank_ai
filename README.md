# Tank AI

A complete pygame + PyTorch project for experimenting with CNN-DQN agents in a simple two-player tank battle environment.

The project includes:

- Pygame tank battle environment
- Human vs human, human vs random, human vs model, model vs model test UI
- CNN-DQN model and agent
- Replay buffer
- Reward functions in a separate file for debugging and experiments
- Graphical training launcher UI
- Training logs and plots
- Tests

## Project layout

```text
tank_ai/
├── README.md
├── requirements.txt
├── .gitignore
├── run_training_ui.py
├── run_battle_ui.py
└── tank_ai/
    ├── src/
    │   ├── train_cnn_dqn.py
    │   ├── training_ui.py
    │   ├── battle_ui.py
    │   ├── play.py
    │   ├── env/
    │   │   └── tank_env.py
    │   ├── agents/
    │   │   ├── random_agent.py
    │   │   └── cnn_dqn_agent.py
    │   ├── models/
    │   │   └── cnn_dqn.py
    │   ├── rewards/
    │   │   └── reward_functions.py
    │   └── utils/
    │       ├── config.py
    │       ├── replay_buffer.py
    │       ├── plotting.py
    │       └── timer.py
    └── tests/
```

## Install

```bash
conda create -n tank_ai python=3.11 -y
conda activate tank_ai
pip install -r requirements.txt
```

For Windows with NVIDIA GPU, install CUDA PyTorch separately:

```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

Check CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

## Start graphical training UI

```bash
python run_training_ui.py
```

The UI lets you set:

- Training episodes
- Render interval
- Save interval
- Reward mode
- Checkpoint path
- Resume checkpoint

## Start battle/test UI

```bash
python run_battle_ui.py
```

The battle UI lets you choose each side:

- Human WASD
- Human arrow keys
- Random agent
- CNN model loaded from `.pth`

It also lets you browse for model files.

## Command-line CNN-DQN training

Basic training:

```bash
python -m tank_ai.src.train_cnn_dqn --episodes 500 --render-every 0 --save-every 50 --reward balanced --checkpoint tank_ai/checkpoints/cnn_balanced_latest.pth
```

With rendering every 50 episodes:

```bash
python -m tank_ai.src.train_cnn_dqn --episodes 500 --render-every 50 --save-every 50 --reward balanced --checkpoint tank_ai/checkpoints/cnn_balanced_latest.pth
```

Resume training:

```bash
python -m tank_ai.src.train_cnn_dqn --episodes 500 --render-every 0 --reward balanced --checkpoint tank_ai/checkpoints/cnn_balanced_latest.pth --resume tank_ai/checkpoints/cnn_balanced_latest.pth
```

## Training parameters

```text
--episodes       Number of training episodes
--render-every   Render one episode every N episodes. Use 0 to disable rendering.
--save-every     Save model every N episodes
--reward         Reward mode
--checkpoint     Path to save model checkpoint
--resume         Optional checkpoint to resume from
--speed          Pygame render speed
```

## Reward modes

Reward functions are in:

```text
tank_ai/src/rewards/reward_functions.py
```

Current modes:

```text
balanced
aggressive
survival
hit_reward
```

### balanced

General default reward:

```text
Win: +20
Lose: -20
Hit enemy: +8
Get hit: -8
Fire: -0.03
Move: -0.01
Wall bump: -0.2
Timeout: -2
```

### aggressive

Encourages shooting and hitting:

```text
Win: +20
Lose: -20
Hit enemy: +12
Get hit: -6
Fire: -0.01
Move: -0.005
Wall bump: -0.15
Timeout: -1
```

### survival

Encourages avoiding damage:

```text
Win: +25
Lose: -25
Hit enemy: +6
Get hit: -12
Fire: -0.04
Move: -0.005
Wall bump: -0.3
Timeout: 0
```

### hit_reward

Simpler shooting reward:

```text
Win: +10
Lose: -10
Hit enemy: +15
Get hit: -5
Fire: -0.02
Move: 0
Wall bump: -0.1
Timeout: 0
```

## Watch a trained model

From the battle UI:

```bash
python run_battle_ui.py
```

Choose `Model file` for one side and select your `.pth` checkpoint.

Command-line quick model vs random:

```bash
python -m tank_ai.src.play --left model --right random --left-model tank_ai/checkpoints/cnn_balanced_latest.pth
```

Human WASD vs model:

```bash
python -m tank_ai.src.play --left human_wasd --right model --right-model tank_ai/checkpoints/cnn_balanced_latest.pth
```

## Run tests

```bash
pytest tank_ai/tests -v
```

## Git note

Do not commit model checkpoints or run outputs. `.gitignore` already ignores:

```gitignore
tank_ai/checkpoints/*.pth
tank_ai/runs/
```

## GitHub upload flow

Create a new repository on GitHub named `tank_ai`, then run:

```bash
cd tank_ai
git init
git add .
git commit -m "initialize tank ai cnn project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/tank_ai.git
git push -u origin main
```

If you have GitHub CLI:

```bash
cd tank_ai
git init
git add .
git commit -m "initialize tank ai cnn project"
gh repo create tank_ai --public --source=. --remote=origin --push
```


## Custom maps

This project supports JSON map files.

Built-in maps:

```bash
python run_map_editor.py
```

Example map paths:

```text
tank_ai/maps/classic.json
tank_ai/maps/open_field.json
```

Map JSON format:

```json
{
  "name": "classic",
  "cols": 20,
  "rows": 15,
  "p1_spawn": [1, 7],
  "p2_spawn": [18, 7],
  "walls": [[10, 4], [10, 5], [10, 6]]
}
```

Train with a custom map:

```bash
python -m tank_ai.src.train_cnn_dqn \
  --episodes 500 \
  --render-every 0 \
  --save-every 50 \
  --reward balanced \
  --map tank_ai/maps/classic.json \
  --checkpoint tank_ai/checkpoints/cnn_balanced_classic_map.pth
```

Play/test with a custom map:

```bash
python -m tank_ai.src.play \
  --left human_wasd \
  --right model \
  --right-model tank_ai/checkpoints/cnn_balanced_classic_map.pth \
  --map tank_ai/maps/classic.json
```

## Training data and plot export

Every episode prints detailed data in the terminal, including score, total reward, winner, health, steps, loss, epsilon, hit status, wall bump status, and time.

Training automatically saves:

```text
tank_ai/runs/<checkpoint_name>_training.csv
tank_ai/runs/<checkpoint_name>_training.png
```

You can also choose custom output paths:

```bash
python -m tank_ai.src.train_cnn_dqn \
  --episodes 500 \
  --reward balanced \
  --checkpoint tank_ai/checkpoints/cnn_balanced_latest.pth \
  --csv exports/my_training_data.csv \
  --plot exports/my_training_plot.png
```

The graphical training UI also contains CSV and plot output fields, plus buttons to export existing CSV or PNG files.

---

## Training UI: live progress and parallel training

Start the training UI:

```bash
python run_training_ui.py
```

The training UI now shows a live progress table for every active run:

```text
Run
Status
Episode
Reward
Score
Total reward
Winner
HP
Steps
Avg100
Loss
Epsilon
Episode time
```

The raw terminal output is also shown in a separate tab for each run.

### Parallel training windows

The `Parallel training windows` option can be set to:

```text
1
2
4
```

This launches 1, 2, or 4 independent training processes at the same time. On a stronger computer, this lets you run multiple experiments simultaneously.

If `Render every` is greater than 0, each process may open its own pygame render window. For maximum speed, use:

```text
Render every = 0
```

When running multiple training processes, the UI automatically adds suffixes to output files:

```text
cnn_balanced_latest_run1.pth
cnn_balanced_latest_run2.pth
cnn_balanced_latest_run3.pth
cnn_balanced_latest_run4.pth
```

The same suffix rule applies to CSV and PNG export paths.
