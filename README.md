# RL Chihuahua Game

> A 2D Chihuahua-themed reinforcement learning environment built with Pygame and Gymnasium.

---

## Overview

2D Pygame game and environment for reinforcement learning. It includes:

* A custom RL environment compatible with Gymnasium
* Modular game structure using Pygame (clean architecture)
* **Object-Oriented Programming (OOP)** for organizing game logic

Check out my another work in progress project: [Car Game](https://github.com/uma-dev/car-game).

---

## Table of Contents

1. [Features](#-features)
2. [Prerequisites](#-prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
5. [RL Setup](#-rl-setup)
6. [Creating Custom Levels](#custom-levels)
7. [Contributing](#-contributing)

---

## Features

* Discrete Reinforcement Learning environment
* Collision-based rewards: hitting a target tile
* Partial rewards: getting the ball near to the target tile
* Frame-based animation, sprites, basic physics and level loading
* Easy to integrate with Gym-based RL pipelines
* Playable characters with simple animations (render states)

<p align="center">
  <img alt="Digitalization" src="https://github.com/user-attachments/assets/4cbf36be-3d27-47a0-a164-aeee6a9abd8a">
</p>

### Project structure

Modular components: game logic, assets, environment definitions, and training scripts:

``` bash
├── assets/              # Game assets (sprites, tiles)
│   └── images/          # Character and tile images
│       └── tiles/       # Grass, wall, and ring textures

├── engine/              # Core game engine and state manager
│   └── game_engine.py   # Game loop and scene handling

├── entities/            # Game entities like the character, ball, and target

├── gym_ext/             # Gym-compatible environment and training script
│   ├── env.py           # Gymnasium environment wrapper
│   └── train.py         # PPO or other RL agent training logic

├── logic/               # Game physics and collision handling

├── logs/                # TensorBoard logs for training visualization

├── map/                 # Level definitions (JSON) and level loader

├── models/              # (Optional) Trained models can be saved here

├── screens/             # UI screens (menu, gameplay, win state)

└── utils/               # Constants, game states, shared helpers
```

---

## Prerequisites

* **Python 3.7+**
* uv (recommended)
* modules: pygame, gymnasium, tensorboard (see requirements.txt)

---

<a name="installation"></a>

## Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/uma-dev/chihuahua-game.git
   cd chihuahua-game
   ```

2. (Optional) Create a virtual environment:

   ```bash
   uv venv 
   source .venv/bin/activate  # Linux/macOS
   .\.venv\Scripts\activate  # Windows
   ```

3. Install dependencies:

   ```bash
   uv pip install -r requirements.txt
   ```

---

<a name="usage"></a>

## Usage

Play the game with: (Use arrow keys (← ↑ → ↓) to move your chihuahua. Enjoy!)

```bash
python game.py
```

Train the model and observe the training process with:

```bash
python main.py train_render 
```

Train the model without render:

```bash
python main.py train_no_render 
```

Eval the policy with:

```bash
python main.py eval
```

Look training in real time with:

```bash
 tensorboard --logdir .\logs\ppo_tensorboard\
```

---

## RL Setup

### Description and goal

Chihuahua 2D videogame where the agent controls a chihuahua character that must hit and push a ball toward a target tile. The environment is built with Pygame and wrapped using Gymnasium API, so its compatible with RL algorithms.

The agent's objective is to learn how to move and act efficiently to guide the ball to the target, maximizing the reward and minimizing steps (penalty).

<p align="center">
  <img
    alt="Executing Game.py"
    src="https://github.com/user-attachments/assets/fad08b8e-f179-4974-bcd4-360608192ee5">
</p>

### Problem Formulation

* **Algorithm:** PPO (Proximal Policy Optimization)
* **Policy:** MLP (Multilayer Perceptron)
* **Library:** Stable-Baselines3
* **Logging:** TensorBoard (`logs/ppo_tensorboard/`)

**State Space (Observation):**
A vector of 10 values:

```python
[char_x, char_y, char_vx, char_vy, ball_x, ball_y, ball_vx, ball_vy, char_jumping, char_sprinting]
```

**Action Space:**
Discrete(5):

```python
0 = No-op, 1 = Left, 2 = Right, 3 = Jump, 4 = Sprint
```

**Reward Function: (work in progress, as it directly changes the learning curve)** 

```python
if ball_hits_target(self.ball, self.target):  # win condition
  reward += 10.0
  # An episode only ends when the target is hitted
  done = True
else:
  # penalty for each step
  reward -= 0.01
```

Training metrics are logged with **TensorBoard**, so you can monitor the agent's behavior over time:

```bash
tensorboard --logdir logs/
```

---

### Total Reward per Episode

This shows the sum of all rewards per episode, as training progresses:

<p align="center">
  <img
    alt="Reward"
    src="https://github.com/user-attachments/assets/6d7a882b-f4e3-44ae-953a-305a826d806b">
</p>

- **Initial plateau**: The agent explores randomly without reward shaping.
- **Reward drop**: Movement/jump penalties and progress bonus in the reward avoid this behaivor (?).
- **Late improvement**: Reward increases as the agent learns to interact better.

> ⚠️ Still in progress: refining the **state representation** (possibly using CNNs) and improving the **reward function**.

---

### Mean Episode Length

This plot shows how long episodes last (in steps), averaged over evaluations:

<p align="center">
  <img
    alt="Reward"
    src="https://github.com/user-attachments/assets/b5aa4794-870e-495c-bab2-967a45245a5a">
</p>

- A strange behavior is observed at first, possibly due to the agent exploring or stalling.
- Eventually, the episode length drops, which **aligns with increasing reward** — the agent finishes the task more efficiently.
- The **minimum mean length** reached is **~1727 steps**.

> ⚠️ When evaluating the policy, consider this number as a typical episode length.

---

### Training stats

- **Duration**: ~2.5 hours  
- **Steps**: 500,000  
- **Eval Interval**: Every 5,000 steps  
  (Note: Since episodes last ~1.7k steps, more than one episode might occur per eval.)

---

### Bloopers

#### Reward pitfall

Partial rewards can introduce unintended behaviors during training. In this blooper, the reward setup was wrong. Aiming to improve learning curves, I gave partial rewards to agent: either when moving the ball closer to the target or when hitting the ball. However, these partial rewards per step were greater than the penalty for each step. So, the total reward per step was positive!

As a result, the agent learned to exploit this by only looking for partial rewards, so the episode mean also grows as is show in the results: 

<p align="center">
  <img
    alt="Reward pitfall"
    src="https://github.com/user-attachments/assets/6fc98d07-4955-4a2a-b900-93223bb9236e">
</p>

> ⚠️ Looks ok, but take a loot to the episode lenght.


<p align="center">
  <img
    alt="Episode lenght pitfall"
    src="https://github.com/user-attachments/assets/a0899cd4-a7f3-45f4-b1ae-60b50e6f1e6b">
</p>

> ⚠️ The result of the reward pitfall. The agent learned to look for partial rewards.

---
---

<a name="custom-levels"></a>

## Custom Levels

Define JSON layout files in the levels/ folder

Add custom tilemaps, special blocks, or enemies

---

## Contributing

1. Fork .
2. Feature branch: `git checkout -b feature/my-feature`
3. Commit: `git commit -m "Add my feature"`
4. Push: `git push origin feature/my-feature`
5. Pull Request.

---
