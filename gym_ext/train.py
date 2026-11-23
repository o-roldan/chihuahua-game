import os
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from gym_ext.env import ChihuahuaEnv

MODEL_PATH = "models/ppo_chihuahua"
LOGS_PATH = "logs/ppo_tensorboard/"

# Episode cap -> `truncated=True` (not `terminated`), so SB3 bootstraps V(s_T).
# Random policy wins ~1 per 7k steps; 15k ≈ 2x that, a safe upper bound.
MAX_EPISODE_STEPS = 15_000


def make_env(render_mode=None):
    env = ChihuahuaEnv(render_mode=render_mode)
    return TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)


def train(render_mode=None):
    env = make_env(render_mode=render_mode)

    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"Loading existing model from {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=env)
        model.set_env(env)
    else:
        print(f"Training new model and saving to {MODEL_PATH}")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOGS_PATH)

    model.learn(total_timesteps=500_000)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def evaluate():
    env = make_env(render_mode="human")
    model = PPO.load(MODEL_PATH)
    obs, _ = env.reset()

    for _ in range(5000):
        action, _ = model.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            reason = "goal reached" if terminated else "time limit"
            print(f"Episode finished ({reason}). Resetting env.")
            obs, _ = env.reset()
