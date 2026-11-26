import argparse
import glob
import os
import sys
import time
from gym_ext.train import MODELS_DIR, train, evaluate


def choose_model():
    """List saved models (newest first) and ask which one to evaluate."""
    paths = sorted(
        glob.glob(os.path.join(MODELS_DIR, "*.zip")), key=os.path.getmtime, reverse=True
    )
    if not paths:
        sys.exit(f"No models found in {MODELS_DIR}/")

    for i, p in enumerate(paths):
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
        print(f"  [{i}] {os.path.basename(p):<40} {stamp}")

    while True:
        raw = input("Model to evaluate [0]: ").strip() or "0"
        if raw.isdigit() and int(raw) < len(paths):
            return paths[int(raw)]
        print(f"Enter a number between 0 and {len(paths) - 1}.")


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate Chihuahua RL agent")
    parser.add_argument(
        "mode",
        choices=["train_render", "train_no_render", "eval"],
        help="Run training or evaluation",
    )
    parser.add_argument("--tag", default="exp", help="experiment name (train)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (train)")
    parser.add_argument(
        "--random-start",
        action="store_true",
        help="randomize dog x, ball x and ball velocity at reset (train)",
    )
    parser.add_argument("--model", help="model path (eval); omit to pick from a list")
    args = parser.parse_args()

    opts = dict(tag=args.tag, seed=args.seed, random_start=args.random_start)
    if args.mode == "train_render":
        train(render_mode="human", **opts)
    elif args.mode == "train_no_render":
        train(render_mode=None, **opts)
    elif args.mode == "eval":
        evaluate(args.model or choose_model())


if __name__ == "__main__":
    main()
