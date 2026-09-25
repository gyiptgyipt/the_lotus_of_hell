"""
Train the intercept policy with PPO.

Prereqs running before you launch this:
  - PX4 SITL + Gazebo world (interceptor + target vehicles spawned)
  - your target_sim node, publishing /uav_intercept/target_state
  - uav_control's offboard_control_node (handles arming/offboard handshake)

Usage:
  python3 train_ppo.py --timesteps 500000 --logdir runs/exp1
"""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from intercept_env import InterceptEnv


def make_env(logdir: str):
    env = InterceptEnv()
    return Monitor(env, filename=f"{logdir}/monitor")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--logdir", type=str, default="runs/exp1")
    parser.add_argument("--checkpoint-every", type=int, default=20_000)
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()

    env = make_env(args.logdir)

    if args.resume_from:
        model = PPO.load(args.resume_from, env=env)
        print(f"Resumed from {args.resume_from}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=args.logdir,
            n_steps=2048,
            batch_size=256,
            gamma=0.995,          # long-ish horizon, intercept reward is sparse-ish
            gae_lambda=0.95,
            learning_rate=3e-4,
            ent_coef=0.005,       # a little exploration bonus; intercept easily collapses to "hover near"
            clip_range=0.2,
            policy_kwargs=dict(net_arch=[128, 128]),
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_every, save_path=args.logdir, name_prefix="ppo_intercept"
    )

    try:
        model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb, progress_bar=True)
    finally:
        model.save(f"{args.logdir}/ppo_intercept_final")
        env.close()


if __name__ == "__main__":
    main()
