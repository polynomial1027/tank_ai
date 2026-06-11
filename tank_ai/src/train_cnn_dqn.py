from __future__ import annotations

import argparse
import os
import sys
from collections import deque

import numpy as np

from tank_ai.src.agents.cnn_dqn_agent import CNNDQNAgent
from tank_ai.src.agents.random_agent import RandomAgent
from tank_ai.src.env.tank_env import TankEnv
from tank_ai.src.rewards.reward_functions import get_available_reward_modes
from tank_ai.src.utils.config import CNNDQNConfig, TankGameConfig
from tank_ai.src.utils.plotting import plot_training_summary, save_training_log_csv
from tank_ai.src.utils.timer import TrainingTimer


def train_cnn_dqn(
    episodes: int,
    render_every: int,
    save_every: int,
    reward_mode: str,
    checkpoint_path: str,
    resume_path: str | None = None,
    speed: int | None = None,
) -> None:
    game_cfg = TankGameConfig(reward_mode=reward_mode)
    if speed is not None:
        game_cfg.speed = speed
    dqn_cfg = CNNDQNConfig()
    probe_env = TankEnv(render_mode=False, reward_mode=reward_mode)
    rows, cols = probe_env.rows, probe_env.cols
    probe_env.close()

    agent = CNNDQNAgent(
        input_channels=dqn_cfg.input_channels,
        output_size=dqn_cfg.output_size,
        rows=rows,
        cols=cols,
        learning_rate=dqn_cfg.learning_rate,
        gamma=dqn_cfg.gamma,
        batch_size=dqn_cfg.batch_size,
        memory_size=dqn_cfg.memory_size,
        epsilon_start=dqn_cfg.epsilon_start,
        epsilon_end=dqn_cfg.epsilon_end,
        epsilon_decay=dqn_cfg.epsilon_decay,
        target_update_every=dqn_cfg.target_update_every,
    )
    opponent = RandomAgent(action_size=dqn_cfg.output_size)
    if resume_path:
        agent.load(resume_path)
        print(f"Loaded checkpoint: {resume_path}")

    scores_window = deque(maxlen=100)
    rows_log = []
    scores = []
    avg_scores = []
    losses = []
    record_score = -10**9
    timer = TrainingTimer()

    print("CNN-DQN tank training started")
    print(f"Device: {agent.device}")
    print(f"Reward mode: {reward_mode}")
    print(f"Episodes: {episodes}")
    print("-" * 80)

    env = None
    for ep in range(1, episodes + 1):
        should_render = render_every > 0 and ep % render_every == 0
        if env is not None:
            env.close()
        env = TankEnv(
            width=game_cfg.width,
            height=game_cfg.height,
            block_size=game_cfg.block_size,
            speed=game_cfg.speed,
            render_mode=should_render,
            max_steps=game_cfg.max_steps,
            wall_count=game_cfg.wall_count,
            bullet_max_life=game_cfg.bullet_max_life,
            tank_max_health=game_cfg.tank_max_health,
            reward_mode=reward_mode,
        )
        env.reset()
        state = env.get_state(player_id=1)
        done = False
        total_reward = 0.0
        ep_losses = []
        timer.start_episode()
        while not done:
            action = agent.get_action(state, training=True)
            enemy_state = env.get_state(player_id=2)
            enemy_action = opponent.get_action(enemy_state)
            reward, _, done, info = env.step(action, enemy_action)
            next_state = env.get_state(player_id=1)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)
            total_reward += reward
            state = next_state
        agent.end_episode()
        winner = info.get("winner")
        score = env.tanks[1].health - env.tanks[2].health
        if winner == 1:
            score += 10
        elif winner == 2:
            score -= 10
        scores_window.append(score)
        record_score = max(record_score, score)
        avg_score = float(np.mean(scores_window))
        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        timing = timer.end_episode(ep)
        scores.append(score)
        avg_scores.append(avg_score)
        losses.append(avg_loss)
        row = {
            "episode": ep,
            "score": score,
            "total_reward": total_reward,
            "winner": winner,
            "p1_health": info.get("p1_health"),
            "p2_health": info.get("p2_health"),
            "record": record_score,
            "avg100": avg_score,
            "loss": avg_loss,
            "epsilon": agent.epsilon,
            "reward_mode": reward_mode,
            "episode_time": timing.elapsed_seconds,
            "total_time": timing.total_elapsed_seconds,
        }
        rows_log.append(row)
        print(
            f"Ep {ep:4d} | score {score:4.1f} | winner {winner} | "
            f"avg100 {avg_score:6.2f} | loss {avg_loss:8.5f} | eps {agent.epsilon:6.3f} | "
            f"time {timing.elapsed_seconds:6.2f}s"
        )
        if save_every > 0 and ep % save_every == 0:
            agent.save(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    if env is not None:
        env.close()
    agent.save(checkpoint_path)
    run_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    csv_path = f"tank_ai/runs/{run_name}_training.csv"
    plot_path = f"tank_ai/runs/{run_name}_training.png"
    save_training_log_csv(csv_path, rows_log)
    plot_training_summary(scores, avg_scores, losses, plot_path, f"Tank CNN-DQN ({reward_mode})")
    print(f"Final checkpoint saved: {checkpoint_path}")
    print(f"Training CSV saved: {csv_path}")
    print(f"Training plot saved: {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN-DQN tank agent")
    parser.add_argument("--episodes", type=int, default=CNNDQNConfig.max_episodes)
    parser.add_argument("--render-every", type=int, default=CNNDQNConfig.render_every)
    parser.add_argument("--save-every", type=int, default=CNNDQNConfig.save_every)
    parser.add_argument("--reward", type=str, default="balanced", choices=get_available_reward_modes())
    parser.add_argument("--checkpoint", type=str, default="tank_ai/checkpoints/cnn_balanced_latest.pth")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--speed", type=int, default=None)
    args = parser.parse_args()
    train_cnn_dqn(
        episodes=args.episodes,
        render_every=args.render_every,
        save_every=args.save_every,
        reward_mode=args.reward,
        checkpoint_path=args.checkpoint,
        resume_path=args.resume,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
