from __future__ import annotations

import argparse
import os
from collections import deque
from typing import Literal

import numpy as np

from tank_ai.src.agents.cnn_dqn_agent import CNNDQNAgent
from tank_ai.src.agents.random_agent import RandomAgent
from tank_ai.src.env.tank_env import TankEnv
from tank_ai.src.rewards.reward_functions import get_available_reward_modes
from tank_ai.src.utils.config import CNNDQNConfig, TankGameConfig
from tank_ai.src.utils.plotting import plot_training_summary, save_training_log_csv
from tank_ai.src.utils.timer import TrainingTimer


OpponentType = Literal["random", "model", "self_play"]


def build_agent(rows: int, cols: int, dqn_cfg: CNNDQNConfig) -> CNNDQNAgent:
    """Create a CNN-DQN agent using the current grid size."""
    return CNNDQNAgent(
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


def choose_opponent_action(
    opponent_type: str,
    opponent_random: RandomAgent,
    opponent_model: CNNDQNAgent | None,
    training_agent: CNNDQNAgent,
    enemy_state: np.ndarray,
) -> int:
    """Choose player 2 action based on selected opponent mode."""
    if opponent_type == "random":
        return opponent_random.get_action(enemy_state)

    if opponent_type == "model":
        if opponent_model is None:
            raise ValueError("opponent_type='model' requires opponent_model_path.")
        return opponent_model.get_action(enemy_state, training=False)

    if opponent_type == "self_play":
        return training_agent.get_action(enemy_state, training=False)

    raise ValueError(f"Unknown opponent type: {opponent_type}")


def train_cnn_dqn(
    episodes: int,
    render_every: int,
    save_every: int,
    reward_mode: str,
    checkpoint_path: str,
    resume_path: str | None = None,
    speed: int | None = None,
    map_path: str | None = None,
    csv_path: str | None = None,
    plot_path: str | None = None,
    opponent_type: str = "random",
    opponent_model_path: str | None = None,
    train_both_sides: bool = True,
) -> None:
    """
    Train player 1 CNN-DQN against a selected opponent.

    Opponent modes:
        random:
            Player 2 uses random actions.
        model:
            Player 2 loads a fixed checkpoint and does not learn.
        self_play:
            Player 2 uses the current training agent. If train_both_sides=True,
            player 2 transitions are also inserted into the same replay buffer.
    """
    if opponent_type not in {"random", "model", "self_play"}:
        raise ValueError("opponent_type must be one of: random, model, self_play")

    if opponent_type == "model" and not opponent_model_path:
        raise ValueError("--opponent-model is required when --opponent model is selected.")

    game_cfg = TankGameConfig(reward_mode=reward_mode, map_path=map_path)
    if speed is not None:
        game_cfg.speed = speed

    dqn_cfg = CNNDQNConfig()
    probe_env = TankEnv(
        width=game_cfg.width,
        height=game_cfg.height,
        block_size=game_cfg.block_size,
        render_mode=False,
        reward_mode=reward_mode,
        map_path=map_path,
    )
    rows, cols = probe_env.rows, probe_env.cols
    probe_env.close()

    agent = build_agent(rows=rows, cols=cols, dqn_cfg=dqn_cfg)
    opponent_random = RandomAgent(action_size=dqn_cfg.output_size)
    opponent_model: CNNDQNAgent | None = None

    if opponent_type == "model":
        opponent_model = build_agent(rows=rows, cols=cols, dqn_cfg=dqn_cfg)
        opponent_model.load(opponent_model_path)  # type: ignore[arg-type]
        opponent_model.epsilon = 0.0
        print(f"Loaded fixed opponent model: {opponent_model_path}")

    if resume_path:
        agent.load(resume_path)
        print(f"Loaded training checkpoint: {resume_path}")

    scores_window = deque(maxlen=100)
    rows_log: list[dict] = []
    scores: list[float] = []
    avg_scores: list[float] = []
    losses: list[float] = []
    record_score = -10**9
    timer = TrainingTimer()

    print("CNN-DQN tank training started")
    print(f"Device: {agent.device}")
    print(f"Reward mode: {reward_mode}")
    print(f"Opponent: {opponent_type}")
    print(f"Opponent model: {opponent_model_path or '-'}")
    print(f"Train both sides: {train_both_sides if opponent_type == 'self_play' else False}")
    print(f"Map: {map_path or 'random generated map'}")
    print(f"Grid: rows={rows}, cols={cols}")
    print(f"Episodes: {episodes}")
    print("-" * 110)

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
            map_path=map_path,
        )
        env.reset()
        state = env.get_state(player_id=1)
        done = False
        total_reward = 0.0
        ep_losses = []
        timer.start_episode()
        last_info = {}

        while not done:
            action = agent.get_action(state, training=True)
            enemy_state = env.get_state(player_id=2)
            enemy_action = choose_opponent_action(
                opponent_type=opponent_type,
                opponent_random=opponent_random,
                opponent_model=opponent_model,
                training_agent=agent,
                enemy_state=enemy_state,
            )

            reward, reward2, done, info = env.step(action, enemy_action)
            last_info = info
            next_state = env.get_state(player_id=1)
            next_enemy_state = env.get_state(player_id=2)

            agent.remember(state, action, reward, next_state, done)

            if opponent_type == "self_play" and train_both_sides:
                agent.remember(enemy_state, enemy_action, reward2, next_enemy_state, done)

            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)
            total_reward += reward
            state = next_state

        agent.end_episode()
        winner = last_info.get("winner")
        score = env.tanks[1].health - env.tanks[2].health
        if winner == 1:
            score += 10
        elif winner == 2:
            score -= 10

        events = last_info.get("events", {})
        p1_events = events.get(1, {})

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
            "p1_health": last_info.get("p1_health"),
            "p2_health": last_info.get("p2_health"),
            "steps": last_info.get("step_count"),
            "hit_enemy": p1_events.get("hit_enemy", False),
            "got_hit": p1_events.get("got_hit", False),
            "wall_bump": p1_events.get("wall_bump", False),
            "fired": p1_events.get("fired", False),
            "timeout": p1_events.get("timeout", False),
            "record": record_score,
            "avg100": avg_score,
            "loss": avg_loss,
            "epsilon": agent.epsilon,
            "reward_mode": reward_mode,
            "opponent_type": opponent_type,
            "opponent_model": opponent_model_path or "",
            "train_both_sides": train_both_sides if opponent_type == "self_play" else False,
            "map_path": map_path or "random",
            "episode_time": timing.elapsed_seconds,
            "total_time": timing.total_elapsed_seconds,
            "training_steps": agent.training_steps,
        }
        rows_log.append(row)

        print(
            f"Ep {ep:4d} | score {score:5.1f} | reward {total_reward:8.2f} | "
            f"opponent {opponent_type:9s} | winner {str(winner):>4s} | "
            f"HP {last_info.get('p1_health')}-{last_info.get('p2_health')} | "
            f"steps {last_info.get('step_count'):4d} | avg100 {avg_score:6.2f} | "
            f"loss {avg_loss:8.5f} | eps {agent.epsilon:6.3f} | "
            f"hit {p1_events.get('hit_enemy', False)} | got_hit {p1_events.get('got_hit', False)} | "
            f"wall {p1_events.get('wall_bump', False)} | time {timing.elapsed_seconds:6.2f}s"
        )

        if save_every > 0 and ep % save_every == 0:
            agent.save(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    if env is not None:
        env.close()

    agent.save(checkpoint_path)
    run_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    csv_path = csv_path or f"tank_ai/runs/{run_name}_training.csv"
    plot_path = plot_path or f"tank_ai/runs/{run_name}_training.png"

    save_training_log_csv(csv_path, rows_log)
    plot_training_summary(scores, avg_scores, losses, plot_path, f"Tank CNN-DQN ({reward_mode}, {opponent_type})")

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
    parser.add_argument("--map", type=str, default=None, help="Optional JSON map path.")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path.")
    parser.add_argument("--plot", type=str, default=None, help="Optional PNG plot output path.")
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "model", "self_play"],
        help="Opponent used during training: random, model, or self_play.",
    )
    parser.add_argument(
        "--opponent-model",
        type=str,
        default=None,
        help="Fixed opponent checkpoint path. Required when --opponent model.",
    )
    parser.add_argument(
        "--no-train-both-sides",
        action="store_true",
        help="For self_play, only store player 1 transitions instead of both player perspectives.",
    )
    args = parser.parse_args()
    train_cnn_dqn(
        episodes=args.episodes,
        render_every=args.render_every,
        save_every=args.save_every,
        reward_mode=args.reward,
        checkpoint_path=args.checkpoint,
        resume_path=args.resume,
        speed=args.speed,
        map_path=args.map,
        csv_path=args.csv,
        plot_path=args.plot,
        opponent_type=args.opponent,
        opponent_model_path=args.opponent_model,
        train_both_sides=not args.no_train_both_sides,
    )


if __name__ == "__main__":
    main()
