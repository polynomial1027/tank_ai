from __future__ import annotations

import argparse

import pygame

from tank_ai.src.agents.cnn_dqn_agent import CNNDQNAgent
from tank_ai.src.agents.random_agent import RandomAgent
from tank_ai.src.env.tank_env import Direction, TankEnv
from tank_ai.src.utils.config import CNNDQNConfig, TankGameConfig


class HumanController:
    def __init__(self, scheme: str):
        self.scheme = scheme
        self.last_action = 0

    def update_from_events(self, events) -> None:
        for event in events:
            if event.type != pygame.KEYDOWN:
                continue
            if self.scheme == "wasd":
                if event.key == pygame.K_w:
                    self.last_action = 1
                elif event.key == pygame.K_s:
                    self.last_action = 2
                elif event.key == pygame.K_a:
                    self.last_action = 3
                elif event.key == pygame.K_d:
                    self.last_action = 4
                elif event.key in [pygame.K_SPACE, pygame.K_LSHIFT]:
                    self.last_action = 5
            else:
                if event.key == pygame.K_UP:
                    self.last_action = 1
                elif event.key == pygame.K_DOWN:
                    self.last_action = 2
                elif event.key == pygame.K_LEFT:
                    self.last_action = 3
                elif event.key == pygame.K_RIGHT:
                    self.last_action = 4
                elif event.key in [pygame.K_RETURN, pygame.K_RSHIFT]:
                    self.last_action = 5

    def get_action(self, state=None, training: bool = False) -> int:
        action = self.last_action
        self.last_action = 0
        return action


def make_controller(kind: str, model_path: str | None, rows: int, cols: int):
    if kind == "random":
        return RandomAgent(6)
    if kind == "human_wasd":
        return HumanController("wasd")
    if kind == "human_arrows":
        return HumanController("arrows")
    if kind == "model":
        if not model_path:
            raise ValueError("model controller requires model path")
        cfg = CNNDQNConfig()
        agent = CNNDQNAgent(input_channels=cfg.input_channels, output_size=cfg.output_size, rows=rows, cols=cols, epsilon_start=0.0, epsilon_end=0.0)
        agent.load(model_path)
        agent.epsilon = 0.0
        return agent
    raise ValueError(f"Unknown controller: {kind}")


def run_match(left: str, right: str, left_model: str | None, right_model: str | None, reward: str = "balanced", speed: int = 20) -> None:
    cfg = TankGameConfig(speed=speed, reward_mode=reward)
    env = TankEnv(width=cfg.width, height=cfg.height, block_size=cfg.block_size, speed=cfg.speed, render_mode=True, reward_mode=reward)
    p1 = make_controller(left, left_model, env.rows, env.cols)
    p2 = make_controller(right, right_model, env.rows, env.cols)
    print("Tank match started. ESC to quit.")
    done = False
    while not done:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                done = True
        if hasattr(p1, "update_from_events"):
            p1.update_from_events(events)
        if hasattr(p2, "update_from_events"):
            p2.update_from_events(events)
        a1 = p1.get_action(env.get_state(1), training=False)
        a2 = p2.get_action(env.get_state(2), training=False)
        _, _, ended, info = env.step(a1, a2)
        if ended:
            print(f"Match ended. Winner: {info.get('winner')}")
            pygame.time.wait(1200)
            env.reset()
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Play/test tank battle")
    parser.add_argument("--left", choices=["human_wasd", "human_arrows", "random", "model"], default="human_wasd")
    parser.add_argument("--right", choices=["human_wasd", "human_arrows", "random", "model"], default="random")
    parser.add_argument("--left-model", default=None)
    parser.add_argument("--right-model", default=None)
    parser.add_argument("--reward", default="balanced")
    parser.add_argument("--speed", type=int, default=20)
    args = parser.parse_args()
    run_match(args.left, args.right, args.left_model, args.right_model, args.reward, args.speed)


if __name__ == "__main__":
    main()
