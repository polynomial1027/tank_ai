from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import pygame

from tank_ai.src.rewards.reward_functions import calculate_reward
from tank_ai.src.utils.map_utils import MapConfig, load_map, validate_map


@dataclass(frozen=True)
class Point:
    x: int
    y: int


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


DIRECTION_VECTORS = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
}


@dataclass
class Tank:
    position: Point
    direction: Direction
    health: int


@dataclass
class Bullet:
    position: Point
    direction: Direction
    owner: int
    life: int


class TankEnv:
    """
    Two-player tank battle environment.

    Action space for each tank:
        0 = no-op
        1 = move up
        2 = move down
        3 = move left
        4 = move right
        5 = fire

    CNN state shape:
        (6, rows, cols)

    Channels from selected player's perspective:
        0 self tank
        1 enemy tank
        2 walls
        3 self bullets
        4 enemy bullets
        5 danger map from enemy bullets
    """

    ACTION_NOOP = 0
    ACTION_UP = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    ACTION_RIGHT = 4
    ACTION_FIRE = 5

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        block_size: int = 32,
        speed: int = 30,
        render_mode: bool = True,
        max_steps: int = 700,
        wall_count: int = 28,
        bullet_max_life: int = 30,
        tank_max_health: int = 3,
        reward_mode: str = "balanced",
        seed: int | None = None,
        map_path: str | None = None,
    ) -> None:
        self.block_size = block_size
        self.map_path = map_path
        self.map_config: MapConfig | None = None

        if map_path:
            self.map_config = load_map(map_path)
            validate_map(self.map_config)
            width = self.map_config.cols * block_size
            height = self.map_config.rows * block_size

        if width % block_size != 0 or height % block_size != 0:
            raise ValueError("width and height must be divisible by block_size")

        self.width = width
        self.height = height
        self.speed = speed
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.wall_count = wall_count
        self.bullet_max_life = bullet_max_life
        self.tank_max_health = tank_max_health
        self.reward_mode = reward_mode
        self.rng = random.Random(seed)
        self.cols = width // block_size
        self.rows = height // block_size
        self.display = None
        self.clock = None

        if render_mode:
            pygame.init()
            self.display = pygame.display.set_mode((width, height))
            title = f"Tank AI | reward={reward_mode}"
            if map_path:
                title += f" | map={self.map_config.name if self.map_config else map_path}"
            pygame.display.set_caption(title)
            self.clock = pygame.time.Clock()

        self.walls: set[Point] = set()
        self.tanks: Dict[int, Tank] = {}
        self.bullets: List[Bullet] = []
        self.step_count = 0
        self.winner: int | None = None
        self.reset()

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.winner = None
        self.bullets = []

        p1_spawn = Point(1, self.rows // 2)
        p2_spawn = Point(self.cols - 2, self.rows // 2)

        if self.map_config is not None:
            if self.map_config.p1_spawn:
                p1_spawn = Point(*self.map_config.p1_spawn)
            if self.map_config.p2_spawn:
                p2_spawn = Point(*self.map_config.p2_spawn)

        self.tanks = {
            1: Tank(p1_spawn, Direction.RIGHT, self.tank_max_health),
            2: Tank(p2_spawn, Direction.LEFT, self.tank_max_health),
        }

        if self.map_config is not None:
            self.walls = {Point(x, y) for x, y in self.map_config.walls}
        else:
            self._generate_walls()

        if self.render_mode:
            self.render()
        return self.get_state(player_id=1)

    def step(self, action_p1: int, action_p2: int) -> Tuple[float, float, bool, dict]:
        self.step_count += 1
        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

        events = {
            1: self._empty_events(),
            2: self._empty_events(),
        }

        self._apply_action(1, action_p1, events[1])
        self._apply_action(2, action_p2, events[2])
        self._update_bullets(events)

        done = False
        if self.tanks[1].health <= 0 and self.tanks[2].health <= 0:
            done = True
            self.winner = 0
            events[1]["lost"] = True
            events[2]["lost"] = True
        elif self.tanks[1].health <= 0:
            done = True
            self.winner = 2
            events[1]["lost"] = True
            events[2]["won"] = True
        elif self.tanks[2].health <= 0:
            done = True
            self.winner = 1
            events[1]["won"] = True
            events[2]["lost"] = True
        elif self.step_count >= self.max_steps:
            done = True
            self.winner = None
            events[1]["timeout"] = True
            events[2]["timeout"] = True

        reward1 = calculate_reward(self.reward_mode, events[1])
        reward2 = calculate_reward(self.reward_mode, events[2])

        if self.render_mode:
            self.render()

        info = {
            "winner": self.winner,
            "p1_health": self.tanks[1].health,
            "p2_health": self.tanks[2].health,
            "step_count": self.step_count,
            "events": events,
            "map_path": self.map_path,
        }
        return reward1, reward2, done, info

    def get_state(self, player_id: int = 1) -> np.ndarray:
        enemy_id = 2 if player_id == 1 else 1
        grid = np.zeros((6, self.rows, self.cols), dtype=np.float32)
        self_pos = self.tanks[player_id].position
        enemy_pos = self.tanks[enemy_id].position
        grid[0, self_pos.y, self_pos.x] = 1.0
        grid[1, enemy_pos.y, enemy_pos.x] = 1.0
        for wall in self.walls:
            grid[2, wall.y, wall.x] = 1.0
        for bullet in self.bullets:
            if 0 <= bullet.position.x < self.cols and 0 <= bullet.position.y < self.rows:
                if bullet.owner == player_id:
                    grid[3, bullet.position.y, bullet.position.x] = 1.0
                else:
                    grid[4, bullet.position.y, bullet.position.x] = 1.0
                    dx, dy = DIRECTION_VECTORS[bullet.direction]
                    nx, ny = bullet.position.x + dx, bullet.position.y + dy
                    if 0 <= nx < self.cols and 0 <= ny < self.rows:
                        grid[5, ny, nx] = 1.0
        return grid

    def render(self) -> None:
        if not self.render_mode or self.display is None or self.clock is None:
            return
        background = (24, 24, 26)
        grid_color = (40, 40, 44)
        wall_color = (95, 95, 105)
        p1_color = (70, 160, 255)
        p2_color = (235, 90, 80)
        bullet_p1 = (160, 210, 255)
        bullet_p2 = (255, 170, 150)
        text_color = (235, 235, 235)
        self.display.fill(background)
        for x in range(0, self.width, self.block_size):
            pygame.draw.line(self.display, grid_color, (x, 0), (x, self.height))
        for y in range(0, self.height, self.block_size):
            pygame.draw.line(self.display, grid_color, (0, y), (self.width, y))
        for wall in self.walls:
            self._draw_cell(wall, wall_color, radius=2)
        self._draw_tank(1, p1_color)
        self._draw_tank(2, p2_color)
        for bullet in self.bullets:
            color = bullet_p1 if bullet.owner == 1 else bullet_p2
            cx = bullet.position.x * self.block_size + self.block_size // 2
            cy = bullet.position.y * self.block_size + self.block_size // 2
            pygame.draw.circle(self.display, color, (cx, cy), max(4, self.block_size // 6))
        font = pygame.font.SysFont("arial", 20)
        map_label = self.map_config.name if self.map_config else "random"
        label = (
            f"P1 HP: {self.tanks[1].health} | P2 HP: {self.tanks[2].health} | "
            f"Steps: {self.step_count} | Reward: {self.reward_mode} | Map: {map_label}"
        )
        text = font.render(label, True, text_color)
        self.display.blit(text, (8, 8))
        pygame.display.flip()
        self.clock.tick(self.speed)

    def close(self) -> None:
        if self.render_mode:
            pygame.quit()

    def _empty_events(self) -> dict:
        return {
            "won": False,
            "lost": False,
            "hit_enemy": False,
            "got_hit": False,
            "fired": False,
            "moved": False,
            "wall_bump": False,
            "timeout": False,
        }

    def _generate_walls(self) -> None:
        self.walls = set()
        protected = {
            self.tanks[1].position,
            Point(2, self.rows // 2),
            Point(1, self.rows // 2 - 1),
            Point(1, self.rows // 2 + 1),
            self.tanks[2].position,
            Point(self.cols - 3, self.rows // 2),
            Point(self.cols - 2, self.rows // 2 - 1),
            Point(self.cols - 2, self.rows // 2 + 1),
        }
        attempts = 0
        while len(self.walls) < self.wall_count and attempts < self.wall_count * 20:
            attempts += 1
            p = Point(self.rng.randint(2, self.cols - 3), self.rng.randint(1, self.rows - 2))
            if p not in protected:
                self.walls.add(p)
        for y in range(max(1, self.rows // 2 - 2), min(self.rows - 1, self.rows // 2 + 3)):
            if y != self.rows // 2:
                self.walls.add(Point(self.cols // 2, y))

    def _apply_action(self, player_id: int, action: int, events: dict) -> None:
        tank = self.tanks[player_id]
        if action == self.ACTION_NOOP:
            return
        if action == self.ACTION_FIRE:
            events["fired"] = True
            self._fire(player_id)
            return
        direction = self._action_to_direction(action)
        tank.direction = direction
        dx, dy = DIRECTION_VECTORS[direction]
        next_pos = Point(tank.position.x + dx, tank.position.y + dy)
        if self._is_blocked(next_pos, ignore_tank=player_id):
            events["wall_bump"] = True
            return
        tank.position = next_pos
        events["moved"] = True

    def _action_to_direction(self, action: int) -> Direction:
        if action == self.ACTION_UP:
            return Direction.UP
        if action == self.ACTION_DOWN:
            return Direction.DOWN
        if action == self.ACTION_LEFT:
            return Direction.LEFT
        if action == self.ACTION_RIGHT:
            return Direction.RIGHT
        raise ValueError(f"Invalid move action: {action}")

    def _fire(self, player_id: int) -> None:
        tank = self.tanks[player_id]
        dx, dy = DIRECTION_VECTORS[tank.direction]
        start = Point(tank.position.x + dx, tank.position.y + dy)
        if self._out_of_bounds(start) or start in self.walls:
            return
        self.bullets.append(Bullet(start, tank.direction, player_id, self.bullet_max_life))

    def _update_bullets(self, events: Dict[int, dict]) -> None:
        new_bullets: List[Bullet] = []
        occupied_by_bullet = set()
        for bullet in self.bullets:
            dx, dy = DIRECTION_VECTORS[bullet.direction]
            next_pos = Point(bullet.position.x + dx, bullet.position.y + dy)
            bullet.life -= 1
            if bullet.life <= 0 or self._out_of_bounds(next_pos) or next_pos in self.walls:
                continue
            enemy_id = 2 if bullet.owner == 1 else 1
            if next_pos == self.tanks[enemy_id].position:
                self.tanks[enemy_id].health -= 1
                events[bullet.owner]["hit_enemy"] = True
                events[enemy_id]["got_hit"] = True
                continue
            if next_pos in occupied_by_bullet:
                continue
            occupied_by_bullet.add(next_pos)
            bullet.position = next_pos
            new_bullets.append(bullet)
        self.bullets = new_bullets

    def _is_blocked(self, point: Point, ignore_tank: int | None = None) -> bool:
        if self._out_of_bounds(point) or point in self.walls:
            return True
        for player_id, tank in self.tanks.items():
            if ignore_tank is not None and player_id == ignore_tank:
                continue
            if tank.position == point:
                return True
        return False

    def _out_of_bounds(self, point: Point) -> bool:
        return point.x < 0 or point.x >= self.cols or point.y < 0 or point.y >= self.rows

    def _draw_cell(self, point: Point, color: tuple[int, int, int], radius: int = 4) -> None:
        rect = pygame.Rect(point.x * self.block_size, point.y * self.block_size, self.block_size, self.block_size)
        pygame.draw.rect(self.display, color, rect, border_radius=radius)
        pygame.draw.rect(self.display, (15, 15, 17), rect, 1, border_radius=radius)

    def _draw_tank(self, player_id: int, color: tuple[int, int, int]) -> None:
        tank = self.tanks[player_id]
        x = tank.position.x * self.block_size
        y = tank.position.y * self.block_size
        margin = 4
        body = pygame.Rect(x + margin, y + margin, self.block_size - 2 * margin, self.block_size - 2 * margin)
        pygame.draw.rect(self.display, color, body, border_radius=5)
        dx, dy = DIRECTION_VECTORS[tank.direction]
        center = (x + self.block_size // 2, y + self.block_size // 2)
        barrel_end = (center[0] + dx * self.block_size // 2, center[1] + dy * self.block_size // 2)
        pygame.draw.line(self.display, (245, 245, 245), center, barrel_end, 4)
