from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class MapConfig:
    name: str
    cols: int
    rows: int
    walls: set[tuple[int, int]]
    p1_spawn: tuple[int, int] | None = None
    p2_spawn: tuple[int, int] | None = None


def load_map(path: str) -> MapConfig:
    """
    Load a custom tank map from JSON.

    Expected JSON format:
        {
          "name": "arena_name",
          "cols": 20,
          "rows": 15,
          "p1_spawn": [1, 7],
          "p2_spawn": [18, 7],
          "walls": [[10, 3], [10, 4]]
        }
    """
    with open(path, "r", encoding="utf-8") as file:
        data: dict[str, Any] = json.load(file)

    name = str(data.get("name", os.path.splitext(os.path.basename(path))[0]))
    cols = int(data["cols"])
    rows = int(data["rows"])

    walls = {(int(x), int(y)) for x, y in data.get("walls", [])}

    p1_spawn = data.get("p1_spawn")
    p2_spawn = data.get("p2_spawn")

    return MapConfig(
        name=name,
        cols=cols,
        rows=rows,
        walls=walls,
        p1_spawn=tuple(p1_spawn) if p1_spawn else None,
        p2_spawn=tuple(p2_spawn) if p2_spawn else None,
    )


def save_map(
    path: str,
    name: str,
    cols: int,
    rows: int,
    walls: set[tuple[int, int]],
    p1_spawn: tuple[int, int],
    p2_spawn: tuple[int, int],
) -> None:
    """Save a custom map to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = {
        "name": name,
        "cols": cols,
        "rows": rows,
        "p1_spawn": list(p1_spawn),
        "p2_spawn": list(p2_spawn),
        "walls": [list(item) for item in sorted(walls)],
    }

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def validate_map(map_config: MapConfig) -> None:
    """Validate map bounds and spawn positions."""
    if map_config.cols <= 4 or map_config.rows <= 4:
        raise ValueError("Map must be at least 5 x 5 cells.")

    for x, y in map_config.walls:
        if x < 0 or x >= map_config.cols or y < 0 or y >= map_config.rows:
            raise ValueError(f"Wall out of bounds: {(x, y)}")

    if map_config.p1_spawn is not None:
        _validate_spawn("p1_spawn", map_config.p1_spawn, map_config)

    if map_config.p2_spawn is not None:
        _validate_spawn("p2_spawn", map_config.p2_spawn, map_config)


def _validate_spawn(name: str, spawn: tuple[int, int], map_config: MapConfig) -> None:
    x, y = spawn
    if x < 0 or x >= map_config.cols or y < 0 or y >= map_config.rows:
        raise ValueError(f"{name} out of bounds: {spawn}")
    if spawn in map_config.walls:
        raise ValueError(f"{name} cannot be inside a wall: {spawn}")
