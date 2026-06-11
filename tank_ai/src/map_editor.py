from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from tank_ai.src.utils.map_utils import load_map, save_map


class MapEditorUI(tk.Tk):
    """Simple grid-based tank map editor."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Tank AI Map Editor")
        self.geometry("900x650")
        self.cols = tk.IntVar(value=20)
        self.rows = tk.IntVar(value=15)
        self.cell_size = 28
        self.tool = tk.StringVar(value="wall")
        self.map_name = tk.StringVar(value="custom_map")
        self.walls: set[tuple[int, int]] = set()
        self.p1_spawn = (1, 7)
        self.p2_spawn = (18, 7)
        self._build()
        self._redraw()

    def _build(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text="Name").pack(side="left")
        ttk.Entry(top, textvariable=self.map_name, width=18).pack(side="left", padx=4)
        ttk.Label(top, text="Cols").pack(side="left")
        ttk.Entry(top, textvariable=self.cols, width=5).pack(side="left", padx=4)
        ttk.Label(top, text="Rows").pack(side="left")
        ttk.Entry(top, textvariable=self.rows, width=5).pack(side="left", padx=4)
        ttk.Button(top, text="Resize/Clear", command=self.resize_clear).pack(side="left", padx=4)

        tools = ttk.Frame(self, padding=8)
        tools.pack(fill="x")
        for value, label in [("wall", "Wall"), ("erase", "Erase"), ("p1", "P1 spawn"), ("p2", "P2 spawn")]:
            ttk.Radiobutton(tools, text=label, value=value, variable=self.tool).pack(side="left", padx=6)
        ttk.Button(tools, text="Load", command=self.load).pack(side="right", padx=4)
        ttk.Button(tools, text="Save", command=self.save).pack(side="right", padx=4)

        self.canvas = tk.Canvas(self, bg="#18181c")
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_click)

        ttk.Label(
            self,
            text="Click cells to edit. Gray=wall, Blue=P1 spawn, Red=P2 spawn. Saved maps are JSON files.",
            padding=8,
        ).pack(fill="x")

    def resize_clear(self) -> None:
        self.walls.clear()
        self.p1_spawn = (1, self.rows.get() // 2)
        self.p2_spawn = (self.cols.get() - 2, self.rows.get() // 2)
        self._redraw()

    def on_click(self, event) -> None:
        x = event.x // self.cell_size
        y = event.y // self.cell_size
        if not (0 <= x < self.cols.get() and 0 <= y < self.rows.get()):
            return
        cell = (x, y)
        tool = self.tool.get()
        if tool == "wall":
            if cell not in [self.p1_spawn, self.p2_spawn]:
                self.walls.add(cell)
        elif tool == "erase":
            self.walls.discard(cell)
        elif tool == "p1":
            self.walls.discard(cell)
            self.p1_spawn = cell
        elif tool == "p2":
            self.walls.discard(cell)
            self.p2_spawn = cell
        self._redraw()

    def _redraw(self) -> None:
        self.canvas.delete("all")
        cols = self.cols.get()
        rows = self.rows.get()
        self.canvas.config(scrollregion=(0, 0, cols * self.cell_size, rows * self.cell_size))
        for y in range(rows):
            for x in range(cols):
                x0 = x * self.cell_size
                y0 = y * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                color = "#242428"
                if (x, y) in self.walls:
                    color = "#70707a"
                if (x, y) == self.p1_spawn:
                    color = "#469fff"
                if (x, y) == self.p2_spawn:
                    color = "#eb5a50"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#353540")

    def save(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile=f"{self.map_name.get()}.json",
            filetypes=[("JSON map", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        save_map(path, self.map_name.get(), self.cols.get(), self.rows.get(), self.walls, self.p1_spawn, self.p2_spawn)
        messagebox.showinfo("Saved", f"Map saved to:\n{path}")

    def load(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON map", "*.json"), ("All files", "*.*")])
        if not path:
            return
        config = load_map(path)
        self.map_name.set(config.name)
        self.cols.set(config.cols)
        self.rows.set(config.rows)
        self.walls = set(config.walls)
        self.p1_spawn = config.p1_spawn or (1, config.rows // 2)
        self.p2_spawn = config.p2_spawn or (config.cols - 2, config.rows // 2)
        self._redraw()


def main() -> None:
    app = MapEditorUI()
    app.mainloop()


if __name__ == "__main__":
    main()
