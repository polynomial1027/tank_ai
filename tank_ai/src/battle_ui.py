from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, ttk


CONTROLLERS = ["human_wasd", "human_arrows", "random", "model"]


class BattleUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Tank AI Battle/Test UI")
        self.geometry("700x360")
        self.process: subprocess.Popen | None = None
        self._build()

    def _build(self) -> None:
        frame = ttk.Frame(self, padding=12)
        frame.pack(fill="both", expand=True)
        self.left = tk.StringVar(value="human_wasd")
        self.right = tk.StringVar(value="random")
        self.left_model = tk.StringVar(value="")
        self.right_model = tk.StringVar(value="")
        self.reward = tk.StringVar(value="balanced")
        self.speed = tk.StringVar(value="20")
        self.map_path = tk.StringVar(value="")
        row = 0
        ttk.Label(frame, text="Left player").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(frame, textvariable=self.left, values=CONTROLLERS, state="readonly").grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Label(frame, text="Left model file").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.left_model, width=60).grid(row=row, column=1, sticky="w")
        ttk.Button(frame, text="Browse", command=lambda: self._browse(self.left_model)).grid(row=row, column=2, padx=4)
        row += 1
        ttk.Label(frame, text="Right player").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(frame, textvariable=self.right, values=CONTROLLERS, state="readonly").grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Label(frame, text="Right model file").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.right_model, width=60).grid(row=row, column=1, sticky="w")
        ttk.Button(frame, text="Browse", command=lambda: self._browse(self.right_model)).grid(row=row, column=2, padx=4)
        row += 1
        ttk.Label(frame, text="Reward mode").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.reward, width=20).grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Label(frame, text="Speed").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.speed, width=20).grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Label(frame, text="Map file").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.map_path, width=60).grid(row=row, column=1, sticky="w")
        ttk.Button(frame, text="Browse", command=lambda: self._browse_map()).grid(row=row, column=2, padx=4)
        row += 1
        ttk.Button(frame, text="Start battle", command=self.start_battle).grid(row=row, column=0, pady=12)
        ttk.Button(frame, text="Stop", command=self.stop_battle).grid(row=row, column=1, sticky="w", pady=12)
        row += 1
        ttk.Label(frame, text="Controls: WASD player fires with Space/Left Shift. Arrow player fires with Enter/Right Shift. ESC quits.").grid(row=row, column=0, columnspan=3, sticky="w")

    def _browse(self, var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(filetypes=[("PyTorch checkpoint", "*.pth"), ("All files", "*.*")])
        if path:
            var.set(path)

    def _browse_map(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Tank map JSON", "*.json"), ("All files", "*.*")])
        if path:
            self.map_path.set(path)

    def start_battle(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        cmd = [
            sys.executable, "-m", "tank_ai.src.play",
            "--left", self.left.get(),
            "--right", self.right.get(),
            "--reward", self.reward.get(),
            "--speed", self.speed.get(),
        ]
        if self.left_model.get().strip():
            cmd.extend(["--left-model", self.left_model.get().strip()])
        if self.right_model.get().strip():
            cmd.extend(["--right-model", self.right_model.get().strip()])
        if self.map_path.get().strip():
            cmd.extend(["--map", self.map_path.get().strip()])
        self.process = subprocess.Popen(cmd)

    def stop_battle(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()


def main() -> None:
    app = BattleUI()
    app.mainloop()
