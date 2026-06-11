from __future__ import annotations

import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from tank_ai.src.rewards.reward_functions import get_available_reward_modes


class TrainingUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Tank AI CNN-DQN Training UI")
        self.geometry("760x560")
        self.process: subprocess.Popen | None = None
        self._build()

    def _build(self) -> None:
        frame = ttk.Frame(self, padding=12)
        frame.pack(fill="both", expand=True)
        self.episodes = tk.StringVar(value="500")
        self.render_every = tk.StringVar(value="0")
        self.save_every = tk.StringVar(value="50")
        self.speed = tk.StringVar(value="30")
        self.reward = tk.StringVar(value="balanced")
        self.checkpoint = tk.StringVar(value="tank_ai/checkpoints/cnn_balanced_latest.pth")
        self.resume = tk.StringVar(value="")
        row = 0
        for label, var in [
            ("Episodes", self.episodes),
            ("Render every", self.render_every),
            ("Save every", self.save_every),
            ("Render speed", self.speed),
        ]:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(frame, textvariable=var, width=20).grid(row=row, column=1, sticky="w", pady=4)
            row += 1
        ttk.Label(frame, text="Reward mode").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(frame, textvariable=self.reward, values=get_available_reward_modes(), state="readonly", width=18).grid(row=row, column=1, sticky="w", pady=4)
        row += 1
        ttk.Label(frame, text="Checkpoint output").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.checkpoint, width=60).grid(row=row, column=1, sticky="w", pady=4)
        ttk.Button(frame, text="Browse", command=self._browse_checkpoint).grid(row=row, column=2, padx=4)
        row += 1
        ttk.Label(frame, text="Resume checkpoint").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.resume, width=60).grid(row=row, column=1, sticky="w", pady=4)
        ttk.Button(frame, text="Browse", command=self._browse_resume).grid(row=row, column=2, padx=4)
        row += 1
        btns = ttk.Frame(frame)
        btns.grid(row=row, column=0, columnspan=3, sticky="w", pady=8)
        ttk.Button(btns, text="Start training", command=self.start_training).pack(side="left", padx=4)
        ttk.Button(btns, text="Stop", command=self.stop_training).pack(side="left", padx=4)
        row += 1
        self.output = tk.Text(frame, height=22, wrap="word")
        self.output.grid(row=row, column=0, columnspan=3, sticky="nsew")
        frame.rowconfigure(row, weight=1)
        frame.columnconfigure(1, weight=1)

    def _browse_checkpoint(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("PyTorch checkpoint", "*.pth"), ("All files", "*.*")])
        if path:
            self.checkpoint.set(path)

    def _browse_resume(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PyTorch checkpoint", "*.pth"), ("All files", "*.*")])
        if path:
            self.resume.set(path)

    def start_training(self) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("Training running", "A training process is already running.")
            return
        cmd = [
            sys.executable, "-m", "tank_ai.src.train_cnn_dqn",
            "--episodes", self.episodes.get(),
            "--render-every", self.render_every.get(),
            "--save-every", self.save_every.get(),
            "--reward", self.reward.get(),
            "--checkpoint", self.checkpoint.get(),
            "--speed", self.speed.get(),
        ]
        if self.resume.get().strip():
            cmd.extend(["--resume", self.resume.get().strip()])
        self.output.insert("end", "Running: " + " ".join(cmd) + "\n")
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self._stream_output, daemon=True).start()

    def _stream_output(self) -> None:
        assert self.process is not None
        for line in self.process.stdout:
            self.output.insert("end", line)
            self.output.see("end")
        self.output.insert("end", "\nProcess finished.\n")
        self.output.see("end")

    def stop_training(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.output.insert("end", "\nTraining stopped.\n")


def main() -> None:
    app = TrainingUI()
    app.mainloop()
