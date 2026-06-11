from __future__ import annotations

import shutil
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
        self.geometry("880x700")
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
        self.map_path = tk.StringVar(value="")
        self.csv_path = tk.StringVar(value="")
        self.plot_path = tk.StringVar(value="")
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
        ttk.Label(frame, text="Map file").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.map_path, width=70).grid(row=row, column=1, sticky="w", pady=4)
        ttk.Button(frame, text="Browse", command=self._browse_map).grid(row=row, column=2, padx=4)
        row += 1
        ttk.Label(frame, text="Checkpoint output").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.checkpoint, width=70).grid(row=row, column=1, sticky="w", pady=4)
        ttk.Button(frame, text="Browse", command=self._browse_checkpoint).grid(row=row, column=2, padx=4)
        row += 1
        ttk.Label(frame, text="Resume checkpoint").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.resume, width=70).grid(row=row, column=1, sticky="w", pady=4)
        ttk.Button(frame, text="Browse", command=self._browse_resume).grid(row=row, column=2, padx=4)
        row += 1
        ttk.Label(frame, text="CSV output").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.csv_path, width=70).grid(row=row, column=1, sticky="w", pady=4)
        ttk.Button(frame, text="Save as", command=self._browse_csv_output).grid(row=row, column=2, padx=4)
        row += 1
        ttk.Label(frame, text="Plot output").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.plot_path, width=70).grid(row=row, column=1, sticky="w", pady=4)
        ttk.Button(frame, text="Save as", command=self._browse_plot_output).grid(row=row, column=2, padx=4)
        row += 1
        btns = ttk.Frame(frame)
        btns.grid(row=row, column=0, columnspan=3, sticky="w", pady=8)
        ttk.Button(btns, text="Start training", command=self.start_training).pack(side="left", padx=4)
        ttk.Button(btns, text="Stop", command=self.stop_training).pack(side="left", padx=4)
        ttk.Button(btns, text="Export existing CSV", command=self.export_existing_csv).pack(side="left", padx=4)
        ttk.Button(btns, text="Export existing plot", command=self.export_existing_plot).pack(side="left", padx=4)
        row += 1
        self.output = tk.Text(frame, height=26, wrap="word")
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

    def _browse_map(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Tank map JSON", "*.json"), ("All files", "*.*")])
        if path:
            self.map_path.set(path)

    def _browse_csv_output(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if path:
            self.csv_path.set(path)

    def _browse_plot_output(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("All files", "*.*")])
        if path:
            self.plot_path.set(path)

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
        if self.map_path.get().strip():
            cmd.extend(["--map", self.map_path.get().strip()])
        if self.csv_path.get().strip():
            cmd.extend(["--csv", self.csv_path.get().strip()])
        if self.plot_path.get().strip():
            cmd.extend(["--plot", self.plot_path.get().strip()])
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

    def export_existing_csv(self) -> None:
        source = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not source:
            return
        target = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not target:
            return
        shutil.copyfile(source, target)
        messagebox.showinfo("Exported", f"CSV exported to:\n{target}")

    def export_existing_plot(self) -> None:
        source = filedialog.askopenfilename(filetypes=[("PNG", "*.png"), ("All files", "*.*")])
        if not source:
            return
        target = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("All files", "*.*")])
        if not target:
            return
        shutil.copyfile(source, target)
        messagebox.showinfo("Exported", f"Plot exported to:\n{target}")


def main() -> None:
    app = TrainingUI()
    app.mainloop()
