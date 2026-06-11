from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict

from tank_ai.src.rewards.reward_functions import get_available_reward_modes


EPISODE_LINE_RE = re.compile(
    r"Ep\s+(?P<episode>\d+)\s+\|\s+score\s+(?P<score>-?\d+(?:\.\d+)?)\s+\|\s+"
    r"reward\s+(?P<total_reward>-?\d+(?:\.\d+)?)\s+\|\s+winner\s+(?P<winner>\S+)\s+\|\s+"
    r"HP\s+(?P<p1_hp>[^-\s]+)-(?P<p2_hp>[^\s]+)\s+\|\s+steps\s+(?P<steps>\d+)\s+\|\s+"
    r"avg100\s+(?P<avg100>-?\d+(?:\.\d+)?)\s+\|\s+loss\s+(?P<loss>-?\d+(?:\.\d+)?)\s+\|\s+"
    r"eps\s+(?P<epsilon>\d+(?:\.\d+)?)\s+\|.*time\s+(?P<episode_time>\d+(?:\.\d+)?)s"
)


class TrainingUI(tk.Tk):
    """
    Tkinter training launcher for CNN-DQN tank experiments.

    Features:
        - Custom training parameters
        - Reward selection
        - Custom map selection
        - CSV / plot export paths
        - Live per-run progress table
        - Raw terminal output per run
        - Parallel training slots: 1, 2, or 4 simultaneous runs
    """

    def __init__(self) -> None:
        super().__init__()
        self.title("Tank AI CNN-DQN Training UI")
        self.geometry("1180x820")
        self.processes: Dict[int, subprocess.Popen] = {}
        self.output_widgets: Dict[int, tk.Text] = {}
        self._build()

    def _build(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        self.episodes = tk.StringVar(value="500")
        self.render_every = tk.StringVar(value="0")
        self.save_every = tk.StringVar(value="50")
        self.speed = tk.StringVar(value="30")
        self.reward = tk.StringVar(value="balanced")
        self.parallel_runs = tk.StringVar(value="1")
        self.checkpoint = tk.StringVar(value="tank_ai/checkpoints/cnn_balanced_latest.pth")
        self.resume = tk.StringVar(value="")
        self.map_path = tk.StringVar(value="")
        self.csv_path = tk.StringVar(value="")
        self.plot_path = tk.StringVar(value="")

        settings = ttk.LabelFrame(root, text="Training settings", padding=10)
        settings.pack(fill="x")

        def add_row(row: int, label: str, widget: tk.Widget, browse: tk.Widget | None = None) -> None:
            ttk.Label(settings, text=label).grid(row=row, column=0, sticky="w", pady=4, padx=(0, 8))
            widget.grid(row=row, column=1, sticky="ew", pady=4)
            if browse is not None:
                browse.grid(row=row, column=2, sticky="w", padx=4)

        row = 0
        add_row(row, "Episodes", ttk.Entry(settings, textvariable=self.episodes, width=18)); row += 1
        add_row(row, "Render every", ttk.Entry(settings, textvariable=self.render_every, width=18)); row += 1
        add_row(row, "Save every", ttk.Entry(settings, textvariable=self.save_every, width=18)); row += 1
        add_row(row, "Render speed", ttk.Entry(settings, textvariable=self.speed, width=18)); row += 1
        add_row(row, "Reward mode", ttk.Combobox(settings, textvariable=self.reward, values=get_available_reward_modes(), state="readonly", width=18)); row += 1
        add_row(row, "Parallel training windows", ttk.Combobox(settings, textvariable=self.parallel_runs, values=["1", "2", "4"], state="readonly", width=18)); row += 1

        add_row(
            row,
            "Map file",
            ttk.Entry(settings, textvariable=self.map_path, width=80),
            ttk.Button(settings, text="Browse", command=self._browse_map),
        ); row += 1
        add_row(
            row,
            "Checkpoint output",
            ttk.Entry(settings, textvariable=self.checkpoint, width=80),
            ttk.Button(settings, text="Browse", command=self._browse_checkpoint),
        ); row += 1
        add_row(
            row,
            "Resume checkpoint",
            ttk.Entry(settings, textvariable=self.resume, width=80),
            ttk.Button(settings, text="Browse", command=self._browse_resume),
        ); row += 1
        add_row(
            row,
            "CSV output",
            ttk.Entry(settings, textvariable=self.csv_path, width=80),
            ttk.Button(settings, text="Save as", command=self._browse_csv_output),
        ); row += 1
        add_row(
            row,
            "Plot output",
            ttk.Entry(settings, textvariable=self.plot_path, width=80),
            ttk.Button(settings, text="Save as", command=self._browse_plot_output),
        ); row += 1

        settings.columnconfigure(1, weight=1)

        help_text = (
            "Parallel training windows launches 1, 2, or 4 independent training processes. "
            "If Render every is greater than 0, each process may open its own pygame render window. "
            "For fast training, keep Render every = 0."
        )
        ttk.Label(root, text=help_text, foreground="#555555").pack(fill="x", pady=(6, 0))

        btns = ttk.Frame(root)
        btns.pack(fill="x", pady=8)
        ttk.Button(btns, text="Start training", command=self.start_training).pack(side="left", padx=4)
        ttk.Button(btns, text="Stop all", command=self.stop_training).pack(side="left", padx=4)
        ttk.Button(btns, text="Export existing CSV", command=self.export_existing_csv).pack(side="left", padx=4)
        ttk.Button(btns, text="Export existing plot", command=self.export_existing_plot).pack(side="left", padx=4)

        progress_frame = ttk.LabelFrame(root, text="Live training progress", padding=8)
        progress_frame.pack(fill="x", pady=(0, 8))

        columns = (
            "run", "status", "episode", "reward_mode", "score", "total_reward",
            "winner", "hp", "steps", "avg100", "loss", "epsilon", "episode_time"
        )
        self.progress = ttk.Treeview(progress_frame, columns=columns, show="headings", height=6)
        headings = {
            "run": "Run",
            "status": "Status",
            "episode": "Episode",
            "reward_mode": "Reward",
            "score": "Score",
            "total_reward": "Total reward",
            "winner": "Winner",
            "hp": "HP",
            "steps": "Steps",
            "avg100": "Avg100",
            "loss": "Loss",
            "epsilon": "Epsilon",
            "episode_time": "Time(s)",
        }
        widths = {
            "run": 55, "status": 95, "episode": 80, "reward_mode": 100, "score": 80,
            "total_reward": 100, "winner": 80, "hp": 80, "steps": 75, "avg100": 80,
            "loss": 85, "epsilon": 80, "episode_time": 80,
        }
        for col in columns:
            self.progress.heading(col, text=headings[col])
            self.progress.column(col, width=widths[col], anchor="center")
        self.progress.pack(fill="x")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

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
        if any(process.poll() is None for process in self.processes.values()):
            messagebox.showwarning("Training running", "At least one training process is already running.")
            return

        try:
            run_count = int(self.parallel_runs.get())
        except ValueError:
            run_count = 1

        if run_count not in [1, 2, 4]:
            messagebox.showerror("Invalid parallel count", "Parallel training windows must be 1, 2, or 4.")
            return

        self._clear_outputs()

        for run_id in range(1, run_count + 1):
            cmd = self._build_command(run_id, run_count)
            self._add_run_tab(run_id, cmd)
            self._set_progress_status(run_id, "Starting")

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            self.processes[run_id] = process
            threading.Thread(target=self._stream_output, args=(run_id, process), daemon=True).start()

    def _build_command(self, run_id: int, run_count: int) -> list[str]:
        checkpoint = self._path_for_run(self.checkpoint.get(), run_id, run_count, ".pth")
        csv_path = self._path_for_run(self.csv_path.get(), run_id, run_count, ".csv") if self.csv_path.get().strip() else ""
        plot_path = self._path_for_run(self.plot_path.get(), run_id, run_count, ".png") if self.plot_path.get().strip() else ""

        cmd = [
            sys.executable, "-u", "-m", "tank_ai.src.train_cnn_dqn",
            "--episodes", self.episodes.get(),
            "--render-every", self.render_every.get(),
            "--save-every", self.save_every.get(),
            "--reward", self.reward.get(),
            "--checkpoint", checkpoint,
            "--speed", self.speed.get(),
        ]
        if self.resume.get().strip():
            cmd.extend(["--resume", self.resume.get().strip()])
        if self.map_path.get().strip():
            cmd.extend(["--map", self.map_path.get().strip()])
        if csv_path:
            cmd.extend(["--csv", csv_path])
        if plot_path:
            cmd.extend(["--plot", plot_path])
        return cmd

    @staticmethod
    def _path_for_run(path: str, run_id: int, run_count: int, default_ext: str) -> str:
        path = path.strip()
        if not path:
            return path
        if run_count == 1:
            return path
        root, ext = os.path.splitext(path)
        if not ext:
            ext = default_ext
        return f"{root}_run{run_id}{ext}"

    def _clear_outputs(self) -> None:
        for item in self.progress.get_children():
            self.progress.delete(item)
        for tab_id in self.notebook.tabs():
            self.notebook.forget(tab_id)
        self.processes.clear()
        self.output_widgets.clear()

    def _add_run_tab(self, run_id: int, cmd: list[str]) -> None:
        frame = ttk.Frame(self.notebook)
        text = tk.Text(frame, wrap="word")
        text.pack(fill="both", expand=True)
        self.notebook.add(frame, text=f"Run {run_id}")
        self.output_widgets[run_id] = text
        text.insert("end", "Running: " + " ".join(cmd) + "\n\n")
        self.progress.insert(
            "", "end", iid=str(run_id),
            values=(run_id, "Starting", "-", self.reward.get(), "-", "-", "-", "-", "-", "-", "-", "-", "-"),
        )

    def _stream_output(self, run_id: int, process: subprocess.Popen) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            self.after(0, self._append_output, run_id, line)
            parsed = EPISODE_LINE_RE.search(line)
            if parsed:
                self.after(0, self._update_progress_from_match, run_id, parsed)
        code = process.wait()
        status = "Finished" if code == 0 else f"Exited {code}"
        self.after(0, self._append_output, run_id, f"\nProcess {status}.\n")
        self.after(0, self._set_progress_status, run_id, status)

    def _append_output(self, run_id: int, line: str) -> None:
        text = self.output_widgets.get(run_id)
        if text is None:
            return
        text.insert("end", line)
        text.see("end")

    def _update_progress_from_match(self, run_id: int, match: re.Match) -> None:
        data = match.groupdict()
        values = (
            run_id,
            "Running",
            data["episode"],
            self.reward.get(),
            data["score"],
            data["total_reward"],
            data["winner"],
            f"{data['p1_hp']}-{data['p2_hp']}",
            data["steps"],
            data["avg100"],
            data["loss"],
            data["epsilon"],
            data["episode_time"],
        )
        self.progress.item(str(run_id), values=values)

    def _set_progress_status(self, run_id: int, status: str) -> None:
        iid = str(run_id)
        if iid not in self.progress.get_children():
            return
        values = list(self.progress.item(iid, "values"))
        if len(values) >= 2:
            values[1] = status
        self.progress.item(iid, values=values)

    def stop_training(self) -> None:
        stopped = False
        for process in self.processes.values():
            if process.poll() is None:
                process.terminate()
                stopped = True
        if stopped:
            for run_id in self.processes:
                self._set_progress_status(run_id, "Stopping")
            messagebox.showinfo("Stopping", "Training processes are being stopped.")

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
