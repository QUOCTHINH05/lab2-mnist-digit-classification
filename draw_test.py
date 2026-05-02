"""draw_test.py — Interactive digit drawing for MNIST classifier inference.

Usage:
    python draw_test.py --model logistic
    python draw_test.py --model nn
    python draw_test.py --model both        # side-by-side comparison

Requires the models to be trained first:
    python main.py configure
    python main.py train --model logistic
    python main.py train --model nn
"""

from __future__ import annotations

import argparse
import os
import sys
import tkinter as tk
from tkinter import ttk

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    sys.exit("Error: TensorFlow is not installed. Run: pip install tensorflow")

try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("Error: Pillow is not installed. Run: pip install Pillow")

from classifier import LogisticRegressionClassifier, NeuralNetworkClassifier, MNISTClassifier

MODEL_DIR   = "models"
MODEL_MAP   = {
    "logistic": LogisticRegressionClassifier,
    "nn":       NeuralNetworkClassifier,
}
CANVAS_SIZE = 280   # drawing canvas pixels
GRID_SIZE   = 28    # MNIST resolution


def load_classifier(model_type: str) -> MNISTClassifier:
    path = os.path.join(MODEL_DIR, f"{model_type}_model.keras")
    if not os.path.exists(path):
        sys.exit(
            f"Error: model not found at '{path}'.\n"
            f"  Run:  python main.py train --model {model_type}"
        )
    clf = MODEL_MAP[model_type]()
    clf.load(path)
    print(f"Loaded {model_type} model from {path}")
    return clf


class DrawWindow:
    """Main application window."""

    BRUSH_RADIUS = 14  # default brush radius in canvas pixels

    def __init__(self, root: tk.Tk, classifiers: dict[str, MNISTClassifier]):
        self.root        = root
        self.classifiers = classifiers
        self.root.title("MNIST Digit Classifier")
        self.root.resizable(False, False)

        # Shadow stroke list: each entry is (x0, y0, x1, y1, radius)
        # We replay these onto a PIL canvas to get clean pixel data —
        # no Ghostscript, no screen capture, no DPI issues.
        self._strokes: list[tuple[float, float, float, float]] = []

        self._build_ui()
        self._reset_canvas()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = dict(padx=10, pady=6)

        # Left: drawing canvas
        frame_left = tk.Frame(self.root, bg="#111")
        frame_left.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)

        self.canvas = tk.Canvas(
            frame_left,
            width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="black", cursor="crosshair",
            highlightthickness=1, highlightbackground="#444",
        )
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>",  self._on_press)
        self.canvas.bind("<B1-Motion>",      self._on_drag)

        # 28×28 preview below the canvas
        preview_frame = tk.Frame(frame_left, bg="#111")
        preview_frame.pack(pady=(8, 0))
        tk.Label(preview_frame, text="28×28 preview  (what the model sees)",
                 bg="#111", fg="#555", font=("Courier", 9)).pack()
        self.preview = tk.Canvas(
            preview_frame,
            width=GRID_SIZE * 4, height=GRID_SIZE * 4,
            bg="black", highlightthickness=1, highlightbackground="#333",
        )
        self.preview.pack()

        # Right: controls + results
        frame_right = tk.Frame(self.root, bg="#1a1a1a")
        frame_right.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)

        # Brush size
        brush_row = tk.Frame(frame_right, bg="#1a1a1a")
        brush_row.pack(fill="x", **pad)
        tk.Label(brush_row, text="Brush size", bg="#1a1a1a", fg="#888",
                 font=("Courier", 10)).pack(side="left")
        self.brush_var = tk.IntVar(value=self.BRUSH_RADIUS)
        tk.Scale(brush_row, from_=6, to=30, orient="horizontal",
                 variable=self.brush_var, bg="#1a1a1a", fg="#ccc",
                 troughcolor="#333", highlightthickness=0,
                 length=120, showvalue=True).pack(side="right")

        # Buttons
        btn_row = tk.Frame(frame_right, bg="#1a1a1a")
        btn_row.pack(fill="x", **pad)
        tk.Button(btn_row, text="Classify", command=self._classify,
                  bg="#f0f0f0", fg="#111", activebackground="#ddd",
                  font=("Courier", 11, "bold"), relief="flat",
                  padx=16, pady=6).pack(side="left", padx=(0, 8))
        tk.Button(btn_row, text="Clear", command=self._reset_canvas,
                  bg="#333", fg="#ccc", activebackground="#444",
                  font=("Courier", 11), relief="flat",
                  padx=16, pady=6).pack(side="left")

        # Status bar
        self.status_var = tk.StringVar(value="Draw a digit and click Classify.")
        tk.Label(frame_right, textvariable=self.status_var,
                 bg="#1a1a1a", fg="#888", font=("Courier", 10),
                 wraplength=280, justify="left").pack(fill="x", **pad)

        ttk.Separator(frame_right, orient="horizontal").pack(fill="x", padx=10, pady=4)

        # Result panels — one per classifier
        self.result_frames: dict[str, dict] = {}
        for label in self.classifiers:
            lf = tk.LabelFrame(
                frame_right, text=f"  {label.upper()}  ",
                bg="#1a1a1a", fg="#888",
                font=("Courier", 9), bd=1, relief="groove",
                padx=10, pady=8,
            )
            lf.pack(fill="x", padx=10, pady=6)

            digit_var = tk.StringVar(value="—")
            conf_var  = tk.StringVar(value="")
            tk.Label(lf, textvariable=digit_var, bg="#1a1a1a", fg="#f0f0f0",
                     font=("Courier", 52, "bold")).pack(side="left", padx=(0, 16))

            right_col = tk.Frame(lf, bg="#1a1a1a")
            right_col.pack(side="left", fill="both", expand=True)
            tk.Label(right_col, textvariable=conf_var, bg="#1a1a1a", fg="#aaa",
                     font=("Courier", 10)).pack(anchor="w")

            bar_frame = tk.Frame(right_col, bg="#1a1a1a")
            bar_frame.pack(fill="x", pady=(4, 0))
            bars: list[tuple] = []
            for d in range(10):
                row_f = tk.Frame(bar_frame, bg="#1a1a1a")
                row_f.pack(fill="x", pady=1)
                tk.Label(row_f, text=str(d), bg="#1a1a1a", fg="#555",
                         font=("Courier", 9), width=2).pack(side="left")
                track = tk.Frame(row_f, bg="#2a2a2a", height=5)
                track.pack(side="left", fill="x", expand=True, padx=(2, 4))
                fill_bar = tk.Frame(track, bg="#666", height=5)
                fill_bar.place(x=0, y=0, relheight=1.0, width=0)
                val_lbl = tk.Label(row_f, text="0.0%", bg="#1a1a1a", fg="#444",
                                   font=("Courier", 8), width=5)
                val_lbl.pack(side="left")
                bars.append((track, fill_bar, val_lbl))

            self.result_frames[label] = {
                "digit": digit_var,
                "conf":  conf_var,
                "bars":  bars,
            }

    # ── Canvas interaction ─────────────────────────────────────────────────────

    def _on_press(self, event):
        self._last_x = event.x
        self._last_y = event.y
        self._paint(event.x, event.y)

    def _on_drag(self, event):
        # Interpolate circles between last and current position for smooth strokes
        x0, y0 = self._last_x, self._last_y
        x1, y1 = event.x, event.y
        dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        steps = max(1, int(dist / 4))
        for i in range(steps + 1):
            t = i / steps
            xi = x0 + (x1 - x0) * t
            yi = y0 + (y1 - y0) * t
            self._paint(xi, yi)
        self._last_x, self._last_y = x1, y1

    def _paint(self, x: float, y: float):
        r = self.brush_var.get()
        self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                fill="white", outline="white")
        self._strokes.append((x - r, y - r, x + r, y + r))

    # ── Pixel extraction ───────────────────────────────────────────────────────

    def _get_pixel_grid(self) -> np.ndarray:
        """Replay strokes onto a PIL image and preprocess exactly like MNIST.

        MNIST digits are:
          - Anti-aliased white on black
          - Centered in a 20×20 bounding box inside a 28×28 image
          - Normalized to [0, 1]

        We replicate this so the model sees data that matches its training
        distribution, which is the main reason for wrong predictions when
        using raw screen captures.
        """
        # 1. Render strokes onto a clean 280×280 image
        img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        draw = ImageDraw.Draw(img)
        for (x0, y0, x1, y1) in self._strokes:
            draw.ellipse([x0, y0, x1, y1], fill=255)

        # 2. Crop tight bounding box around the digit
        bbox = img.getbbox()
        if bbox is None:
            # Empty canvas
            return np.zeros((1, 784), dtype=np.float32)

        digit = img.crop(bbox)

        # 3. Scale so the longest side = 20px (MNIST uses 20×20 bounding box)
        digit.thumbnail((20, 20), Image.LANCZOS)

        # 4. Paste centered into a 28×28 black canvas
        final = Image.new("L", (28, 28), color=0)
        offset_x = (28 - digit.width)  // 2
        offset_y = (28 - digit.height) // 2
        final.paste(digit, (offset_x, offset_y))

        arr = np.array(final, dtype=np.float32) / 255.0
        return arr.reshape(1, 784)

    # ── Preview ────────────────────────────────────────────────────────────────

    def _update_preview(self, grid: np.ndarray):
        arr = grid.reshape(GRID_SIZE, GRID_SIZE)
        self.preview.delete("all")
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                v = int(arr[r, c] * 255)
                if v > 8:
                    color = f"#{v:02x}{v:02x}{v:02x}"
                    px, py = c * 4, r * 4
                    self.preview.create_rectangle(
                        px, py, px + 4, py + 4,
                        fill=color, outline="",
                    )

    # ── Reset ──────────────────────────────────────────────────────────────────

    def _reset_canvas(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(
            0, 0, CANVAS_SIZE, CANVAS_SIZE, fill="black", outline=""
        )
        self._strokes.clear()
        self.preview.delete("all")
        for rf in self.result_frames.values():
            rf["digit"].set("—")
            rf["conf"].set("")
            for _track, fill_bar, lbl in rf["bars"]:
                fill_bar.place(width=0)
                lbl.config(text="0.0%")
        self.status_var.set("Canvas cleared. Draw a digit.")

    # ── Classify ───────────────────────────────────────────────────────────────

    def _classify(self):
        self.status_var.set("Running inference…")
        self.root.update_idletasks()

        x = self._get_pixel_grid()
        if x.max() < 0.05:
            self.status_var.set("Canvas is empty — draw a digit first!")
            return

        self._update_preview(x)

        summaries = []
        for label, clf in self.classifiers.items():
            probs = clf.model.predict(x, verbose=0)[0]
            pred  = int(np.argmax(probs))
            conf  = float(probs[pred]) * 100

            rf = self.result_frames[label]
            rf["digit"].set(str(pred))
            rf["conf"].set(f"{conf:.1f}% confidence")

            track_width = 180
            for d, (_track, fill_bar, lbl) in enumerate(rf["bars"]):
                p = float(probs[d])
                fill_bar.config(bg="#f0f0f0" if d == pred else "#444")
                fill_bar.place(width=int(p * track_width))
                lbl.config(text=f"{p*100:.1f}%")

            summaries.append(f"{label.upper()}: {pred} ({conf:.1f}%)")

        self.status_var.set("  |  ".join(summaries))


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive MNIST digit drawing + inference"
    )
    parser.add_argument(
        "--model",
        choices=["logistic", "nn", "both"],
        default="both",
        help="Which model(s) to use (default: both)",
    )
    args = parser.parse_args()

    keys = ["logistic", "nn"] if args.model == "both" else [args.model]
    classifiers = {k: load_classifier(k) for k in keys}

    root = tk.Tk()
    DrawWindow(root, classifiers)
    root.mainloop()


if __name__ == "__main__":
    main()