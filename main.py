"""
Linear Regression Platform v2.0
Black & Orange Dark Theme — Production Ready
Compatible with PyInstaller .exe packaging
"""

import sys
import os
import multiprocessing

# Required for PyInstaller .exe support
if __name__ == "__main__":
    multiprocessing.freeze_support()

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================
#  THEME — BLACK & ORANGE
# ============================================================
BG         = "#080808"
BG_CARD    = "#0f0f0f"
BG_PANEL   = "#141414"
BG_INPUT   = "#1a1a1a"
BG_HOVER   = "#1e1e1e"

ORANGE     = "#f97316"
ORANGE_LT  = "#fb923c"
ORANGE_DK  = "#c2410c"
ORANGE_DIM = "#7c2d12"

TEXT       = "#f5f5f5"
TEXT_SEC   = "#888888"
TEXT_MUT   = "#444444"
SUCCESS    = "#22c55e"
DANGER     = "#ef4444"
BORDER     = "#222222"
BORDER_LT  = "#2a2a2a"

FF_TITLE   = ("Consolas", 16, "bold")
FF_SUB     = ("Consolas", 10, "bold")
FF_BODY    = ("Segoe UI",  9)
FF_MONO    = ("Consolas", 10)
FF_BTN     = ("Consolas", 10, "bold")
FF_SMALL   = ("Segoe UI",  8)


# ============================================================
#  HELPER — HOVERABLE BUTTON
# ============================================================
class HoverButton(tk.Button):
    def __init__(self, parent, hover_bg=ORANGE, hover_fg="white", **kwargs):
        self._default_bg = kwargs.get("bg", BG_CARD)
        self._default_fg = kwargs.get("fg", TEXT)
        self._hover_bg   = hover_bg
        self._hover_fg   = hover_fg
        super().__init__(parent, **kwargs, relief="flat", bd=0,
                         cursor="hand2",
                         activebackground=hover_bg,
                         activeforeground=hover_fg)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _on_enter(self, _):
        self.config(bg=self._hover_bg, fg=self._hover_fg)

    def _on_leave(self, _):
        self.config(bg=self._default_bg, fg=self._default_fg)


# ============================================================
#  MAIN APPLICATION
# ============================================================
class LinearRegressionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Linear Regression Platform  v2.0")
        self.root.geometry("1160x740")
        self.root.minsize(960, 640)
        self.root.configure(bg=BG)

        # ── State ──────────────────────────────────────────
        self.x_data:  list  = []
        self.y_data:  list  = []
        self.model          = None
        self.df             = pd.DataFrame()
        self.test_size_var  = tk.DoubleVar(value=0.2)
        self.x_col_name     = "—"
        self.y_col_name     = "—"

        self._apply_ttk_style()
        self._build_ui()
        self._setup_empty_plot()

    # ────────────────────────────────────────────────────────
    #  TTK STYLE (dark theme for combobox, treeview, scrollbar)
    # ────────────────────────────────────────────────────────
    def _apply_ttk_style(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TScrollbar",
                        background=BORDER_LT, troughcolor=BG_PANEL,
                        borderwidth=0, arrowsize=12, arrowcolor=TEXT_SEC)
        style.map("TScrollbar", background=[("active", ORANGE_DK)])

        style.configure("LR.Treeview",
                        background=BG_PANEL, foreground=TEXT,
                        fieldbackground=BG_PANEL, rowheight=22,
                        borderwidth=0, font=FF_MONO)
        style.configure("LR.Treeview.Heading",
                        background=BORDER, foreground=ORANGE,
                        font=("Consolas", 9, "bold"), relief="flat")
        style.map("LR.Treeview",
                  background=[("selected", ORANGE_DIM)],
                  foreground=[("selected", TEXT)])

        style.configure("TCombobox",
                        fieldbackground=BG_INPUT, background=BORDER,
                        foreground=TEXT, selectbackground=ORANGE_DK,
                        insertcolor=ORANGE, padding=5)
        style.map("TCombobox",
                  fieldbackground=[("readonly", BG_INPUT)],
                  foreground=[("readonly", TEXT)],
                  selectbackground=[("readonly", ORANGE_DK)])

    # ────────────────────────────────────────────────────────
    #  UI CONSTRUCTION
    # ────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_header()

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=(8, 0))

        # Left sidebar (300 px fixed)
        sidebar = tk.Frame(body, bg=BG_CARD, width=296)
        sidebar.pack(side="left", fill="y", padx=(0, 10))
        sidebar.pack_propagate(False)

        # Right workspace
        workspace = tk.Frame(body, bg=BG)
        workspace.pack(side="left", fill="both", expand=True)

        self._build_sidebar(sidebar)
        self._build_workspace(workspace)
        self._build_statusbar()

    # ── Header ───────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self.root, bg=BG_CARD, height=52)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        accent = tk.Frame(self.root, bg=ORANGE, height=2)
        accent.pack(fill="x")

        icon_lbl = tk.Label(hdr, text="◈", fg=ORANGE, bg=BG_CARD,
                            font=("Consolas", 20))
        icon_lbl.pack(side="left", padx=(16, 6), pady=8)

        tk.Label(hdr, text="LINEAR REGRESSION PLATFORM",
                 fg=TEXT, bg=BG_CARD, font=FF_TITLE).pack(side="left", pady=8)

        self.lbl_header_status = tk.Label(
            hdr, text="◉  NO DATA", fg=DANGER, bg=BG_CARD, font=FF_BTN)
        self.lbl_header_status.pack(side="right", padx=20)

    # ── Sidebar ───────────────────────────────────────────────
    def _build_sidebar(self, parent):
        # Scrollable canvas so every section is always reachable
        canvas = tk.Canvas(parent, bg=BG_CARD, highlightthickness=0, bd=0)
        vsb    = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner  = tk.Frame(canvas, bg=BG_CARD)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win_id, width=e.width))
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        parent = inner   # all widgets packed into scrollable inner frame

        # ── Section: Data Input ──────────────────
        self._section(parent, "DATA INPUT")

        g = tk.Frame(parent, bg=BG_CARD)
        g.pack(fill="x", padx=10, pady=4)

        self._btn(g, "▲  IMPORT CSV",    ORANGE,    BG_CARD, self.import_csv).pack(fill="x", pady=2)
        self._btn(g, "✎  MANUAL INPUT",  BG_HOVER,  ORANGE,  self.manual_input).pack(fill="x", pady=2)
        self._btn(g, "⊘  CLEAR DATA",    BG_CARD,   DANGER,  self.clear_data).pack(fill="x", pady=2)

        # ── Section: Data Summary ─────────────────
        self._section(parent, "DATA SUMMARY")
        summary = tk.Frame(parent, bg=BG_PANEL)
        summary.pack(fill="x", padx=10, pady=4)

        self.lbl_pts     = self._row(summary, "Points",  "0")
        self.lbl_x_range = self._row(summary, "X Range", "—")
        self.lbl_y_range = self._row(summary, "Y Range", "—")
        self.lbl_x_col   = self._row(summary, "X Column","—")
        self.lbl_y_col   = self._row(summary, "Y Column","—")

        # ── Section: Training ─────────────────────
        self._section(parent, "TRAINING CONFIG")
        train_fr = tk.Frame(parent, bg=BG_CARD)
        train_fr.pack(fill="x", padx=10, pady=4)

        tk.Label(train_fr, text="Test Split Size",
                 fg=TEXT_SEC, bg=BG_CARD, font=FF_BODY).pack(anchor="w", pady=(4, 0))

        slider_row = tk.Frame(train_fr, bg=BG_CARD)
        slider_row.pack(fill="x")

        self.slider = tk.Scale(
            slider_row, from_=0.1, to=0.5, resolution=0.05,
            orient="horizontal", variable=self.test_size_var,
            bg=BG_CARD, fg=ORANGE, highlightthickness=0,
            troughcolor=BORDER_LT, activebackground=ORANGE_LT,
            sliderrelief="flat", showvalue=True, length=220,
            font=FF_SMALL
        )
        self.slider.pack(fill="x")

        self._btn(train_fr, "▶  TRAIN MODEL", ORANGE, BG_CARD, self.train_model).pack(fill="x", pady=6)

        # ── Section: Model Metrics ────────────────
        self._section(parent, "MODEL METRICS")
        met = tk.Frame(parent, bg=BG_PANEL)
        met.pack(fill="x", padx=10, pady=4)

        self.lbl_slope     = self._row(met, "Slope (a)",    "—")
        self.lbl_intercept = self._row(met, "Intercept (b)","—")
        self.lbl_r2        = self._row(met, "R² Score",     "—")
        self.lbl_mse       = self._row(met, "MSE",          "—")
        self.lbl_mae       = self._row(met, "MAE",          "—")

        # ── Section: Prediction ───────────────────
        self._section(parent, "PREDICTION")
        pred_fr = tk.Frame(parent, bg=BG_CARD)
        pred_fr.pack(fill="x", padx=10, pady=4)

        self._btn(pred_fr, "◈  OPEN PREDICTION WINDOW",
                  ORANGE, BG_CARD, self.open_prediction_window).pack(fill="x", pady=4)

        tk.Label(pred_fr,
                 text="Run predictions, view history\nand export results",
                 fg=TEXT_MUT, bg=BG_CARD, font=FF_SMALL,
                 justify="center").pack(pady=(0, 6))

    # ── Workspace (right side) ───────────────────────────────
    def _build_workspace(self, parent):
        # Graph container
        graph_wrap = tk.Frame(parent, bg=BG_CARD)
        graph_wrap.pack(fill="both", expand=True)

        # Graph toolbar bar
        g_toolbar = tk.Frame(graph_wrap, bg=BG_PANEL, height=32)
        g_toolbar.pack(fill="x")
        g_toolbar.pack_propagate(False)

        tk.Label(g_toolbar, text="REGRESSION PLOT",
                 fg=ORANGE, bg=BG_PANEL, font=FF_SUB).pack(side="left", padx=12, pady=6)

        self._small_btn(g_toolbar, "↺ Refresh", self.update_graph).pack(side="right", padx=4, pady=4)
        self._small_btn(g_toolbar, "⬇ Save Plot", self.save_plot).pack(side="right", padx=4, pady=4)

        # Matplotlib figure
        plt.style.use("dark_background")
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.fig.patch.set_facecolor(BG_CARD)
        self.ax.set_facecolor(BG_PANEL)

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_wrap)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Matplotlib nav toolbar (hidden but functional for zoom/pan)
        nav_frame = tk.Frame(graph_wrap, bg=BG_CARD)
        nav_frame.pack(fill="x")
        nav = NavigationToolbar2Tk(self.canvas, nav_frame)
        nav.config(bg=BG_CARD)
        nav._message_label.config(bg=BG_CARD, fg=TEXT_SEC)
        for child in nav.winfo_children():
            try:
                child.config(bg=BG_CARD, fg=TEXT_SEC,
                             activebackground=ORANGE_DK,
                             highlightbackground=BG_CARD)
            except tk.TclError:
                pass
        nav.update()

        # Data Preview table
        table_wrap = tk.Frame(parent, bg=BG_CARD, height=145)
        table_wrap.pack(fill="x", pady=(8, 0))
        table_wrap.pack_propagate(False)

        tk.Label(table_wrap, text="DATA PREVIEW",
                 fg=ORANGE, bg=BG_CARD, font=FF_SUB).pack(anchor="w", padx=12, pady=(6, 2))

        tree_frame = tk.Frame(table_wrap, bg=BG_CARD)
        tree_frame.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        self.tree = ttk.Treeview(
            tree_frame, columns=("idx", "x", "y"), show="headings",
            style="LR.Treeview", height=4
        )
        self.tree.heading("idx", text="#")
        self.tree.heading("x",   text="X Value")
        self.tree.heading("y",   text="Y Value")
        self.tree.column("idx", width=40,  minwidth=40,  stretch=False)
        self.tree.column("x",   width=180, minwidth=100)
        self.tree.column("y",   width=180, minwidth=100)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical",   command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side="right",  fill="y")
        hsb.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)

    # ── Status Bar ───────────────────────────────────────────
    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=BG_PANEL, height=24)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self.status_var = tk.StringVar(value="Ready — Import data or enter manually.")
        tk.Label(bar, textvariable=self.status_var,
                 fg=TEXT_SEC, bg=BG_PANEL, font=FF_SMALL, anchor="w").pack(side="left", padx=10)

        tk.Label(bar, text="Linear Regression Platform  v2.0  |  sklearn + matplotlib",
                 fg=TEXT_MUT, bg=BG_PANEL, font=FF_SMALL).pack(side="right", padx=10)

    # ────────────────────────────────────────────────────────
    #  UI WIDGET HELPERS
    # ────────────────────────────────────────────────────────
    def _section(self, parent, label: str):
        f = tk.Frame(parent, bg=BG_CARD)
        f.pack(fill="x", padx=10, pady=(12, 2))
        tk.Label(f, text=label, fg=ORANGE, bg=BG_CARD,
                 font=("Consolas", 8, "bold")).pack(side="left")
        tk.Frame(f, bg=BORDER_LT, height=1).pack(side="left", fill="x",
                                                   expand=True, padx=(8, 0), pady=6)

    def _btn(self, parent, text, bg, sidebar_bg, cmd,
             hover_bg=None, fg=None):
        if fg is None:
            fg = TEXT if bg not in (ORANGE,) else "white"
        if hover_bg is None:
            hover_bg = ORANGE_LT if bg == ORANGE else ORANGE
        btn = HoverButton(parent, text=text, bg=bg, fg=fg,
                          hover_bg=hover_bg, hover_fg="white",
                          font=FF_BTN, command=cmd,
                          padx=10, pady=7)
        return btn

    def _small_btn(self, parent, text, cmd):
        return HoverButton(parent, text=text, bg=BORDER, fg=TEXT_SEC,
                           hover_bg=ORANGE_DK, hover_fg="white",
                           font=FF_SMALL, command=cmd, padx=8, pady=3)

    def _row(self, parent, label: str, value: str):
        f = tk.Frame(parent, bg=BG_PANEL)
        f.pack(fill="x", padx=8, pady=2)
        tk.Label(f, text=label, fg=TEXT_SEC, bg=BG_PANEL,
                 font=FF_BODY, width=13, anchor="w").pack(side="left")
        val = tk.Label(f, text=value, fg=TEXT, bg=BG_PANEL,
                       font=FF_MONO, anchor="e")
        val.pack(side="right")
        return val

    # ────────────────────────────────────────────────────────
    #  STATE HELPERS
    # ────────────────────────────────────────────────────────
    def _status(self, msg: str):
        self.status_var.set(msg)

    def _header_status(self, text: str, color: str):
        self.lbl_header_status.config(text=text, fg=color)

    def _reset_metrics(self):
        for lbl in (self.lbl_slope, self.lbl_intercept,
                    self.lbl_r2, self.lbl_mse, self.lbl_mae):
            lbl.config(text="—", fg=TEXT)
        self.lbl_pred_result.config(text="Y  =  —")

    def _update_summary(self):
        n = len(self.x_data)
        self.lbl_pts.config(text=str(n))
        if n > 0:
            x = np.array(self.x_data)
            y = np.array(self.y_data)
            self.lbl_x_range.config(text=f"{x.min():.3g} → {x.max():.3g}")
            self.lbl_y_range.config(text=f"{y.min():.3g} → {y.max():.3g}")
        else:
            self.lbl_x_range.config(text="—")
            self.lbl_y_range.config(text="—")
        self.lbl_x_col.config(text=self.x_col_name)
        self.lbl_y_col.config(text=self.y_col_name)

    def _refresh_table(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for i, (x, y) in enumerate(zip(self.x_data, self.y_data), 1):
            tag = "even" if i % 2 == 0 else "odd"
            self.tree.insert("", "end",
                             values=(i, f"{x:.6g}", f"{y:.6g}"),
                             tags=(tag,))
        self.tree.tag_configure("even", background=BG_PANEL)
        self.tree.tag_configure("odd",  background=BG_CARD)

    # ────────────────────────────────────────────────────────
    #  PLOT
    # ────────────────────────────────────────────────────────
    def _setup_empty_plot(self):
        self.ax.clear()
        self.ax.set_facecolor(BG_PANEL)
        self.ax.text(0.5, 0.5,
                     "Import data  →  Train model  →  Visualize",
                     transform=self.ax.transAxes,
                     ha="center", va="center",
                     color="#333333", fontsize=12,
                     fontfamily="Consolas")
        self._style_axes()
        self.canvas.draw()

    def _style_axes(self):
        self.ax.tick_params(colors=TEXT_SEC, labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(BORDER_LT)
        self.ax.grid(True, color=BORDER_LT, linewidth=0.4, linestyle="--")

    def update_graph(self):
        if not self.x_data:
            self._setup_empty_plot()
            return

        x = np.array(self.x_data)
        y = np.array(self.y_data)

        self.ax.clear()
        self.ax.set_facecolor(BG_PANEL)

        # Scatter
        self.ax.scatter(x, y, color=ORANGE, alpha=0.75, s=45,
                        edgecolors=ORANGE_DK, linewidths=0.5,
                        zorder=3, label="Data Points")

        if self.model is not None:
            x_line = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)
            y_line = self.model.predict(x_line)

            # Regression line
            self.ax.plot(x_line, y_line,
                         color=TEXT, linewidth=1.8,
                         label="Regression Line", zorder=4)

            # Residual bands
            y_pred_all = self.model.predict(x.reshape(-1, 1))
            residuals  = y - y_pred_all
            std        = residuals.std()
            self.ax.fill_between(
                x_line.flatten(),
                y_line - std, y_line + std,
                alpha=0.08, color=ORANGE, label="±1σ Band"
            )

            # Equation annotation
            a = self.model.coef_[0]
            b = self.model.intercept_
            eq = f"y = {a:.4f}x + {b:.4f}"
            self.ax.annotate(
                eq,
                xy=(0.04, 0.93), xycoords="axes fraction",
                color=ORANGE, fontsize=9,
                fontfamily="Consolas",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc=BG_PANEL, ec=ORANGE_DIM, alpha=0.8)
            )

        self.ax.set_xlabel(self.x_col_name, color=TEXT_SEC, fontsize=9)
        self.ax.set_ylabel(self.y_col_name, color=TEXT_SEC, fontsize=9)
        self.ax.set_title("Regression Analysis", color=TEXT, fontsize=11,
                          fontfamily="Consolas")

        legend = self.ax.legend(
            facecolor=BG_CARD, edgecolor=BORDER_LT,
            labelcolor=TEXT, fontsize=8
        )

        self._style_axes()
        self.fig.tight_layout()
        self.canvas.draw()

    def save_plot(self):
        file = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"),
                       ("SVG Vector", "*.svg"),
                       ("PDF Document", "*.pdf")]
        )
        if file:
            self.fig.savefig(file, dpi=150,
                             bbox_inches="tight",
                             facecolor=BG_CARD)
            self._status(f"Plot saved → {file}")

    # ────────────────────────────────────────────────────────
    #  DATA IMPORT — CSV
    # ────────────────────────────────────────────────────────
    def import_csv(self):
        path = filedialog.askopenfilename(
            title="Open CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not path:
            return

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            messagebox.showerror("Import Error", f"Could not read file:\n{exc}")
            return

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols) < 2:
            messagebox.showerror("Import Error",
                                 "File must contain at least 2 numeric columns.")
            return

        filename = os.path.basename(path)
        self._column_selector(df, num_cols, filename)

    def _column_selector(self, df: pd.DataFrame, cols: list, filename: str):
        win = tk.Toplevel(self.root)
        win.title("Select Columns")
        win.geometry("360x300")
        win.configure(bg=BG)
        win.resizable(False, False)
        win.grab_set()

        # Title bar
        tk.Frame(win, bg=ORANGE, height=2).pack(fill="x")
        tk.Label(win, text="SELECT COLUMNS",
                 fg=ORANGE, bg=BG, font=FF_SUB).pack(pady=14)
        tk.Label(win, text=filename,
                 fg=TEXT_SEC, bg=BG, font=FF_SMALL).pack()

        form = tk.Frame(win, bg=BG)
        form.pack(padx=24, pady=12, fill="x")

        tk.Label(form, text="X  (Feature / Input)",
                 fg=TEXT_SEC, bg=BG, font=FF_BODY).grid(row=0, column=0,
                                                          sticky="w", pady=6)
        x_var = tk.StringVar(value=cols[0])
        ttk.Combobox(form, textvariable=x_var, values=cols,
                     state="readonly", width=22).grid(row=0, column=1, padx=10)

        tk.Label(form, text="Y  (Target / Output)",
                 fg=TEXT_SEC, bg=BG, font=FF_BODY).grid(row=1, column=0,
                                                          sticky="w", pady=6)
        y_var = tk.StringVar(value=cols[1])
        ttk.Combobox(form, textvariable=y_var, values=cols,
                     state="readonly", width=22).grid(row=1, column=1, padx=10)

        tk.Label(win, text=f"{len(df)} rows  ·  {len(cols)} numeric columns",
                 fg=TEXT_MUT, bg=BG, font=FF_SMALL).pack()

        def confirm():
            xc = x_var.get()
            yc = y_var.get()

            if xc == yc:
                messagebox.showerror("Error",
                                     "X and Y must be different columns.",
                                     parent=win)
                return

            x_vals = df[xc].dropna().values
            y_vals = df[yc].dropna().values
            n      = min(len(x_vals), len(y_vals))

            self.x_data.clear()
            self.y_data.clear()
            self.x_data.extend(x_vals[:n])
            self.y_data.extend(y_vals[:n])

            self.x_col_name = xc
            self.y_col_name = yc
            self.model = None
            self._reset_metrics()
            self._update_summary()
            self._refresh_table()
            self._header_status(f"◉  {n} POINTS LOADED", SUCCESS)
            self._status(f"Imported {n} rows from '{filename}'  ·  X={xc}  Y={yc}")
            self.update_graph()
            win.destroy()
            messagebox.showinfo(
                "Import Successful ✓",
                f"File loaded successfully!\n\n"
                f"  File    :  {filename}\n"
                f"  Rows    :  {n}\n"
                f"  X col   :  {xc}\n"
                f"  Y col   :  {yc}\n\n"
                f"You can now click  ▶ TRAIN MODEL  to continue."
            )

        btn = HoverButton(win, text="✓  CONFIRM",
                          bg=ORANGE, fg="white",
                          hover_bg=ORANGE_LT, hover_fg="white",
                          font=FF_BTN, command=confirm, padx=20, pady=8)
        btn.pack(pady=16)

    # ────────────────────────────────────────────────────────
    #  MANUAL INPUT
    # ────────────────────────────────────────────────────────
    def manual_input(self):
        win = tk.Toplevel(self.root)
        win.title("Manual Data Entry")
        win.geometry("400x500")
        win.configure(bg=BG)
        win.resizable(False, True)
        win.grab_set()

        tk.Frame(win, bg=ORANGE, height=2).pack(fill="x")
        tk.Label(win, text="MANUAL DATA ENTRY",
                 fg=ORANGE, bg=BG, font=FF_SUB).pack(pady=12)

        # Entry row
        entry_row = tk.Frame(win, bg=BG)
        entry_row.pack(padx=16, fill="x")

        for col, label in enumerate(("X Value", "Y Value")):
            tk.Label(entry_row, text=label, fg=TEXT_SEC, bg=BG,
                     font=FF_BODY).grid(row=0, column=col, padx=5, sticky="w")

        x_wrap = tk.Frame(entry_row, bg=BORDER, pady=1, padx=1)
        x_wrap.grid(row=1, column=0, padx=5, pady=4)
        x_e = tk.Entry(x_wrap, bg=BG_INPUT, fg=TEXT,
                       insertbackground=ORANGE, relief="flat",
                       font=FF_MONO, width=14)
        x_e.pack()

        y_wrap = tk.Frame(entry_row, bg=BORDER, pady=1, padx=1)
        y_wrap.grid(row=1, column=1, padx=5, pady=4)
        y_e = tk.Entry(y_wrap, bg=BG_INPUT, fg=TEXT,
                       insertbackground=ORANGE, relief="flat",
                       font=FF_MONO, width=14)
        y_e.pack()

        add_btn = HoverButton(entry_row, text="+ ADD",
                              bg=ORANGE, fg="white",
                              hover_bg=ORANGE_LT, hover_fg="white",
                              font=FF_BTN, padx=10, pady=6)
        add_btn.grid(row=1, column=2, padx=5, pady=4)

        # List
        list_frame = tk.Frame(win, bg=BG)
        list_frame.pack(fill="both", expand=True, padx=16, pady=4)

        lb = tk.Listbox(list_frame, bg=BG_PANEL, fg=TEXT,
                        font=FF_MONO, selectbackground=ORANGE_DIM,
                        activestyle="none", relief="flat",
                        highlightthickness=0, borderwidth=0)

        for x, y in zip(self.x_data, self.y_data):
            lb.insert(tk.END, f"  X = {x:.6g}   │   Y = {y:.6g}")

        vsb = ttk.Scrollbar(list_frame, command=lb.yview)
        lb.config(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        lb.pack(fill="both", expand=True)

        count_lbl = tk.Label(win, text=f"{len(self.x_data)} point(s)",
                             fg=TEXT_SEC, bg=BG, font=FF_BODY)
        count_lbl.pack()

        def add_row():
            try:
                xv = float(x_e.get())
                yv = float(y_e.get())
            except ValueError:
                messagebox.showerror("Input Error",
                                     "Enter valid numeric values.", parent=win)
                return
            self.x_data.append(xv)
            self.y_data.append(yv)
            lb.insert(tk.END, f"  X = {xv:.6g}   │   Y = {yv:.6g}")
            lb.yview_moveto(1)
            x_e.delete(0, tk.END)
            y_e.delete(0, tk.END)
            x_e.focus()
            count_lbl.config(text=f"{len(self.x_data)} point(s)")

        def remove_sel():
            sel = lb.curselection()
            if not sel:
                return
            idx = sel[0]
            self.x_data.pop(idx)
            self.y_data.pop(idx)
            lb.delete(idx)
            count_lbl.config(text=f"{len(self.x_data)} point(s)")

        def done():
            if len(self.x_data) < 2:
                messagebox.showerror("Error",
                                     "At least 2 data points required.", parent=win)
                return
            self.x_col_name = "X (Manual)"
            self.y_col_name = "Y (Manual)"
            self.model = None
            self._reset_metrics()
            self._update_summary()
            self._refresh_table()
            self._header_status(f"◉  {len(self.x_data)} POINTS LOADED", SUCCESS)
            self._status(f"Manual entry complete — {len(self.x_data)} points ready.")
            self.update_graph()
            win.destroy()

        add_btn.config(command=add_row)
        x_e.bind("<Return>", lambda _: y_e.focus())
        y_e.bind("<Return>", lambda _: add_row())

        btn_row = tk.Frame(win, bg=BG)
        btn_row.pack(pady=10)

        HoverButton(btn_row, text="⊗ Remove Selected",
                    bg=BG_PANEL, fg=DANGER,
                    hover_bg=DANGER, hover_fg="white",
                    font=FF_BODY, command=remove_sel,
                    padx=10, pady=6).pack(side="left", padx=6)

        HoverButton(btn_row, text="✓  DONE",
                    bg=ORANGE, fg="white",
                    hover_bg=ORANGE_LT, hover_fg="white",
                    font=FF_BTN, command=done,
                    padx=14, pady=6).pack(side="left", padx=6)

    # ────────────────────────────────────────────────────────
    #  CLEAR
    # ────────────────────────────────────────────────────────
    def clear_data(self):
        if not self.x_data:
            return
        if not messagebox.askyesno("Confirm Clear",
                                   "Clear all data and reset the model?"):
            return
        self.x_data.clear()
        self.y_data.clear()
        self.model = None
        self.x_col_name = "—"
        self.y_col_name = "—"
        self._reset_metrics()
        self._update_summary()
        self._refresh_table()
        self._header_status("◉  NO DATA", DANGER)
        self._status("Data cleared — ready for new input.")
        self._setup_empty_plot()

    # ────────────────────────────────────────────────────────
    #  TRAIN
    # ────────────────────────────────────────────────────────
    def train_model(self):
        n = len(self.x_data)
        if n < 3:
            messagebox.showerror("Training Error",
                                 "At least 3 data points are required.")
            return

        x = np.array(self.x_data).reshape(-1, 1)
        y = np.array(self.y_data)

        test_sz = self.test_size_var.get()
        # Guard: test set must have at least 1 sample
        min_test = max(1, int(n * test_sz))
        if min_test >= n:
            messagebox.showerror("Training Error",
                                 "Not enough data for the selected test split.")
            return

        X_tr, X_te, y_tr, y_te = train_test_split(
            x, y, test_size=test_sz, random_state=42
        )

        self.model = LinearRegression()
        self.model.fit(X_tr, y_tr)

        y_pred = self.model.predict(X_te)
        a    = self.model.coef_[0]
        b    = self.model.intercept_
        r2   = r2_score(y_te, y_pred)
        mse  = mean_squared_error(y_te, y_pred)
        mae  = mean_absolute_error(y_te, y_pred)

        # Update metric labels
        self.lbl_slope.config(text=f"{a:.5f}",     fg=ORANGE)
        self.lbl_intercept.config(text=f"{b:.5f}", fg=ORANGE)
        self.lbl_r2.config(
            text=f"{r2:.4f}",
            fg=SUCCESS if r2 >= 0.7 else (ORANGE if r2 >= 0.4 else DANGER)
        )
        self.lbl_mse.config(text=f"{mse:.5f}", fg=TEXT)
        self.lbl_mae.config(text=f"{mae:.5f}", fg=TEXT)

        self._header_status(
            f"◉  MODEL READY  ·  R²={r2:.3f}", SUCCESS
        )
        self._status(
            f"Trained on {len(X_tr)} samples  ·  tested on {len(X_te)}  ·  "
            f"slope={a:.4f}  intercept={b:.4f}  R²={r2:.4f}"
        )
        self.update_graph()

    # ────────────────────────────────────────────────────────
    #  PREDICTION WINDOW
    # ────────────────────────────────────────────────────────
    def open_prediction_window(self):
        if self.model is None:
            messagebox.showerror("Error",
                                 "No trained model found.\nTrain the model first!")
            return

        win = tk.Toplevel(self.root)
        win.title("Prediction Window")
        win.geometry("780x560")
        win.minsize(680, 480)
        win.configure(bg=BG)
        win.resizable(True, True)

        # ── Orange top accent ──────────────────────────────
        tk.Frame(win, bg=ORANGE, height=2).pack(fill="x")

        # ── Header bar ────────────────────────────────────
        hdr = tk.Frame(win, bg=BG_CARD, height=44)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="◈  PREDICTION WINDOW",
                 fg=ORANGE, bg=BG_CARD, font=FF_TITLE).pack(side="left", padx=14, pady=6)

        a = self.model.coef_[0]
        b = self.model.intercept_
        eq = f"y = {a:.5f}·x  +  ({b:.5f})"
        tk.Label(hdr, text=eq, fg=TEXT_SEC, bg=BG_CARD,
                 font=FF_MONO).pack(side="right", padx=14)

        # ── Body split: left = input+history  right = chart ──
        body = tk.Frame(win, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        left  = tk.Frame(body, bg=BG_CARD, width=300)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        right = tk.Frame(body, bg=BG_CARD)
        right.pack(side="left", fill="both", expand=True)

        # ══ LEFT PANEL ══════════════════════════════════════

        # ── Input area ────────────────────────────────────
        inp_fr = tk.Frame(left, bg=BG_CARD)
        inp_fr.pack(fill="x", padx=12, pady=12)

        tk.Label(inp_fr, text="INPUT  X  VALUE",
                 fg=ORANGE, bg=BG_CARD, font=("Consolas", 8, "bold")).pack(anchor="w")
        tk.Frame(inp_fr, bg=BORDER_LT, height=1).pack(fill="x", pady=(3, 8))

        entry_wrap = tk.Frame(inp_fr, bg=ORANGE, pady=1, padx=1)
        entry_wrap.pack(fill="x")

        x_entry = tk.Entry(
            entry_wrap, bg=BG_INPUT, fg=TEXT,
            insertbackground=ORANGE, relief="flat",
            font=("Consolas", 15, "bold"),
            highlightthickness=0, justify="center"
        )
        x_entry.pack(fill="x", ipady=10)
        x_entry.focus()

        # ── Result display ────────────────────────────────
        result_card = tk.Frame(left, bg=BG_PANEL)
        result_card.pack(fill="x", padx=12, pady=(0, 8))

        tk.Label(result_card, text="PREDICTED  Y",
                 fg=TEXT_MUT, bg=BG_PANEL, font=("Consolas", 8, "bold")).pack(pady=(8, 0))

        lbl_result = tk.Label(
            result_card, text="—",
            fg=ORANGE, bg=BG_PANEL,
            font=("Consolas", 26, "bold")
        )
        lbl_result.pack(pady=(2, 2))

        lbl_formula_hint = tk.Label(
            result_card, text="",
            fg=TEXT_MUT, bg=BG_PANEL, font=FF_SMALL
        )
        lbl_formula_hint.pack(pady=(0, 8))

        # ── Predict button ────────────────────────────────
        predict_btn = HoverButton(
            left, text="→  PREDICT",
            bg=ORANGE, fg="white",
            hover_bg=ORANGE_LT, hover_fg="white",
            font=("Consolas", 11, "bold"),
            padx=10, pady=10
        )
        predict_btn.pack(fill="x", padx=12, pady=(0, 8))

        # ── History section ───────────────────────────────
        tk.Label(left, text="PREDICTION HISTORY",
                 fg=ORANGE, bg=BG_CARD,
                 font=("Consolas", 8, "bold")).pack(anchor="w", padx=12, pady=(4, 2))
        tk.Frame(left, bg=BORDER_LT, height=1).pack(fill="x", padx=12, pady=(0, 4))

        hist_frame = tk.Frame(left, bg=BG_CARD)
        hist_frame.pack(fill="both", expand=True, padx=12)

        hist_tree = ttk.Treeview(
            hist_frame,
            columns=("num", "x", "y"), show="headings",
            style="LR.Treeview", height=8
        )
        hist_tree.heading("num", text="#")
        hist_tree.heading("x",   text="X Input")
        hist_tree.heading("y",   text="Y Predicted")
        hist_tree.column("num", width=30,  minwidth=30,  stretch=False)
        hist_tree.column("x",   width=110, minwidth=80)
        hist_tree.column("y",   width=110, minwidth=80)

        vsb_h = ttk.Scrollbar(hist_frame, orient="vertical",
                               command=hist_tree.yview)
        hist_tree.configure(yscrollcommand=vsb_h.set)
        vsb_h.pack(side="right", fill="y")
        hist_tree.pack(fill="both", expand=True)

        # ── Bottom action row ─────────────────────────────
        action_row = tk.Frame(left, bg=BG_CARD)
        action_row.pack(fill="x", padx=12, pady=8)

        def clear_history():
            for row in hist_tree.get_children():
                hist_tree.delete(row)
            pred_history.clear()
            _reset_mini_chart()
            lbl_result.config(text="—", fg=ORANGE)
            lbl_formula_hint.config(text="")
            self._status("Prediction history cleared.")

        def export_history():
            if not pred_history:
                messagebox.showinfo("Export", "No predictions to export.",
                                    parent=win)
                return
            path = filedialog.asksaveasfilename(
                parent=win,
                defaultextension=".csv",
                filetypes=[("CSV File", "*.csv")],
                title="Export Prediction History"
            )
            if path:
                pd.DataFrame(pred_history,
                             columns=["X_input", "Y_predicted"]).to_csv(path, index=False)
                self._status(f"History exported → {path}")

        HoverButton(action_row, text="⊗ Clear",
                    bg=BG_PANEL, fg=DANGER,
                    hover_bg=DANGER, hover_fg="white",
                    font=FF_BODY, command=clear_history,
                    padx=8, pady=5).pack(side="left", padx=(0, 4))

        HoverButton(action_row, text="⬇ Export CSV",
                    bg=BG_PANEL, fg=TEXT_SEC,
                    hover_bg=BORDER_LT, hover_fg=TEXT,
                    font=FF_BODY, command=export_history,
                    padx=8, pady=5).pack(side="left")

        # ══ RIGHT PANEL — Mini chart ═════════════════════════

        tk.Label(right, text="LIVE CHART",
                 fg=ORANGE, bg=BG_CARD,
                 font=("Consolas", 8, "bold")).pack(anchor="w", padx=10, pady=(8, 2))
        tk.Frame(right, bg=BORDER_LT, height=1).pack(fill="x", padx=10, pady=(0, 4))

        mini_fig, mini_ax = plt.subplots(figsize=(5.2, 4.4))
        mini_fig.patch.set_facecolor(BG_CARD)
        mini_ax.set_facecolor(BG_PANEL)

        mini_canvas = FigureCanvasTkAgg(mini_fig, master=right)
        mini_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # ── Mini chart helpers ────────────────────────────
        pred_history: list = []   # [(x, y), ...]
        pred_counter = [0]

        x_arr = np.array(self.x_data)
        y_arr = np.array(self.y_data)
        x_line = np.linspace(x_arr.min(), x_arr.max(), 200).reshape(-1, 1)
        y_line = self.model.predict(x_line)

        def _reset_mini_chart():
            mini_ax.clear()
            mini_ax.set_facecolor(BG_PANEL)
            mini_ax.scatter(x_arr, y_arr,
                            color=ORANGE, alpha=0.6, s=30, zorder=2,
                            label="Data")
            mini_ax.plot(x_line, y_line,
                         color=TEXT, linewidth=1.5, zorder=3,
                         label="Regression")
            for (px, py) in pred_history:
                mini_ax.scatter(px, py, color="#ff0099", s=80,
                                marker="*", zorder=5)
            _style_mini()
            mini_canvas.draw()

        def _style_mini():
            mini_ax.tick_params(colors=TEXT_SEC, labelsize=7)
            for sp in mini_ax.spines.values():
                sp.set_edgecolor(BORDER_LT)
            mini_ax.grid(True, color=BORDER_LT, linewidth=0.3, linestyle="--")
            mini_ax.set_xlabel(self.x_col_name, color=TEXT_SEC, fontsize=8)
            mini_ax.set_ylabel(self.y_col_name, color=TEXT_SEC, fontsize=8)
            mini_ax.legend(facecolor=BG_PANEL, edgecolor=BORDER_LT,
                           labelcolor=TEXT, fontsize=7)
            mini_fig.tight_layout()

        _reset_mini_chart()

        # ── Core predict logic ────────────────────────────
        def do_predict(*_):
            raw = x_entry.get().strip()
            if not raw:
                return
            try:
                xv = float(raw)
            except ValueError:
                lbl_result.config(text="ERROR", fg=DANGER)
                lbl_formula_hint.config(text="Enter a valid number")
                return

            yv = self.model.predict([[xv]])[0]
            pred_counter[0] += 1
            pred_history.append((xv, yv))

            # Update result display
            lbl_result.config(text=f"{yv:.5f}", fg=ORANGE)
            lbl_formula_hint.config(
                text=f"f({xv:.4g}) = {a:.4f}·{xv:.4g} + {b:.4f}",
                fg=TEXT_MUT
            )

            # Add to history table
            tag = "even" if pred_counter[0] % 2 == 0 else "odd"
            hist_tree.insert("", "end",
                             values=(pred_counter[0], f"{xv:.6g}", f"{yv:.6g}"),
                             tags=(tag,))
            hist_tree.tag_configure("even", background=BG_PANEL)
            hist_tree.tag_configure("odd",  background=BG_CARD)
            hist_tree.yview_moveto(1)

            # Plot the new prediction point
            mini_ax.scatter(xv, yv,
                            color="#ff0099", s=90, zorder=5,
                            marker="*", edgecolors="white", linewidths=0.3)
            # Drop a vertical dashed line to axis
            mini_ax.plot([xv, xv], [mini_ax.get_ylim()[0], yv],
                         color="#ff0099", linewidth=0.7,
                         linestyle=":", alpha=0.5, zorder=4)
            # Annotate only the latest point
            mini_ax.annotate(
                f"  ({xv:.3g}, {yv:.3g})",
                xy=(xv, yv), fontsize=7,
                color="#ff0099", fontfamily="Consolas"
            )

            mini_canvas.draw()
            self._status(f"Prediction #{pred_counter[0]}:  X = {xv}  →  Y = {yv:.5f}")

            # Also mark on main chart
            self.ax.scatter([xv], [yv],
                            color="#ff0099", s=100, zorder=6,
                            marker="*", edgecolors="white", linewidths=0.4)
            self.canvas.draw()

        predict_btn.config(command=do_predict)
        x_entry.bind("<Return>", do_predict)


# ============================================================
#  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = LinearRegressionApp(root)
    root.mainloop()