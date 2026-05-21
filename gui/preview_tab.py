"""Tab 3: Preview with k-space toggle, colormap picker, contrast sliders."""

from pathlib import Path

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QCheckBox, QSlider, QGroupBox, QMessageBox,
)
from PySide6.QtCore import Qt, Signal

from converter.readers.txt_reader import read_txt
from converter.readers.pxt_reader import read_pxt
from converter.preview import compute_contrast, to_kspace
from converter.engine import detect_format


COLORMAPS = ["inferno", "viridis", "plasma", "gray", "jet", "turbo"]


class PreviewTab(QWidget):
    preview_enabled_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._file_paths = []
        self._current_data = None
        self._current_energy = None
        self._current_angle = None

        layout = QVBoxLayout(self)

        title = QLabel("Preview")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # Controls row
        ctrl_row = QHBoxLayout()

        ctrl_row.addWidget(QLabel("File:"))
        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(self._on_file_selected)
        ctrl_row.addWidget(self.file_combo)

        ctrl_row.addSpacing(16)

        self.kspace_cb = QCheckBox("k-space")
        self.kspace_cb.stateChanged.connect(lambda: self._refresh_preview())
        ctrl_row.addWidget(self.kspace_cb)

        ctrl_row.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.currentTextChanged.connect(lambda: self._refresh_preview())
        ctrl_row.addWidget(self.cmap_combo)

        self.log_cb = QCheckBox("Log scale")
        self.log_cb.setChecked(True)
        self.log_cb.stateChanged.connect(lambda: self._refresh_preview())
        ctrl_row.addWidget(self.log_cb)

        self.save_preview_cb = QCheckBox("Save with conversion")
        self.save_preview_cb.setChecked(True)
        self.save_preview_cb.stateChanged.connect(
            lambda s: self.preview_enabled_changed.emit(s == Qt.Checked.value)
        )
        ctrl_row.addWidget(self.save_preview_cb)

        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # Main area: canvas + contrast
        main_row = QHBoxLayout()

        self.canvas = FigureCanvas(Figure(figsize=(7, 5)))
        self.ax = self.canvas.figure.add_subplot(111)
        self.canvas.figure.tight_layout()
        main_row.addWidget(self.canvas, 1)

        contrast_group = QGroupBox("Contrast")
        contrast_layout = QVBoxLayout(contrast_group)

        contrast_layout.addWidget(QLabel("vmin percentile:"))
        self.vmin_slider = QSlider(Qt.Horizontal)
        self.vmin_slider.setRange(0, 10)
        self.vmin_slider.setValue(1)
        self.vmin_label = QLabel("1%")
        self.vmin_slider.valueChanged.connect(
            lambda v: self.vmin_label.setText(f"{v}%")
        )
        contrast_layout.addWidget(self.vmin_slider)
        contrast_layout.addWidget(self.vmin_label)

        contrast_layout.addWidget(QLabel("vmax percentile:"))
        self.vmax_slider = QSlider(Qt.Horizontal)
        self.vmax_slider.setRange(90, 100)
        self.vmax_slider.setValue(99)
        self.vmax_label = QLabel("99%")
        self.vmax_slider.valueChanged.connect(
            lambda v: self.vmax_label.setText(f"{v}%")
        )
        contrast_layout.addWidget(self.vmax_slider)
        contrast_layout.addWidget(self.vmax_label)

        refresh_btn = QPushButton("Refresh Preview")
        refresh_btn.clicked.connect(self._refresh_preview)
        contrast_layout.addWidget(refresh_btn)

        contrast_layout.addStretch()
        main_row.addWidget(contrast_group)

        layout.addLayout(main_row)

        nav = QHBoxLayout()
        back_btn = QPushButton("← Back: Parameters")
        back_btn.clicked.connect(lambda: self._nav_to(1))
        next_btn = QPushButton("Next: Convert →")
        next_btn.clicked.connect(lambda: self._nav_to(3))
        nav.addWidget(back_btn)
        nav.addStretch()
        nav.addWidget(next_btn)
        layout.addLayout(nav)

    def set_files(self, file_paths):
        self._file_paths = file_paths
        self.file_combo.clear()
        for fp in file_paths:
            self.file_combo.addItem(Path(fp).name, fp)
        if file_paths:
            self._on_file_selected(0)

    def _on_file_selected(self, index):
        if index < 0 or not self._file_paths:
            return
        fp = self._file_paths[index]
        try:
            fmt = detect_format(Path(fp))
            if fmt == "txt":
                data = read_txt(Path(fp))
            elif fmt == "pxt":
                data = read_pxt(Path(fp))
            else:
                return
            self._current_data = data["spectrum"]
            self._current_energy = data["energy"]
            self._current_angle = data["thetax"]
            self._refresh_preview()
        except Exception as exc:
            QMessageBox.warning(self, "Preview Error", f"Could not read file:\n{exc}")

    def _refresh_preview(self):
        if self._current_data is None:
            return
        self.ax.clear()
        data = np.clip(self._current_data, a_min=0.0, a_max=None)
        use_kspace = self.kspace_cb.isChecked()
        use_log = self.log_cb.isChecked()
        cmap = self.cmap_combo.currentText()
        pmin = self.vmin_slider.value()
        pmax = self.vmax_slider.value()

        if use_kspace:
            params_tab = self._get_params_tab()
            hv = None
            work_function = 4.2
            if params_tab:
                batch = params_tab.get_all_params()
                hv = batch.get("photon_energy_eV")
                wf = batch.get("work_function_eV", 4.2)
                if wf:
                    try:
                        work_function = float(wf)
                    except (ValueError, TypeError):
                        pass
            if hv is None or hv == "":
                QMessageBox.warning(
                    self, "Missing hv",
                    "Please enter Photon Energy (hv) in the Parameters tab for k-space conversion."
                )
                self.kspace_cb.setChecked(False)
                use_kspace = False

        if use_kspace:
            try:
                hv_val = float(hv)
                k_axis, e_axis = to_kspace(self._current_energy, self._current_angle, hv_val, work_function)
                x_label = r"$k_{\parallel}$ [$\AA^{-1}$]"
            except Exception:
                use_kspace = False
                e_axis = self._current_energy
                k_axis = self._current_angle
                x_label = "Angle [deg]"
        else:
            e_axis = self._current_energy
            k_axis = self._current_angle
            x_label = "Angle [deg]"

        extent = [
            float(k_axis[0]), float(k_axis[-1]),
            float(e_axis[0]), float(e_axis[-1]),
        ]

        norm = None
        if use_log:
            vmin, vmax = compute_contrast(data, pmin, pmax)
            norm = LogNorm(vmin=vmin, vmax=vmax)

        kwargs = {"origin": "lower", "aspect": "auto", "cmap": cmap, "extent": extent}
        if norm is not None:
            kwargs["norm"] = norm
        im = self.ax.imshow(data, **kwargs)
        self.canvas.figure.colorbar(im, ax=self.ax)
        title = "ARPES Spectrum"
        if use_log:
            title += " (log)"
        if use_kspace:
            title += " — k-space"
        self.ax.set_title(title)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel("Energy [eV]")
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def get_settings(self):
        return {
            "cmap": self.cmap_combo.currentText(),
            "pmin": float(self.vmin_slider.value()),
            "pmax": float(self.vmax_slider.value()),
            "use_log": self.log_cb.isChecked(),
            "use_kspace": self.kspace_cb.isChecked(),
        }

    def is_preview_enabled(self):
        return self.save_preview_cb.isChecked()

    def _get_params_tab(self):
        w = self.window()
        if w and hasattr(w, "params_tab"):
            return w.params_tab
        return None

    def _nav_to(self, idx):
        w = self.window()
        if w and hasattr(w, "tabs"):
            w.tabs.setCurrentIndex(idx)
