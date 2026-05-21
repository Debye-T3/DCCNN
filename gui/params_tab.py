"""Tab 2: Batch parameter defaults + per-file override table."""

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit,
    QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QPushButton, QSplitter,
)
from PySide6.QtCore import Qt


FIELD_KEYS = [
    "sample_name", "sample_id",
    "position_x", "position_y", "position_z",
    "position_polar", "position_tilt", "position_azimuth",
    "temperature_K", "photon_energy_eV", "polarization", "slit",
    "work_function_eV",
]

FIELD_LABELS = [
    "Sample Name", "Sample ID",
    "Position X", "Position Y", "Position Z",
    "Polar", "Tilt", "Azimuth",
    "Temperature (K)", "Photon Energy hv (eV)", "Polarization", "Slit",
    "Work Function Φ (eV)",
]

FIELD_DEFAULTS = {
    "work_function_eV": "4.2",
}


class ParamsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._file_paths = []
        self._overrides = {}

        layout = QVBoxLayout(self)

        title = QLabel("Experiment Parameters")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        splitter = QSplitter(Qt.Horizontal)

        # Left: batch defaults
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        form_group = QGroupBox("Batch Defaults")
        form_layout = QFormLayout(form_group)
        self._fields = {}
        for key, label in zip(FIELD_KEYS, FIELD_LABELS):
            edit = QLineEdit()
            edit.setPlaceholderText(label)
            if key in FIELD_DEFAULTS:
                edit.setText(FIELD_DEFAULTS[key])
            form_layout.addRow(label, edit)
            self._fields[key] = edit

        left_layout.addWidget(form_group)

        tip = QLabel("These values apply to all files. Edit individual\nfiles in the table on the right.")
        tip.setStyleSheet("color: #888; font-size: 10pt; padding: 8px;")
        left_layout.addWidget(tip)
        left_layout.addStretch()

        splitter.addWidget(left)

        # Right: per-file override table
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        table_group = QGroupBox("Per-File Overrides")
        table_layout = QVBoxLayout(table_group)

        self.table = QTableWidget()
        self.table.setColumnCount(len(FIELD_KEYS) + 1)
        self.table.setHorizontalHeaderLabels(["File"] + FIELD_LABELS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.cellChanged.connect(self._on_cell_changed)
        self._rebuilding = False
        table_layout.addWidget(self.table)

        table_tip = QLabel("Click any cell to override the batch default. Empty = use default.")
        table_tip.setStyleSheet("color: #888; font-size: 10pt; padding: 4px;")
        table_layout.addWidget(table_tip)

        right_layout.addWidget(table_group)
        splitter.addWidget(right)

        splitter.setSizes([360, 540])
        layout.addWidget(splitter)

        nav = QHBoxLayout()
        back_btn = QPushButton("← Back: Select Files")
        back_btn.clicked.connect(self._go_back)
        next_btn = QPushButton("Next: Preview →")
        next_btn.clicked.connect(self._go_next)
        nav.addWidget(back_btn)
        nav.addStretch()
        nav.addWidget(next_btn)
        layout.addLayout(nav)

    def set_files(self, file_paths):
        self._file_paths = file_paths
        self._rebuild_table()

    def _rebuild_table(self):
        self._rebuilding = True
        self.table.setRowCount(len(self._file_paths))
        for i, fp in enumerate(self._file_paths):
            name_item = QTableWidgetItem(Path(fp).name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 0, name_item)
            for j, key in enumerate(FIELD_KEYS):
                val = self._overrides.get(fp, {}).get(key, "")
                self.table.setItem(i, j + 1, QTableWidgetItem(str(val) if val else ""))
        self._rebuilding = False

    def _on_cell_changed(self, row, col):
        if self._rebuilding or row >= len(self._file_paths):
            return
        fp = self._file_paths[row]
        key = FIELD_KEYS[col - 1]
        item = self.table.item(row, col)
        text = item.text().strip() if item else ""
        if fp not in self._overrides:
            self._overrides[fp] = {}
        if text:
            self._overrides[fp][key] = text
        else:
            self._overrides[fp].pop(key, None)
            if not self._overrides[fp]:
                del self._overrides[fp]

    def get_all_params(self):
        batch = {}
        for key, edit in self._fields.items():
            text = edit.text().strip()
            if text:
                try:
                    batch[key] = float(text)
                except ValueError:
                    batch[key] = text
        batch["_overrides"] = dict(self._overrides)
        return batch

    def _go_back(self):
        w = self.window()
        if w and hasattr(w, "tabs"):
            w.tabs.setCurrentIndex(0)

    def _go_next(self):
        w = self.window()
        if w and hasattr(w, "tabs"):
            w.tabs.setCurrentIndex(2)
