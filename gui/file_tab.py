"""Tab 1: File selection with drag-drop, browse, and file list."""

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox,
)
from PySide6.QtCore import Signal, Qt


VALID_EXTENSIONS = {".txt", ".pxt", ".bin"}


class FileTab(QWidget):
    files_changed = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._file_paths = []

        layout = QVBoxLayout(self)

        title = QLabel("Select ARPES Data Files")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        self.drop_label = QLabel("Drag & drop .txt, .pxt, or .bin files here\n— or —")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setStyleSheet(
            "border: 2px dashed #888; border-radius: 8px; padding: 30px; color: #888;"
        )
        layout.addWidget(self.drop_label)

        btn_row = QHBoxLayout()
        browse_btn = QPushButton("Browse Files")
        browse_btn.clicked.connect(self._browse)
        btn_row.addStretch()
        btn_row.addWidget(browse_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.file_list = QListWidget()
        layout.addWidget(self.file_list)

        bottom = QHBoxLayout()
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected)
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_all)
        self.next_btn = QPushButton("Next: Parameters →")
        self.next_btn.clicked.connect(self._go_next)
        bottom.addWidget(remove_btn)
        bottom.addWidget(clear_btn)
        bottom.addStretch()
        bottom.addWidget(self.next_btn)
        layout.addLayout(bottom)

    def _browse(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select ARPES Data Files", "",
            "ARPES Files (*.txt *.pxt *.bin);;All Files (*)"
        )
        for p in paths:
            self._add_file(Path(p))

    def _add_file(self, path: Path):
        path = path.resolve()
        if str(path) in self._file_paths:
            return
        suffix = path.suffix.lower()
        if suffix not in VALID_EXTENSIONS:
            return
        self._file_paths.append(str(path))
        item = QListWidgetItem(f"{path.name}  —  {suffix[1:].upper()}")
        item.setToolTip(str(path))
        self.file_list.addItem(item)
        self.files_changed.emit(self._file_paths)

    def _remove_selected(self):
        for item in self.file_list.selectedItems():
            idx = self.file_list.row(item)
            self.file_list.takeItem(idx)
            if idx < len(self._file_paths):
                del self._file_paths[idx]
        self.files_changed.emit(self._file_paths)

    def _clear_all(self):
        self.file_list.clear()
        self._file_paths.clear()
        self.files_changed.emit(self._file_paths)

    def _go_next(self):
        if not self._file_paths:
            QMessageBox.warning(self, "No Files", "Please add at least one file.")
            return
        w = self.window()
        if w and hasattr(w, "tabs"):
            w.tabs.setCurrentIndex(1)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_file():
                self._add_file(path)

    def get_files(self):
        return list(self._file_paths)
