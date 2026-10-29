"""TakeSync desktop window (PySide6). All processing lives in project.py and
runs on a worker thread; this file only shows state and collects input."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QDesktopServices
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QSlider, QSpinBox, QSplitter,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .cli import parse_time
from .media import Cancelled, MediaError, find_tool
from .planner import CutPlan, Segment
from .project import Project
from .render import RenderSettings, fmt_time
from .validate import passed, report

try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget
    HAVE_PLAYER = True
except Exception:  # missing multimedia backend: fall back to the external player
    HAVE_PLAYER = False

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".m4v", ".flv", ".ts", ".avi", ".webm"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".aif", ".aiff"}
STATUS_COLOURS = {"ok": "#2e7d32", "reference": "#1565c0", "manual": "#6a1b9a",
                  "low": "#ef6c00", "ambiguous": "#ef6c00", "failed": "#c62828"}


class Bridge(QObject):
    progress = Signal(str, int, int)
    done = Signal(object)
    failed = Signal(str)


class TakesTable(QTableWidget):
    COLS = ["Use", "File", "Video", "Offset (ms)", "Status", "Match", "Usable song time", "Warnings"]

    def __init__(self, on_drop):
        super().__init__(0, len(self.COLS))
        self.on_drop = on_drop
        self.setHorizontalHeaderLabels(self.COLS)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.verticalHeader().setVisible(False)
        h = self.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.ResizeToContents)
        h.setSectionResizeMode(7, QHeaderView.Stretch)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.on_drop([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])
        e.acceptProposedAction()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TakeSync")
        self.resize(1400, 900)
        self.setAcceptDrops(True)
        self.project = Project()
        self.plan: CutPlan | None = None
        self.preview_path: str | None = None
        self.preview_dir = tempfile.mkdtemp(prefix="takesync_preview_")
        self.preview_count = 0
        self.cancel_event: threading.Event | None = None
        self.bridge: Bridge | None = None
        self._task_cb = None
        self._updating = False
        self._build()
        self._refresh_all()

    # ================================================================ layout
    def _build(self):
        left = QWidget()
        lv = QVBoxLayout(left)

        g1 = QGroupBox("1. Recordings  (drag OBS files here)")
        v = QVBoxLayout(g1)
        self.table = TakesTable(self._dropped)
        self.table.itemChanged.connect(self._take_item_changed)
        v.addWidget(self.table)
        row = QHBoxLayout()
        for text, fn in (("Add files...", self._add_files), ("Remove selected", self._remove_selected),
                         ("Reset offset", self._reset_offset)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        v.addLayout(row)
        lv.addWidget(g1, 3)

        g2 = QGroupBox("2. Reference and sync")
        f = QFormLayout(g2)
        rrow = QHBoxLayout()
        self.ref_combo = QComboBox()
        self.ref_combo.currentIndexChanged.connect(self._ref_changed)
        rrow.addWidget(self.ref_combo, 1)
        b = QPushButton("Use audio file...")
        b.clicked.connect(self._choose_ref_audio)
        rrow.addWidget(b)
        f.addRow("Reference:", rrow)
        self.ref_note = QLabel()
        self.ref_note.setWordWrap(True)
        self.ref_note.setStyleSheet("color: #666")
        f.addRow("", self.ref_note)
        arow = QHBoxLayout()
        self.cam_delay = QSpinBox()
        self.cam_delay.setRange(-1000, 1000)
        self.cam_delay.setSuffix(" ms")
        self.cam_delay.setToolTip("If your camera picture lags the audio (lips late), enter the lag here. "
                                  "Same as OBS 'Sync Offset'. Leave at 0 if your raw OBS recordings look in sync.")
        self.cam_delay.valueChanged.connect(self._cam_delay_changed)
        arow.addWidget(QLabel("Camera delay:"))
        arow.addWidget(self.cam_delay)
        self.track = QSpinBox()
        self.track.setRange(1, 6)
        self.track.setToolTip("Which audio track of the OBS files holds the FL Studio playback")
        self.track.valueChanged.connect(self._track_changed)
        arow.addWidget(QLabel("  Audio track:"))
        arow.addWidget(self.track)
        arow.addStretch()
        self.btn_analyze = QPushButton("Analyze")
        self.btn_analyze.setMinimumWidth(140)
        self.btn_analyze.clicked.connect(self._analyze)
        arow.addWidget(self.btn_analyze)
        f.addRow(arow)
        lv.addWidget(g2)

        g3 = QGroupBox("3. Excerpt and style")
        f = QFormLayout(g3)
        erow = QHBoxLayout()
        self.ex_start = QLineEdit()
        self.ex_end = QLineEdit()
        for w in (self.ex_start, self.ex_end):
            w.setPlaceholderText("m:ss.mmm")
            w.setMaximumWidth(110)
        erow.addWidget(QLabel("Start"))
        erow.addWidget(self.ex_start)
        erow.addWidget(QLabel("End"))
        erow.addWidget(self.ex_end)
        b = QPushButton("Use shared overlap")
        b.clicked.connect(self._use_auto_excerpt)
        erow.addWidget(b)
        erow.addStretch()
        f.addRow("Song time:", erow)
        self.overlap_label = QLabel("-")
        f.addRow("Shared overlap:", self.overlap_label)
        srow = QHBoxLayout()
        self.pacing = QComboBox()
        self.pacing.addItems(["Natural (~2.6 s)", "Fast (~1.3 s)"])
        srow.addWidget(self.pacing)
        srow.addWidget(QLabel("  Seed:"))
        self.seed = QSpinBox()
        self.seed.setRange(1, 99999)
        srow.addWidget(self.seed)
        srow.addWidget(QLabel("  Crop centre:"))
        self.crop = QSlider(Qt.Horizontal)
        self.crop.setRange(0, 100)
        self.crop.setValue(50)
        self.crop.setMaximumWidth(160)
        self.crop_label = QLabel("50%")
        self.crop.valueChanged.connect(lambda v: self.crop_label.setText(f"{v}%"))
        srow.addWidget(self.crop)
        srow.addWidget(self.crop_label)
        srow.addStretch()
        f.addRow("Cuts:", srow)
        lv.addWidget(g3)

        lv.addWidget(QLabel("Log"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        lv.addWidget(self.log, 2)

        right = QWidget()
        rv = QVBoxLayout(right)
        prow = QHBoxLayout()
        self.btn_preview = QPushButton("Generate Preview")
        self.btn_preview.setToolTip("Plan new cuts from the settings and render a low-res draft")
        self.btn_preview.clicked.connect(self._generate_preview)
        self.btn_update = QPushButton("Update Preview")
        self.btn_update.setToolTip("Render the edited cut list again")
        self.btn_update.clicked.connect(self._update_preview)
        self.btn_export = QPushButton("Export...")
        self.btn_export.setStyleSheet("font-weight: bold")
        self.btn_export.clicked.connect(self._export)
        for b in (self.btn_preview, self.btn_update, self.btn_export):
            b.setMinimumHeight(32)
            prow.addWidget(b)
        rv.addLayout(prow)

        if HAVE_PLAYER:
            self.video = QVideoWidget()
            self.video.setMinimumSize(270, 480)
            self.player = QMediaPlayer(self)
            self.audio_out = QAudioOutput(self)
            self.player.setAudioOutput(self.audio_out)
            self.player.setVideoOutput(self.video)
            rv.addWidget(self.video, 3)
        else:
            self.player = None
            rv.addWidget(QLabel("(In-app player unavailable - use 'Open in player')"))
        crow = QHBoxLayout()
        self.btn_play = QPushButton("Play / Pause")
        self.btn_play.clicked.connect(self._play_pause)
        self.btn_ext = QPushButton("Open in player")
        self.btn_ext.clicked.connect(self._open_external)
        crow.addWidget(self.btn_play)
        crow.addWidget(self.btn_ext)
        crow.addStretch()
        rv.addLayout(crow)

        rv.addWidget(QLabel("Cut list  (edit start times, pick takes; click a row to jump the preview)"))
        self.cuts = QTableWidget(0, 4)
        self.cuts.setHorizontalHeaderLabels(["Start", "End", "Length", "Take"])
        self.cuts.verticalHeader().setVisible(True)
        self.cuts.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.cuts.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.cuts.itemChanged.connect(self._cut_item_changed)
        self.cuts.cellClicked.connect(self._cut_clicked)
        rv.addWidget(self.cuts, 2)
        krow = QHBoxLayout()
        for text, fn in (("Split selected", self._split_cut), ("Merge into previous", self._merge_cut)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            krow.addWidget(b)
        krow.addStretch()
        rv.addLayout(krow)
        self.plan_status = QLabel("")
        self.plan_status.setWordWrap(True)
        rv.addWidget(self.plan_status)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        bottom = QHBoxLayout()
        self.status = QLabel("Add your OBS recordings to start.")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(300)
        self.progress.setTextVisible(False)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_cancel.setEnabled(False)
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.progress)
        bottom.addWidget(self.btn_cancel)

        central = QWidget()
        cv = QVBoxLayout(central)
        cv.addWidget(split, 1)
        cv.addLayout(bottom)
        self.setCentralWidget(central)

    # ================================================================ helpers
    def say(self, text: str):
        self.log.appendPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def error(self, title: str, text: str):
        self.say(f"ERROR: {title}: {text}")
        QMessageBox.critical(self, title, text)

    @property
    def busy(self) -> bool:
        return self.cancel_event is not None

    def run_task(self, label: str, fn, on_done):
        """Run fn(cancel_event, progress) on a worker thread."""
        if self.busy:
            return
        self.cancel_event = threading.Event()
        self.bridge = Bridge()
        self.bridge.progress.connect(self._task_progress)
        self.bridge.done.connect(self._task_done)
        self.bridge.failed.connect(self._task_failed)
        self._task_cb = on_done
        self._task_label = label
        self.status.setText(label + "...")
        self.progress.setRange(0, 0)
        self._set_enabled(False)
        bridge, cancel = self.bridge, self.cancel_event

        def emit(msg, done=0, total=0):
            bridge.progress.emit(str(msg), int(done), int(total))

        def work():
            try:
                result = fn(cancel, emit)
            except Cancelled:
                bridge.failed.emit("__cancelled__")
            except (MediaError, ValueError) as e:
                bridge.failed.emit(str(e))
            except Exception:
                bridge.failed.emit(traceback.format_exc())
            else:
                bridge.done.emit(result)
        threading.Thread(target=work, daemon=True).start()

    def _task_progress(self, msg, done, total):
        if msg:
            self.status.setText(msg)
        if total > 0:
            self.progress.setTextVisible(True)
            self.progress.setRange(0, total)
            self.progress.setValue(done)

    def _finish_task(self):
        self.cancel_event = None
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self._set_enabled(True)
        self._refresh_all()

    def _task_done(self, result):
        cb = self._task_cb
        self._finish_task()
        self.status.setText(self._task_label + " - done.")
        if cb:
            try:
                cb(result)
            except Exception:
                self.error("Unexpected error", traceback.format_exc())
        self._refresh_all()

    def _task_failed(self, msg):
        self._finish_task()
        if msg == "__cancelled__":
            self.status.setText(self._task_label + " - cancelled.")
            self.say(f"{self._task_label}: cancelled.")
        else:
            self.status.setText(self._task_label + " - failed.")
            self.error(self._task_label + " failed", msg)

    def _cancel(self):
        if self.cancel_event:
            self.cancel_event.set()
            self.status.setText("Cancelling...")

    def _set_enabled(self, on: bool):
        for w in (self.btn_analyze, self.btn_preview, self.btn_update, self.btn_export, self.table,
                  self.ref_combo, self.cuts):
            w.setEnabled(on)
        self.btn_cancel.setEnabled(not on)

    # ================================================================ files
    def _dropped(self, paths):
        vids = [p for p in paths if Path(p).suffix.lower() in VIDEO_EXT]
        auds = [p for p in paths if Path(p).suffix.lower() in AUDIO_EXT]
        other = [p for p in paths if p not in vids and p not in auds]
        if other:
            self.say("Ignored (not a video/audio file): " + ", ".join(os.path.basename(p) for p in other))
        if auds:
            self._set_ref_audio(auds[0])
        if vids:
            self._add_paths(vids)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self._dropped([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add OBS recordings", "",
                                                "Videos (" + " ".join("*" + e for e in sorted(VIDEO_EXT)) + ")")
        if paths:
            self._add_paths(paths)

    def _add_paths(self, paths):
        def work(cancel, emit):
            return self.project.add(paths, cancel=cancel, progress=lambda m, i, n: emit(m, i, n))

        def done(skipped):
            for p in skipped:
                self.say(f"Already added: {os.path.basename(p)}")
            for t in self.project.takes:
                for w in t.info.warnings:
                    self.say(f"{t.name}: {w}")
            self._invalidate("Files changed - run Analyze.")
        self.run_task("Reading files", work, done)

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if rows:
            self.project.remove(rows)
            self._invalidate("Files changed - run Analyze.")

    def _choose_ref_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Reference audio (exported song)", "",
                                              "Audio (" + " ".join("*" + e for e in sorted(AUDIO_EXT)) + ")")
        if path:
            self._set_ref_audio(path)

    def _set_ref_audio(self, path):
        def work(cancel, emit):
            emit(f"Reading {os.path.basename(path)}")
            self.project.set_reference_wav(path, cancel=cancel)

        self.run_task("Loading reference audio", work, lambda _: self._invalidate("Reference changed - run Analyze."))

    def _ref_changed(self, idx):
        if self._updating or idx < 0:
            return
        data = self.ref_combo.itemData(idx)
        if data == "wav":
            return
        if self.project.ref_wav is not None:
            self.project.set_reference_wav(None)
        self.project.ref_take = int(data)
        self._invalidate("Reference changed - run Analyze.")

    def _cam_delay_changed(self, v):
        self.project.camera_delay = v / 1000.0
        if self.project.analyzed:
            self._refresh_all()

    def _track_changed(self, v):
        if self._updating or self.project.audio_track == v - 1:
            return
        self.project.audio_track = v - 1

        def work(cancel, emit):
            self.project.reload_audio(cancel=cancel, progress=lambda m, i, n: emit(m, i, n))
        self.run_task("Reading audio track", work, lambda _: self._invalidate("Audio track changed - run Analyze."))

    def _invalidate(self, msg):
        self.project.analyzed = False
        for t in self.project.takes:
            t.result = t.auto_result = None
            t.manual_offset = None
            t.enabled = True
        self.plan = None
        self.status.setText(msg)
        self._refresh_all()

    # ================================================================ analysis
    def _analyze(self):
        if not self.project.takes:
            self.error("Nothing to analyze", "Add some OBS recordings first.")
            return

        def work(cancel, emit):
            self.project.analyze(cancel=cancel, progress=lambda m, i, n: emit(m, i, n))
            return self.project.alignment_report()

        def done(rep):
            self.say("\n" + rep)
            self._use_auto_excerpt()
            flagged = [t.name for t in self.project.takes if not t.enabled]
            if flagged:
                QMessageBox.warning(self, "Some takes need attention",
                                    "These takes are NOT used yet:\n\n" + "\n".join(flagged) +
                                    "\n\nSee the Warnings column. Tick 'Use' to accept a flagged offset, "
                                    "or type the correct offset (ms) into the Offset column.")
        self.run_task("Analyzing", work, done)

    def _use_auto_excerpt(self):
        ex = self.project.auto_excerpt()
        if ex is None:
            span = self.project.union_span()
            if span:
                self.say("No song region is shared by ALL used takes; filled in the union instead. "
                         "Parts covered by only one take will not have cuts.")
                ex = span
        if ex:
            self.ex_start.setText(fmt_time(ex[0]))
            self.ex_end.setText(fmt_time(ex[1]))

    def _take_item_changed(self, item):
        if self._updating:
            return
        row, col = item.row(), item.column()
        if col == 0:
            want = item.checkState() == Qt.Checked
            t = self.project.takes[row]
            if want and t.offset is None:
                self.error("No offset", f"{t.name} has no offset yet. Type one into the Offset column first.")
            self.project.accept(row, want)
            self._plan_stale()
        elif col == 3:
            text = item.text().strip().replace("ms", "").strip()
            try:
                ms = float(text)
            except ValueError:
                self.error("Offset", f"'{item.text()}' is not a number of milliseconds.")
            else:
                cur = self.project.takes[row].offset
                if cur is None or abs(cur * 1000 - ms) > 1e-6:
                    self.project.set_manual_offset(row, ms / 1000.0)
                    r = self.project.takes[row].result
                    self.say(f"{self.project.takes[row].name}: manual offset {ms:+.3f} ms, "
                             f"audio agreement score {r.score:.2f}" +
                             ("".join("\n  WARNING: " + w for w in r.warnings)))
                    self._plan_stale()
        self._refresh_all()

    def _reset_offset(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        for r in rows:
            self.project.set_manual_offset(r, None)
        self._plan_stale()
        self._refresh_all()

    def _plan_stale(self):
        if self.plan is not None:
            self.say("Takes changed: the cut list was kept and re-checked. Click Update Preview to see it.")

    # ================================================================ plan + preview
    def _excerpt(self):
        try:
            a, b = parse_time(self.ex_start.text()), parse_time(self.ex_end.text())
        except ValueError:
            raise ValueError("Excerpt start/end must look like 1:23.456 or 83.456")
        if b <= a:
            raise ValueError("Excerpt end must be after start.")
        return a, b

    def _settings(self, preview: bool) -> RenderSettings:
        s = RenderSettings.preview() if preview else RenderSettings()
        s.crop_center = self.crop.value() / 100.0
        return s

    def _ready(self) -> bool:
        if not self.project.analyzed:
            self.error("Not analyzed", "Run Analyze first.")
            return False
        if not self.project.usable_map():
            self.error("No usable takes", "No take has a confirmed offset. Check the Warnings column.")
            return False
        return True

    def _generate_preview(self):
        if not self._ready():
            return
        try:
            a, b = self._excerpt()
            pacing = "fast" if self.pacing.currentIndex() == 1 else "natural"
            self.plan = self.project.make_plan(a, b, pacing, self.seed.value())
        except ValueError as e:
            self.error("Cannot plan cuts", str(e))
            return
        for w in self.plan.warnings:
            self.say("Plan: " + w)
        self._refresh_cuts()
        self._render_preview()

    def _update_preview(self):
        if self.plan is None:
            self._generate_preview()
            return
        self._render_preview()

    def _render_preview(self):
        errs = self.project.check_plan(self.plan)
        if errs:
            self.error("Cut list problems", "\n".join(errs))
            return
        if self.player:
            self.player.stop()
            self.player.setSource(QUrl())  # release the old file (Windows locks it)
        self.preview_count += 1
        out = os.path.join(self.preview_dir, f"preview_{self.preview_count}.mp4")
        plan = self.plan
        settings = self._settings(preview=True)

        def work(cancel, emit):
            return self.project.export(plan, out, settings, cancel=cancel, validate=False,
                                       progress=lambda d, t: emit("Rendering preview", d, t))

        def done(_):
            old = self.preview_path
            self.preview_path = out
            if old and os.path.exists(old):
                try:
                    os.remove(old)
                except OSError:
                    pass
            self.say(f"Preview ready ({plan.duration:.1f} s, {len(plan.segments)} cuts).")
            if self.player:
                self.player.setSource(QUrl.fromLocalFile(out))
                self.player.play()
        self.run_task("Rendering preview", work, done)

    def _play_pause(self):
        if not self.player or not self.preview_path:
            return
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _open_external(self):
        if self.preview_path and os.path.exists(self.preview_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.preview_path))
        else:
            self.error("No preview", "Generate a preview first.")

    def _cut_clicked(self, row, col):
        if self.player and self.plan and self.preview_path and row < len(self.plan.segments):
            seg = self.plan.segments[row]
            self.player.setPosition(int(seg.start / float(self.plan.fps) * 1000))

    # ================================================================ cut list edits
    def _cut_item_changed(self, item):
        if self._updating or self.plan is None or item.column() != 0:
            return
        row = item.row()
        segs = self.plan.segments
        try:
            t = parse_time(item.text())
        except ValueError:
            self.error("Cut list", f"'{item.text()}' is not a time like 1:23.456")
            self._refresh_cuts()
            return
        if row == 0:
            self.say("The first segment always starts at the excerpt start; change Start above instead.")
        else:
            j = self.plan.frame_at(t)
            lo, hi = segs[row - 1].start + 1, segs[row].end - 1
            if not lo <= j <= hi:
                self.error("Cut list", f"That cut must lie between {fmt_time(self.plan.frame_time(lo))} and "
                                       f"{fmt_time(self.plan.frame_time(hi))}.")
            else:
                segs[row - 1].end = segs[row].start = j
        self._refresh_cuts()

    def _cut_take_changed(self, row, combo):
        if self._updating or self.plan is None:
            return
        self.plan.segments[row].take = int(combo.currentData())
        self._refresh_cuts()

    def _split_cut(self):
        if self.plan is None:
            return
        rows = sorted({i.row() for i in self.cuts.selectedIndexes()})
        if not rows:
            return
        r = rows[0]
        s = self.plan.segments[r]
        if s.end - s.start < 4:
            return
        mid = (s.start + s.end) // 2
        others = [k for k in self.project.usable_map() if k != s.take] or [s.take]
        self.plan.segments.insert(r + 1, Segment(mid, s.end, others[0]))
        s.end = mid
        self._refresh_cuts()

    def _merge_cut(self):
        if self.plan is None:
            return
        rows = sorted({i.row() for i in self.cuts.selectedIndexes()})
        if not rows or rows[0] == 0:
            return
        r = rows[0]
        self.plan.segments[r - 1].end = self.plan.segments[r].end
        del self.plan.segments[r]
        self._refresh_cuts()

    # ================================================================ export
    def _export(self):
        if not self._ready():
            return
        if self.plan is None:
            self.error("No cut list", "Generate a preview first so you can check the cuts.")
            return
        errs = self.project.check_plan(self.plan)
        if errs:
            self.error("Cut list problems", "\n".join(errs))
            return
        first = Path(self.project.takes[0].info.path)
        default = str(first.with_name(first.stem + "_TakeSync.mp4"))
        path, _ = QFileDialog.getSaveFileName(self, "Export video", default, "MP4 video (*.mp4)",
                                              options=QFileDialog.DontConfirmOverwrite)
        if not path:
            return
        if not path.lower().endswith(".mp4"):
            path += ".mp4"
        inputs = {os.path.normcase(os.path.abspath(t.info.path)) for t in self.project.takes}
        if self.project.ref_wav:
            inputs.add(os.path.normcase(os.path.abspath(self.project.ref_wav.path)))
        if os.path.normcase(os.path.abspath(path)) in inputs:
            self.error("Export", "That is one of your input files. Choose another name.")
            return
        if os.path.exists(path):
            if QMessageBox.question(self, "Overwrite?", f"{os.path.basename(path)} already exists.\nReplace it?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
        plan = self.plan
        settings = self._settings(preview=False)

        def work(cancel, emit):
            return self.project.export(plan, path, settings, cancel=cancel,
                                       progress=lambda d, t: emit("Exporting (then validating)", d, t))

        def done(checks):
            rep = report(checks)
            self.say(f"\nExported {path}\n{rep}")
            if passed(checks):
                QMessageBox.information(self, "Export finished", f"Saved:\n{path}\n\nValidation passed. "
                                        "Please still watch it once for lip-sync and cut quality.")
            else:
                QMessageBox.warning(self, "Export finished with problems", f"Saved:\n{path}\n\n{rep}")
        self.run_task("Exporting", work, done)

    # ================================================================ refresh
    def _refresh_all(self):
        self._updating = True
        try:
            self._refresh_takes()
            self._refresh_ref()
            ex = self.project.auto_excerpt() if self.project.analyzed else None
            self.overlap_label.setText(f"{fmt_time(ex[0])} - {fmt_time(ex[1])}  ({ex[1] - ex[0]:.2f} s)" if ex
                                       else ("none - set Start/End manually" if self.project.analyzed else "-"))
        finally:
            self._updating = False
        self._refresh_cuts()

    def _refresh_ref(self):
        c = self.ref_combo
        c.clear()
        p = self.project
        for i, t in enumerate(p.takes):
            c.addItem(f"Take {i + 1}: {t.name}" + ("  (auto-picked)" if i == p.auto_reference_take() else ""), i)
        if p.ref_wav is not None:
            c.addItem(f"Audio file: {p.ref_wav.name}", "wav")
            c.setCurrentIndex(c.count() - 1)
            self.ref_note.setText("Using the exported song as reference: most reliable sync, clean soundtrack.")
        else:
            c.setCurrentIndex(p.ref_take if p.takes else -1)
            self.ref_note.setText("Using a take as reference: its FL playback audio becomes the timeline and the "
                                  "soundtrack. Needs clean playback audio in that take; an exported WAV of the "
                                  "song is more robust.")

    def _refresh_takes(self):
        p = self.project
        tb = self.table
        tb.setRowCount(len(p.takes))
        for i, t in enumerate(p.takes):
            r = t.result
            iv = p.usable(i) if p.analyzed else None
            use = QTableWidgetItem()
            use.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            use.setCheckState(Qt.Checked if (t.enabled and p.analyzed) else Qt.Unchecked)
            tb.setItem(i, 0, use)
            tb.setItem(i, 1, self._ro(t.name, t.info.path))
            info = t.info
            tb.setItem(i, 2, self._ro(f"{info.width}x{info.height} {float(info.fps or 0):.2f}fps "
                                      f"{info.video_duration:.1f}s" + ("" if info.audio else " NO AUDIO")))
            off = QTableWidgetItem("" if t.offset is None else f"{t.offset * 1000:+.2f}")
            off.setToolTip("song time = file time + offset. Type a value (ms) to correct it.")
            tb.setItem(i, 3, off)
            status = "-" if r is None else (r.status + (" (manual)" if t.manual_offset is not None else ""))
            if self.project.ref_wav is None and i == p.ref_take and r is None:
                status = "reference"
            st = self._ro(status.upper())
            colour = STATUS_COLOURS.get(r.status if r else "", None)
            if colour:
                st.setForeground(QBrush(QColor(colour)))
            tb.setItem(i, 4, st)
            match = "" if r is None or r.status == "reference" else f"{r.score:.2f} / amb {r.ambiguity:.2f}"
            tb.setItem(i, 5, self._ro(match))
            tb.setItem(i, 6, self._ro(f"{fmt_time(iv[0])} - {fmt_time(iv[1])}" if iv else ""))
            warns = list(info.warnings) + (r.warnings if r else [])
            w = self._ro(" | ".join(warns))
            w.setToolTip("\n".join(warns))
            tb.setItem(i, 7, w)

    def _refresh_cuts(self):
        self._updating = True
        try:
            plan = self.plan
            tb = self.cuts
            if plan is None:
                tb.setRowCount(0)
                self.plan_status.setText("")
                return
            usable = self.project.usable_map()
            tb.setRowCount(len(plan.segments))
            for i, s in enumerate(plan.segments):
                start = QTableWidgetItem(fmt_time(plan.frame_time(s.start)))
                if i == 0:
                    start.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                tb.setItem(i, 0, start)
                tb.setItem(i, 1, self._ro(fmt_time(plan.frame_time(s.end))))
                tb.setItem(i, 2, self._ro(f"{(s.end - s.start) / float(plan.fps):.2f} s"))
                combo = QComboBox()
                for k in sorted(set(usable) | {s.take}):
                    combo.addItem(f"Take {k + 1}: {self.project.takes[k].name}", k)
                combo.setCurrentIndex(combo.findData(s.take))
                combo.currentIndexChanged.connect(lambda _, row=i, cb=combo: self._cut_take_changed(row, cb))
                tb.setCellWidget(i, 3, combo)
            errs = self.project.check_plan(plan)
            if errs:
                self.plan_status.setStyleSheet("color: #c62828")
                self.plan_status.setText("Cut list problems (export disabled):\n" + "\n".join(errs))
            else:
                self.plan_status.setStyleSheet("color: #2e7d32")
                self.plan_status.setText(f"{len(plan.segments)} segments, {plan.duration:.2f} s at "
                                         f"{float(plan.fps):.3f} fps. Cut list OK.")
            self.btn_export.setEnabled(not errs and not self.busy)
        finally:
            self._updating = False

    @staticmethod
    def _ro(text, tip=None):
        it = QTableWidgetItem(text)
        it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        if tip:
            it.setToolTip(tip)
        return it

    def closeEvent(self, e):
        if self.busy:
            if QMessageBox.question(self, "Quit?", "A job is running. Cancel it and quit?") != QMessageBox.Yes:
                e.ignore()
                return
            self.cancel_event.set()
        if self.player:
            self.player.stop()
            self.player.setSource(QUrl())
        shutil.rmtree(self.preview_dir, ignore_errors=True)
        e.accept()


def main(argv=None) -> int:
    app = QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("TakeSync")
    win = MainWindow()
    try:
        find_tool("ffmpeg")
        find_tool("ffprobe")
    except MediaError as e:
        win.show()
        QMessageBox.critical(win, "FFmpeg missing", str(e))
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
