"""Drive the real window headless (QT_QPA_PLATFORM=offscreen)."""
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

from takesync import gui  # noqa: E402
from takesync.validate import passed  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialogs(monkeypatch):
    calls = []
    for name in ("critical", "warning", "information"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, _n=name, **k: calls.append((_n, a[1:]))))
    return calls


def wait(app, win, timeout=300):
    t = time.time()
    while win.busy:
        app.processEvents()
        time.sleep(0.01)
        assert time.time() - t < timeout, "task timed out"
    app.processEvents()


def test_gui_full_flow(app, dialogs, media, tmp_path, monkeypatch):
    win = gui.MainWindow()
    win.show()
    win._add_paths([media["ref"], media["pos"], media["neg"]])
    wait(app, win)
    assert win.table.rowCount() == 3
    win._dropped([media["ref"]])          # duplicate is skipped, not added twice
    wait(app, win)
    assert win.table.rowCount() == 3

    win._analyze()
    wait(app, win)
    assert not [c for c in dialogs if c[0] in ("critical", "warning")], dialogs
    assert win.table.item(1, 4).text() == "OK"
    assert abs(float(win.table.item(1, 3).text()) - 7345.6) < 1.0
    assert win.ex_start.text() and win.ex_end.text()

    # shorter excerpt to keep the test quick
    win.ex_start.setText("0:08.000")
    win.ex_end.setText("0:16.000")
    win._generate_preview()
    wait(app, win)
    assert win.preview_path and os.path.exists(win.preview_path)
    assert win.cuts.rowCount() == len(win.plan.segments) >= 2
    win.grab().save(str(tmp_path / "gui.png"))
    if os.environ.get("TAKESYNC_SCREENSHOT"):
        win.grab().save(os.environ["TAKESYNC_SCREENSHOT"])

    # edit a cut boundary and a take choice
    s1, e1 = win.plan.segments[1].start, win.plan.segments[1].end
    assert e1 - s1 > 31
    win.cuts.item(1, 0).setText(gui.fmt_time(win.plan.frame_time(s1 + 30)))
    app.processEvents()
    assert win.plan.segments[1].start == s1 + 30 and win.plan.segments[0].end == s1 + 30
    combo = win.cuts.cellWidget(0, 3)
    other = next(i for i in range(combo.count()) if combo.itemData(i) != win.plan.segments[0].take
                 and combo.itemData(i) != win.plan.segments[1].take)
    combo.setCurrentIndex(other)
    app.processEvents()
    assert win.plan.segments[0].take == combo.itemData(other)
    assert win.project.check_plan(win.plan) == []

    # an out-of-range edit is refused with a visible error
    before = [vars(s).copy() for s in win.plan.segments]
    win.cuts.item(1, 0).setText("9:00.000")
    app.processEvents()
    assert [vars(s) for s in win.plan.segments] == before
    assert dialogs and dialogs[-1][0] == "critical"

    # manual offset correction (ms precision)
    win.table.item(2, 3).setText("-2500.000")
    app.processEvents()
    assert win.project.takes[2].manual_offset == -2.5
    assert "MANUAL" in win.table.item(2, 4).text()

    # export
    out = str(tmp_path / "export.mp4")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (out, "")))
    win._export()
    wait(app, win, timeout=600)
    assert os.path.exists(out)
    assert dialogs[-1][0] == "information", dialogs[-1]
    assert "Validation passed" in dialogs[-1][1][1]

    # refusing the overwrite prompt leaves the file alone
    mtime = os.path.getmtime(out)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    win._export()
    wait(app, win)
    assert os.path.getmtime(out) == mtime

    # exporting over an input file is refused
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (media["ref"], "")))
    n = len(dialogs)
    win._export()
    assert dialogs[n][0] == "critical"

    # cancel a running export
    out2 = str(tmp_path / "cancelled.mp4")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (out2, "")))
    win._export()
    t = time.time()
    while time.time() - t < 0.5:
        app.processEvents()
    win._cancel()
    wait(app, win)
    assert "cancelled" in win.status.text().lower()
    assert not os.path.exists(out2) and not os.path.exists(out2.replace(".mp4", ".partial.mp4"))
    win.close()


def test_gui_flagged_take_needs_confirmation(app, dialogs, media):
    win = gui.MainWindow()
    win._add_paths([media["ref"], media["wrong_song"]])
    wait(app, win)
    win._analyze()
    wait(app, win)
    assert any(c[0] == "warning" for c in dialogs)          # user is told
    assert not win.project.takes[1].enabled                  # and the take is not used
    assert "FAILED" in win.table.item(1, 4).text()
    win.close()
