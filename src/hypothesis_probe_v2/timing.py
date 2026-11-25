"""Wall-clock timing per stage, appended to a JSON-lines log (one row per finished stage)."""
import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm


def fmt(sec):
    m, s = divmod(int(round(sec)), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


class StageTimer:
    """`path`: JSON-lines log (None = keep nothing). `echo`: also print finished stages above the progress bar."""

    def __init__(self, path=None, echo=True):
        self.path = Path(path) if path else None
        self.echo = echo

    def begin(self, name, echo=None, **meta):
        return {"stage": name, "meta": meta, "echo": self.echo if echo is None else echo,
                "start": datetime.now(), "t0": time.perf_counter()}

    def end(self, s, status="ok", **more):
        """Write the row for a stage from begin(); returns its seconds."""
        sec = time.perf_counter() - s["t0"]
        meta = {**s["meta"], **more}
        row = {"stage": s["stage"], **meta, "status": status, "start": s["start"].isoformat(timespec="seconds"),
               "end": datetime.now().isoformat(timespec="seconds"), "seconds": round(sec, 2)}
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if s["echo"]:
            parts = [s["stage"], *(f"{k}={v}" for k, v in meta.items()), fmt(sec)]
            tqdm.write("[time] " + " ".join(parts) + ("" if status == "ok" else f" ({status})"))
        return round(sec, 2)

    @contextmanager
    def stage(self, name, echo=None, **meta):
        """Time the enclosed block; a failure is logged with status "error" and re-raised.
        The block may add fields to the yielded dict (e.g. cached=True)."""
        s = self.begin(name, echo, **meta)
        extra = {}
        try:
            yield extra
        except BaseException:
            self.end(s, "error", **extra)
            raise
        self.end(s, **extra)


NULL = StageTimer(path=None, echo=False)

STAGES = {
    1: "Profiling",
    2: "Hypothesis Generation",
    3: "Hypothesis-based Probing",
    4: "Response Analysis & Verification",
    5: "Hypothesis Update & Strategy Selection",
}
RULE = "=" * 70


def banner(stage, detail="", echo=True):
    """Print a block like
    ======================================================================
    [Stage 3/5] Hypothesis-based Probing | policy=ucb1, round 2/5
    ======================================================================
    above any progress bar."""
    if echo:
        title = f"[Stage {stage}/{len(STAGES)}] {STAGES[stage]}" + (f" | {detail}" if detail else "")
        tqdm.write(f"\n{RULE}\n{title}\n{RULE}")
