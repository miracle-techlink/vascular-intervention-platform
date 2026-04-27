"""
Launch sofa_runner.py in miniforge3/envs/sofa and stream stdout lines.
The sofa env has eve + SOFA BeamAdapter; dsa3d env does not.
"""
import json
import subprocess
import threading
import queue
from pathlib import Path
from typing import Generator

SOFA_PYTHON = Path("/home/liuyue/miniforge3/envs/sofa/bin/python3")
RUNNER      = Path(__file__).parent / "sofa_runner.py"
SOFA_BASE   = Path("/home/liuyue/Research/血管介入手术机器人/deps/sofa/SOFA_v23.06.00_Linux")

SOFA_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "SOFA_ROOT": str(SOFA_BASE),
    "PYTHONPATH": str(SOFA_BASE / "plugins/SofaPython3/lib/python3/site-packages"),
    "LD_LIBRARY_PATH": ":".join([
        str(SOFA_BASE / "lib"),
        str(SOFA_BASE / "plugins/SofaPython3/lib"),
        "/home/liuyue/miniforge3/envs/sofa/lib",
    ]),
    "PATH": "/home/liuyue/miniforge3/envs/sofa/bin:/usr/bin:/bin",
}


def run_infer(mesh_dir: Path, max_steps: int = 800,
             threshold_mm: float = 15.0, friction: float = 0.01) -> Generator[dict, None, None]:
    proc = subprocess.Popen(
        [str(SOFA_PYTHON), str(RUNNER), str(mesh_dir),
         str(max_steps), str(threshold_mm), str(friction)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=SOFA_ENV,
        text=True,
    )

    q: queue.Queue = queue.Queue()

    def _read_stderr():
        for line in proc.stderr:
            line = line.strip()
            if line:
                q.put({"log": line})
        q.put(None)  # stderr done

    def _read_stdout():
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("{"):
                try:
                    q.put(json.loads(line))
                except json.JSONDecodeError:
                    pass
        proc.wait()
        q.put("__done__")

    threading.Thread(target=_read_stderr, daemon=True).start()
    threading.Thread(target=_read_stdout, daemon=True).start()

    stderr_done = False
    while True:
        item = q.get()
        if item is None:
            stderr_done = True
            continue
        if item == "__done__":
            break
        yield item
