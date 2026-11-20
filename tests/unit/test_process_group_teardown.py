"""Teardown of a gloo process group through tests.helpers."""

import subprocess
import sys
import textwrap

from tests.helpers import REPO_ROOT

_TEARDOWN_SCRIPT = textwrap.dedent("""
    import sys
    import time

    import torch
    import torch.distributed

    from tests.helpers import destroy_process_group

    torch.distributed.init_process_group(
        "gloo", init_method=sys.argv[1], rank=0, world_size=1
    )
    # From here the main thread keeps the GIL until it blocks. The gloo
    # worker that ran the all_reduce then waits for the GIL inside the
    # work's destructor, to free a tensor whose Python object is gone.
    sys.setswitchinterval(1000.0)
    tensor = torch.ones(1024)
    work = torch.distributed.all_reduce(tensor, async_op=True)
    del tensor, work
    deadline = time.perf_counter() + 0.5
    while time.perf_counter() < deadline:
        pass
    destroy_process_group()
    """)


def test_destroy_process_group_joins_a_gloo_worker_waiting_for_the_gil(tmp_path):
    subprocess.run(
        [sys.executable, "-c", _TEARDOWN_SCRIPT, (tmp_path / "rendezvous").as_uri()],
        check=True,
        timeout=60,
        cwd=REPO_ROOT,
    )
