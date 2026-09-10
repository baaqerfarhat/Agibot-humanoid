"""Run an explicit experiment manifest with bounded parallelism, without a shell.

Each job has id, argv, log, and optional cwd/env. Existing logs/status are never
overwritten. This launches the declared commands; it does not select treatments,
resume partial studies, or interpret successful process exits as valid results.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from run_joint_followup import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--status", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=3)
    args = p.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    jobs = manifest["jobs"]
    if args.jobs < 1 or not jobs:
        p.error("positive worker count and nonempty jobs required")
    if len({j['id'] for j in jobs}) != len(jobs) or len({j['log'] for j in jobs}) != len(jobs):
        p.error("job ids and log paths must be unique")
    if args.status.exists() or any(Path(j['log']).exists() for j in jobs):
        p.error("status/log already exists; partial runs are not automatically resumed")
    for job in jobs:
        if not isinstance(job['argv'], list) or not job['argv'] or not all(isinstance(x, str) for x in job['argv']):
            p.error("each argv must be a nonempty string list")
        Path(job['log']).parent.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    status = dict(manifest=str(args.manifest.resolve()),
                  manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                  started_unix=time.time(), workers=args.jobs, status="running", jobs={})
    atomic_json(args.status, status)

    def run(job):
        start = time.time()
        env = dict(os.environ, **manifest.get('env', {}), **job.get('env', {}))
        with Path(job['log']).open('x') as log:
            try:
                result = subprocess.run(job['argv'], cwd=job.get('cwd'), env=env,
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                return dict(returncode=result.returncode, started_unix=start,
                            finished_unix=time.time(), log=job['log'])
            except Exception as exc:
                return dict(returncode=None, error=repr(exc), started_unix=start,
                            finished_unix=time.time(), log=job['log'])

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {pool.submit(run, job): job['id'] for job in jobs}
        for future in as_completed(pending):
            identifier = pending[future]
            status['jobs'][identifier] = future.result()
            atomic_json(args.status, status)
            print(identifier, status['jobs'][identifier], flush=True)
    status['status'] = ('complete' if all(j.get('returncode') == 0 for j in status['jobs'].values()) else 'failed')
    status['finished_unix'] = time.time()
    atomic_json(args.status, status)
    return 0 if status['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
