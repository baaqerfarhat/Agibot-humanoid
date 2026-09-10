"""Collect fresh healthy LIBERO FIR data, then identify M at a separate initial state."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--control', required=True)
    p.add_argument('--ack', required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    args = p.parse_args()
    if args.out_dir.exists():
        p.error('out directory exists; preserve prior or incomplete calibration')
    args.out_dir.mkdir(parents=True)
    code = Path(__file__).resolve().parent
    commands = [
        [sys.executable, str(code/'error_signal.py'), '--suite', args.suite,
         '--out', str(args.out_dir/'fir.json'), '--control', args.control, '--ack', args.ack,
         '--host', args.host, '--port', str(args.port), '--init-base', '26',
         '--episodes', '10', '--healthy-only'],
        [sys.executable, str(code/'openloop_id.py'), '--suite', args.suite,
         '--log', str(args.out_dir/'fir.json'), '--out', str(args.out_dir/'M.json'),
         '--source-episode', '0', '--probe-task', '0', '--probe-init', '25',
         '--steps', '80', '--probe', '0.02']]
    from run_joint_followup import atomic_json
    record = dict(status='running', argv=commands, completed_commands=0)
    atomic_json(args.out_dir/'pipeline.json', record)
    try:
        for command in commands:
            subprocess.run(command, check=True)
            record['completed_commands'] += 1
            atomic_json(args.out_dir/'pipeline.json', record)
        record['status'] = 'complete'
    except Exception as exc:
        record.update(status='failed', error=repr(exc))
        raise
    finally:
        atomic_json(args.out_dir/'pipeline.json', record)


if __name__ == '__main__':
    main()
