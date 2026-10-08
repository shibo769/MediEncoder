"""Run the full experiment with durable logs and an automatically refreshed report.

This supervisor keeps Windows awake while computation runs; it does not override
an explicit sleep, lid-close, shutdown, or a user's stop request. It never changes
scientific settings, replaces failed seeds, or retries failed statistical fits.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import html
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def atomic_write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(value, encoding='utf-8')
    os.replace(temp, path)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def render_report(output):
    status = read_json(output / 'status.json')
    config = read_json(output / 'manifest.json').get('config', {})
    dgp = config.get('dgp', {})
    target = status.get('execution_target_reps', config.get('B_requested', 0))
    reserved = status.get('reserved_reps', config.get('B_requested', 0))
    planned = target * len(config.get('n_values', [])) * len(config.get('methods', []))
    try:
        with (output / 'summary.csv').open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        rows = []
    def cell(value):
        if value in (None, ''):
            return '&mdash;'
        try:
            return f'{float(value):.3f}'
        except (ValueError, TypeError):
            return html.escape(str(value))
    by = {(int(r['n']), r['method']): r for r in rows}
    sizes = sorted({int(r['n']) for r in rows})
    labels = {'projection': 'Projection', 'autoencoder': 'Autoencoder', 'vae': 'VAE',
              'mediencoder': 'MediEncoder', 'mediencoder_l3zero': 'MediEncoder (lambda3=0)'}
    main = []
    ablation = []
    for n in sizes:
        for method, name in labels.items():
            if method not in config.get('methods', labels):
                continue
            r = by.get((n, method), {})
            metrics = ''.join('<td>' + cell(r.get(k)) + '</td>' for k in ('SD', 'RMSE', 'CI_Length', 'Coverage'))
            counts = f"{r.get('B_completed', 0)} / {r.get('B_requested', target)}; failed {r.get('B_failed', 0)}"
            main.append(f'<tr><td>{n}</td><td>{name}</td>{metrics}<td>{counts}</td></tr>')
        cells = []
        for metric in ('SD', 'RMSE', 'CI_Length'):
            for method in ('mediencoder_l3zero', 'mediencoder'):
                cells.append('<td>' + cell(by.get((n, method), {}).get(metric)) + '</td>')
        z = by.get((n, 'mediencoder_l3zero'), {})
        t = by.get((n, 'mediencoder'), {})
        ablation.append(f"<tr><td>{n}</td>{''.join(cells)}<td>{z.get('B_completed',0)} / {t.get('B_completed',0)}</td></tr>")
    phase = status.get('phase', 'starting')
    finished = phase in ('complete', 'complete_with_failures', 'finished')
    pending = status.get('pending', planned)
    title = 'Corrected MediEncoder experiment' if finished and pending == 0 else 'Corrected MediEncoder experiment — in progress'
    completed = status.get('completed', 0)
    failed = status.get('failed', 0)
    requested = status.get('requested', planned)
    stamp = html.escape(status.get('updated_at', datetime.now(timezone.utc).isoformat()))
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta http-equiv="refresh" content="60"><title>{title}</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#172033;background:#f7f8fa}}h1{{font-size:28px}}h2{{font-size:22px;margin-top:36px}}table{{border-collapse:collapse;width:100%;background:white;font-variant-numeric:tabular-nums}}th,td{{padding:9px 12px;border-bottom:1px solid #dbe0e7;text-align:right}}th{{background:#eaf0f6}}td:nth-child(2){{text-align:left}}.status{{padding:16px;background:#eaf0f6;border-radius:8px}}.note{{color:#4c5667}}a{{color:#175ca8}}</style>
<h1>{title}</h1><p class="status"><b>{completed:,} valid / {requested:,} planned fits</b>; {failed} failed; {pending:,} pending. Phase: {html.escape(phase)}.<br>Updated {stamp} (UTC). This page refreshes every minute.</p>
<p>p={cell(dgp.get('p'))}, q={cell(dgp.get('q'))}; latent dimensions {cell(dgp.get('bar_p'))}/{cell(dgp.get('bar_q'))}; learned dimensions {cell(config.get('tilde_p'))}/{cell(config.get('tilde_q'))}; {target} replications per size and arm in this execution phase, with {reserved} reserved. Run kind: {html.escape(config.get('run_kind', 'starting'))}. Fixed mechanism; population truth; loss normalization: {html.escape(str(config.get('loss_normalization', 'not recorded')))}; retained stop-gradient; dataset-specific within-fold influence-function confidence intervals.</p>
<p class="note">Partial estimates are provisional. Coverage is conditional on valid completed replications. The two tables reuse the same tuned MediEncoder fits. No manuscript or proof has been changed.</p>
<h2>Main comparison</h2><table><thead><tr><th>n</th><th>Estimator</th><th>SD</th><th>RMSE</th><th>CI length</th><th>Coverage</th><th>Valid / planned; failures</th></tr></thead><tbody>{''.join(main)}</tbody></table>
<h2>Alignment ablation</h2><table><thead><tr><th></th><th colspan="2">SD</th><th colspan="2">RMSE</th><th colspan="2">CI length</th><th>Valid counts</th></tr><tr><th>n</th><th>lambda3=0</th><th>Tuning</th><th>lambda3=0</th><th>Tuning</th><th>lambda3=0</th><th>Tuning</th><th>Zero / tuning</th></tr></thead><tbody>{''.join(ablation)}</tbody></table>
<p><a href="summary.csv">Full summary CSV</a> · <a href="manifest.json">Run configuration and provenance</a> · <a href="status.json">Detailed status</a> · <a href="run.log">Execution log</a></p></html>'''
    atomic_write(output / 'progress.html', page)


def monitor_command(argv=None):
    parser = argparse.ArgumentParser(description="Supervise a simulation run. Additional options are forwarded to the simulation runner.")
    parser.add_argument('--output-dir', type=Path, default=Path('results') / 'formal')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--reps', type=int, default=200)
    parser.add_argument('--target-reps', type=int)
    args, forwarded = parser.parse_known_args(argv)
    if args.reps < 1 or args.workers < 1 or (args.target_reps is not None and not 1 <= args.target_reps <= args.reps):
        parser.error('workers/reps must be positive and target-reps must be between 1 and reps')
    output = args.output_dir.resolve()
    command = [sys.executable, '-u', '-m', 'mediencoder.simulation.runner',
               '--output-dir', str(output), '--workers', str(args.workers), '--device', args.device,
               '--reps', str(args.reps)]
    if args.target_reps is not None:
        command.extend(['--target-reps', str(args.target_reps)])
    return output, command + forwarded


def stop_owned_process_tree(proc):
    """Stop only the live process tree/session launched by this supervisor."""
    if proc.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != 'nt':
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait(timeout=10)


def main(argv=None):
    output, command = monitor_command(argv)
    output.mkdir(parents=True, exist_ok=True)
    keep_awake = os.name == 'nt'
    if keep_awake:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    record = {'supervisor_pid': os.getpid(), 'command': command,
              'started_at': datetime.now(timezone.utc).isoformat()}
    proc = None
    try:
        with (output / 'run.log').open('a', encoding='utf-8', buffering=1) as log:
            log.write('\nSUPERVISOR START ' + record['started_at'] + '\n')
            # An isolated child group lets this supervisor stop its own workers
            # without affecting another experiment or the caller's shell.
            launch = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, **launch)
            record['runner_pid'] = proc.pid
            atomic_write(output / 'supervisor.json', json.dumps(record, indent=2))
            while proc.poll() is None:
                render_report(output)
                time.sleep(15)
            render_report(output)
            record.update(exit_code=proc.returncode, finished_at=datetime.now(timezone.utc).isoformat())
            atomic_write(output / 'supervisor.json', json.dumps(record, indent=2))
            return proc.returncode
    finally:
        try:
            if proc is not None and proc.poll() is None:
                stop_owned_process_tree(proc)
                record.update(exit_code=proc.returncode, interrupted=True,
                              finished_at=datetime.now(timezone.utc).isoformat())
                atomic_write(output / 'supervisor.json', json.dumps(record, indent=2))
                status = read_json(output / 'status.json')
                status.update(phase='interrupted', active_tasks=[], updated_at=record['finished_at'])
                atomic_write(output / 'status.json', json.dumps(status, indent=2))
                render_report(output)
        finally:
            if keep_awake:
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    raise SystemExit(main())
