#!/usr/bin/env python3
"""
The daily pipeline: sync -> categorise -> rebuild events.

One entry point for the whole update, so it can be scheduled and forgotten.
Each stage is an existing script run as a subprocess, which keeps them usable on
their own and means one stage blowing up cannot corrupt another's state.

    1. sync        setlistfm_db.py sync         new + edited setlists from the API
    2. categories  artist_categories.py refresh kworb pull, thresholds, A-E tiers
    3. pollstar    build_events.py load-pollstar box office  (only with --pollstar)
    4. events      build_events.py build        bills, routing, categories, Pollstar
    5. export      build_events.py export       events.csv  (only with --export)

THREE WAYS TO RUN IT
--------------------
    python pipeline.py run
        One pass, now. This is what a scheduler should call.

    python pipeline.py schedule --at 03:00
        Stays running and does a pass at that time every day. Pure stdlib, no
        extra packages. Fine if the machine is always on; if it sleeps or
        reboots, prefer the Task Scheduler route below.

    python pipeline.py install-task --at 03:00
        Prints (and with --yes, runs) the schtasks command that registers this
        as a Windows scheduled task. Survives reboots. Recommended.

Everything is logged to logs/pipeline-YYYY-MM-DD.log as well as the console, and
a lock file stops two runs overlapping -- which matters, because a full events
rebuild takes about six minutes and the API sync has a daily request budget.

Exit code is 0 only if every stage succeeded.
"""

import argparse
import datetime as dt
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
LOCK_FILE = os.path.join(HERE, ".pipeline.lock")
STALE_LOCK_HOURS = 12

# (name, argv, always) -- `always` false means the step needs an explicit flag
STEPS = [
    ("sync", ["setlistfm_db.py", "sync"], True),
    ("categories", ["artist_categories.py", "refresh"], True),
    # opt-in: re-reading the 135MB Pollstar workbook takes ~3 minutes and is
    # only needed when that file changes, not on every run
    ("pollstar", ["build_events.py", "load-pollstar"], False),
    ("events", ["build_events.py", "build"], True),
    ("export", ["build_events.py", "export"], False),
]


def now():
    return dt.datetime.now()


def log(msg, handle=None):
    line = f"[{now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, file=sys.stderr, flush=True)
    if handle:
        handle.write(line + "\n")
        handle.flush()


def open_log():
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"pipeline-{now():%Y-%m-%d}.log")
    return open(path, "a", encoding="utf-8"), path


# ----------------------------------------------------------------------------
# locking
# ----------------------------------------------------------------------------

def acquire_lock(force=False):
    """Refuse to start a second run on top of a first."""
    if os.path.exists(LOCK_FILE):
        age_h = (time.time() - os.path.getmtime(LOCK_FILE)) / 3600
        try:
            with open(LOCK_FILE, encoding="utf-8") as f:
                owner = f.read().strip()
        except OSError:
            owner = "?"
        if age_h > STALE_LOCK_HOURS:
            log(f"!! stale lock from {owner} ({age_h:.1f}h old) - taking it over")
        elif force:
            log(f"!! lock held by {owner} ({age_h:.1f}h old) - overriding (--force)")
        else:
            log(f"!! another run is in progress ({owner}, {age_h:.1f}h old). "
                f"Use --force to override, or delete {LOCK_FILE}.")
            return False
    with open(LOCK_FILE, "w", encoding="utf-8") as f:
        f.write(f"pid {os.getpid()} started {now():%Y-%m-%d %H:%M:%S}")
    return True


def release_lock():
    try:
        os.remove(LOCK_FILE)
    except OSError:
        pass


# ----------------------------------------------------------------------------
# running one pass
# ----------------------------------------------------------------------------

def run_step(name, argv, handle, timeout, extra=()):
    script = os.path.join(HERE, argv[0])
    if not os.path.exists(script):
        log(f"-- {name}: SKIPPED, {argv[0]} not found", handle)
        return "skipped", 0.0
    cmd = [sys.executable, script] + list(argv[1:]) + list(extra)
    log(f"-- {name}: {' '.join(argv)}", handle)
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=HERE, timeout=timeout, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = proc.returncode
    except subprocess.TimeoutExpired:
        log(f"   {name}: TIMED OUT after {timeout}s", handle)
        return "timeout", time.monotonic() - started
    took = time.monotonic() - started

    for line in (proc.stdout or "").splitlines():
        log(f"   | {line}", handle)
    if out == 0:
        log(f"   {name}: ok in {took/60:.1f} min", handle)
        return "ok", took
    log(f"   {name}: FAILED (exit {out}) after {took/60:.1f} min", handle)
    return "failed", took


def one_pass(args):
    handle, path = open_log()
    started = time.monotonic()
    log("=" * 64, handle)
    log(f"pipeline run starting (logging to {os.path.relpath(path, HERE)})", handle)

    if not acquire_lock(args.force):
        handle.close()
        return 2

    results = []
    try:
        for name, argv, always in STEPS:
            if args.only and name not in args.only:
                continue
            if name in args.skip:
                log(f"-- {name}: skipped (--skip)", handle)
                continue
            # an opt-in stage (export) runs when its flag is set, or when the
            # user named it explicitly with --only
            if not always and not getattr(args, name, False) and not args.only:
                continue
            status, took = run_step(name, argv, handle, args.timeout)
            results.append((name, status, took))
            if status in ("failed", "timeout") and not args.keep_going:
                log(f"!! stopping: {name} did not succeed "
                    f"(use --keep-going to carry on regardless)", handle)
                break
    finally:
        release_lock()

    total = time.monotonic() - started
    log("-" * 64, handle)
    for name, status, took in results:
        log(f"   {name:12s} {status:8s} {took/60:6.1f} min", handle)
    bad = [n for n, s, _ in results if s in ("failed", "timeout")]
    log(f"pipeline {'FAILED' if bad else 'complete'} in {total/60:.1f} min"
        + (f" - problems in: {', '.join(bad)}" if bad else ""), handle)
    log("=" * 64, handle)
    handle.close()
    return 1 if bad else 0


# ----------------------------------------------------------------------------
# scheduling
# ----------------------------------------------------------------------------

def parse_hhmm(value):
    try:
        hh, mm = value.split(":")
        return dt.time(int(hh), int(mm))
    except (ValueError, TypeError):
        raise SystemExit(f"--at wants HH:MM, got {value!r}")


def next_run(at):
    target = dt.datetime.combine(now().date(), at)
    if target <= now():
        target += dt.timedelta(days=1)
    return target


def cmd_schedule(args):
    at = parse_hhmm(args.at)
    log(f"scheduler started - a pass every day at {args.at}. Ctrl-C to stop.")
    if args.now:
        one_pass(args)
    while True:
        nxt = next_run(at)
        wait = (nxt - now()).total_seconds()
        log(f"next run {nxt:%Y-%m-%d %H:%M} (in {wait/3600:.1f}h)")
        try:
            # wake up hourly so a laptop suspend or clock change can't overshoot
            while True:
                slice_ = min(3600, (nxt - now()).total_seconds())
                if slice_ <= 0:
                    break
                time.sleep(slice_)
        except KeyboardInterrupt:
            log("scheduler stopped")
            return 0
        one_pass(args)


def cmd_install_task(args):
    """
    Register this pipeline with the Windows Task Scheduler.

    schtasks takes the whole command as one /TR string, and nesting quotes
    inside it (for the interpreter path, the script path AND a cd) misparses.
    So we write a one-line .bat wrapper and point the task at that instead --
    which also gives you something you can double-click to run on demand.
    """
    at = parse_hhmm(args.at)
    bat = os.path.join(HERE, "run_pipeline.bat")
    # newline="" plus explicit CRLF: .bat files want Windows line endings.
    with open(bat, "w", encoding="utf-8", newline="") as f:
        # %~dp0 is the folder the .bat itself sits in, so moving the project
        # does not break the scheduled task the way a baked-in path would
        for line in ("@echo off",
                     'cd /d "%~dp0"',
                     f'"{sys.executable}" "%~dp0pipeline.py" run %*'):
            f.write(line + chr(13) + chr(10))
    log(f"wrote {bat}")

    schtasks = ["schtasks", "/Create", "/SC", "DAILY", "/ST", f"{at:%H:%M}",
                "/TN", args.name, "/TR", bat, "/F"]
    log("This will register a daily task:")
    log("   " + " ".join(f'"{a}"' if " " in a else a for a in schtasks))
    if not args.yes:
        log("")
        log("Nothing has been scheduled. Re-run with --yes to register it, or "
            "copy the command above and run it yourself in an elevated prompt.")
        return 0
    proc = subprocess.run(schtasks, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in (proc.stdout or "").splitlines():
        log("   | " + line)
    if proc.returncode == 0:
        log(f'registered. Inspect it with:  schtasks /Query /TN "{args.name}"')
        log(f'remove it with:               schtasks /Delete /TN "{args.name}" /F')
    else:
        log(f"!! schtasks failed (exit {proc.returncode}). Creating a daily task "
            f"usually needs an elevated prompt.")
    return proc.returncode


# ----------------------------------------------------------------------------

def add_run_flags(p):
    p.add_argument("--export", action="store_true",
                   help="Also write events.csv at the end.")
    p.add_argument("--pollstar", action="store_true",
                   help="Also reload pollstar-data.xlsx (~3 min). Only needed "
                        "when that file has changed.")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=[s[0] for s in STEPS], help="Stages to leave out.")
    p.add_argument("--only", nargs="*", default=None,
                   choices=[s[0] for s in STEPS], help="Run only these stages.")
    p.add_argument("--keep-going", action="store_true",
                   help="Carry on after a stage fails.")
    p.add_argument("--force", action="store_true",
                   help="Start even if a lock file says a run is in progress.")
    p.add_argument("--timeout", type=int, default=6 * 3600,
                   help="Per-stage timeout in seconds (default 6h).")


def main():
    p = argparse.ArgumentParser(
        description="Run or schedule the setlist.fm -> categories -> events update.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="One pass, now. Point your scheduler at this.")
    add_run_flags(r)
    r.set_defaults(func=lambda a: one_pass(a))

    s = sub.add_parser("schedule", help="Stay running; one pass a day at --at.")
    s.add_argument("--at", default="03:00", help="Local time HH:MM (default 03:00).")
    s.add_argument("--now", action="store_true", help="Also run once on startup.")
    add_run_flags(s)
    s.set_defaults(func=cmd_schedule)

    i = sub.add_parser("install-task",
                       help="Register a daily Windows scheduled task (survives reboots).")
    i.add_argument("--at", default="03:00")
    i.add_argument("--name", default="jambase daily update")
    i.add_argument("--yes", action="store_true",
                   help="Actually register it. Without this, the command is only shown.")
    i.set_defaults(func=cmd_install_task)

    args = p.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
