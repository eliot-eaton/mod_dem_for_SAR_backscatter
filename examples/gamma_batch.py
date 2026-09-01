#!/usr/bin/env python3
"""
Standalone batch GAMMA runner with multiprocessing support.

Runs in the fixed GAMMA/py_gamma environment and intentionally does not
import the installed toposhapes_sar package.

Examples
--------
Serial (same behaviour as the original script):

    python gamma_batch.py \
        ./mod_dem/synthetic_sweep \
        ./sim_sar \
        ./slcs/20201226M/20201226.mli.par \
        0001-0100

Four concurrent GAMMA jobs:

    python gamma_batch.py \
        ./mod_dem/synthetic_sweep \
        ./sim_sar \
        ./slcs/20201226M/20201226.mli.par \
        0001-0100 \
        -j 4

Existing P.{ID}.sim_sar.radar.tif outputs are skipped unless --overwrite
is supplied.

When -j/--workers is greater than 1, each run writes console output to:

    OUTPUT_DIR/logs/gamma_{ID}.log

This keeps parallel GAMMA output from becoming interleaved on the terminal.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from multiprocessing import get_context
from pathlib import Path


def expand_run_ids(run_ids):
    """Expand run IDs and zero-padded ranges."""
    if isinstance(run_ids, (str, int)):
        run_ids = [run_ids]

    expanded = []

    for item in run_ids:
        item = str(item)

        if "-" in item:
            start_str, end_str = item.split("-", 1)
            start = int(start_str)
            end = int(end_str)

            if end < start:
                raise ValueError(
                    f"Invalid run-ID range '{item}': end must be >= start."
                )

            pad_width = max(len(start_str), len(end_str))
            expanded.extend(
                f"{run_id:0{pad_width}d}"
                for run_id in range(start, end + 1)
            )
        else:
            expanded.append(item)

    return expanded


def _deduplicate_preserving_order(items):
    seen = set()
    unique = []
    duplicates = []

    for item in items:
        if item in seen:
            duplicates.append(item)
            continue
        seen.add(item)
        unique.append(item)

    return unique, duplicates


@contextmanager
def _redirect_process_output(log_path):
    """
    Redirect this worker process's stdout/stderr file descriptors to a log.

    File-descriptor redirection also catches output from GAMMA child
    processes that inherit stdout/stderr.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    sys.stdout.flush()
    sys.stderr.flush()

    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)

    try:
        with log_path.open("w", buffering=1) as log_file:
            os.dup2(log_file.fileno(), 1)
            os.dup2(log_file.fileno(), 2)
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def _process_one_job(job):
    """Top-level worker function so it can be pickled by multiprocessing."""
    (
        input_dir,
        output_dir,
        mli_par,
        run_id,
        lat_ovr,
        lon_ovr,
        log_path,
    ) = job

    started = time.perf_counter()

    try:
        # Import inside the worker. This avoids importing py_gamma in the
        # parent process before workers are created.
        from gamma_processing import run_gamma_processing

        kwargs = dict(
            input_dir=input_dir,
            run_id=run_id,
            mli_par_path=mli_par,
            output_dir=output_dir,
            lat_ovr=lat_ovr,
            lon_ovr=lon_ovr,
        )

        if log_path is None:
            run_gamma_processing(**kwargs)
        else:
            with _redirect_process_output(log_path):
                print(f"[WORKER] Starting GAMMA run {run_id}")
                run_gamma_processing(**kwargs)

        return {
            "run_id": run_id,
            "status": "successful",
            "error": None,
            "elapsed_s": time.perf_counter() - started,
            "log_path": log_path,
        }

    except Exception as exc:
        return {
            "run_id": run_id,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": time.perf_counter() - started,
            "log_path": log_path,
        }


def run_gamma_batch(
    input_dir,
    output_dir,
    mli_par,
    run_ids,
    *,
    overwrite=False,
    lat_ovr=1,
    lon_ovr=1,
    workers=1,
):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    mli_par = Path(mli_par).resolve()

    if workers < 1:
        raise ValueError("workers must be >= 1")

    output_dir.mkdir(parents=True, exist_ok=True)

    dem_par = input_dir / "P.dem_par"
    if not dem_par.exists():
        raise FileNotFoundError(dem_par)

    if not mli_par.exists():
        raise FileNotFoundError(mli_par)

    run_ids = expand_run_ids(run_ids)
    run_ids, duplicates = _deduplicate_preserving_order(run_ids)

    successful = []
    skipped = []
    failed = []
    pending = []

    print("\n[SETUP] GAMMA batch")
    print(f"        input DEM directory: {input_dir}")
    print(f"        output directory:    {output_dir}")
    print(f"        MLI parameter file:  {mli_par}")
    print(f"        workers:             {workers}")
    print(f"        IDs:                 {' '.join(run_ids)}")

    if duplicates:
        duplicate_text = " ".join(dict.fromkeys(duplicates))
        print(f"[WARN] Duplicate IDs ignored: {duplicate_text}")

    # Validate/skip in the parent process before any worker is launched.
    for run_id in run_ids:
        dem_path = input_dir / f"P.{run_id}.dem"
        output_tif = output_dir / f"P.{run_id}.sim_sar.radar.tif"

        if not dem_path.exists():
            print(f"[FAIL] {run_id}: missing DEM: {dem_path}")
            failed.append(run_id)
            continue

        if output_tif.exists() and not overwrite:
            print(f"[SKIP] {run_id}: existing output: {output_tif}")
            skipped.append(run_id)
            successful.append(run_id)
            continue

        pending.append(run_id)

    if not pending:
        print("\n[INFO] No GAMMA jobs need to run.")
        return _print_summary(successful, skipped, failed, 0.0, 0)

    started = time.perf_counter()

    # Keep serial mode easy to debug and behaviour close to the original.
    if workers == 1:
        for index, run_id in enumerate(pending, start=1):
            print("\n" + "=" * 72)
            print(f"[GAMMA {run_id}] {index}/{len(pending)}")
            print("=" * 72)

            result = _process_one_job(
                (
                    str(input_dir),
                    str(output_dir),
                    str(mli_par),
                    run_id,
                    lat_ovr,
                    lon_ovr,
                    None,
                )
            )

            if result["status"] == "successful":
                successful.append(run_id)
                print(
                    f"[PASS] {run_id} "
                    f"({result['elapsed_s'] / 60.0:.2f} min)"
                )
            else:
                failed.append(run_id)
                print(f"[FAIL] {run_id}: {result['error']}")

    else:
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        effective_workers = min(workers, len(pending))
        print(
            f"\n[RUN] Processing {len(pending)} job(s) with "
            f"{effective_workers} worker(s)"
        )
        print(f"      worker logs: {logs_dir}")
        print("      multiprocessing start method: spawn")

        jobs = []
        for run_id in pending:
            jobs.append(
                (
                    str(input_dir),
                    str(output_dir),
                    str(mli_par),
                    run_id,
                    lat_ovr,
                    lon_ovr,
                    str(logs_dir / f"gamma_{run_id}.log"),
                )
            )

        # 'spawn' gives every GAMMA worker a clean Python/py_gamma process,
        # avoiding forked library state inherited from the parent.
        mp_context = get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=mp_context,
        ) as executor:
            future_to_id = {
                executor.submit(_process_one_job, job): job[3]
                for job in jobs
            }

            completed = 0
            for future in as_completed(future_to_id):
                completed += 1
                run_id = future_to_id[future]

                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "run_id": run_id,
                        "status": "failed",
                        "error": f"Worker process error: {type(exc).__name__}: {exc}",
                        "elapsed_s": 0.0,
                        "log_path": str(logs_dir / f"gamma_{run_id}.log"),
                    }

                if result["status"] == "successful":
                    successful.append(run_id)
                    print(
                        f"[PASS {completed:>4}/{len(pending)}] "
                        f"{run_id}  {result['elapsed_s'] / 60.0:.2f} min"
                    )
                else:
                    failed.append(run_id)
                    print(
                        f"[FAIL {completed:>4}/{len(pending)}] "
                        f"{run_id}: {result['error']}"
                    )
                    if result.get("log_path"):
                        print(f"       log: {result['log_path']}")

    elapsed_s = time.perf_counter() - started
    return _print_summary(
        successful,
        skipped,
        failed,
        elapsed_s,
        len(pending),
    )


def _print_summary(successful, skipped, failed, elapsed_s, jobs_run):
    print("\n" + "=" * 72)
    print("[DONE] GAMMA batch summary")
    print("=" * 72)
    print(f"        successful: {len(successful)}")
    print(f"        skipped:    {len(skipped)}")
    print(f"        failed:     {len(failed)}")

    if jobs_run and elapsed_s > 0:
        print(f"        wall time:  {elapsed_s / 60.0:.2f} min")
        print(f"        throughput: {jobs_run / (elapsed_s / 60.0):.2f} jobs/min")

    if failed:
        print("        failed IDs: " + " ".join(failed))

    return {
        "successful": successful,
        "skipped": skipped,
        "failed": failed,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate sim_sar GeoTIFFs for a list of P.{ID}.dem files, "
            "optionally using multiple concurrent GAMMA worker processes."
        )
    )
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("mli_par")
    parser.add_argument("ids", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--lat-ovr", type=int, default=1)
    parser.add_argument("--lon-ovr", type=int, default=1)
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of independent GAMMA jobs to run concurrently "
            "(default: 1). Start with 2-4 and benchmark throughput."
        ),
    )

    args = parser.parse_args()

    run_gamma_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mli_par=args.mli_par,
        run_ids=args.ids,
        overwrite=args.overwrite,
        lat_ovr=args.lat_ovr,
        lon_ovr=args.lon_ovr,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
