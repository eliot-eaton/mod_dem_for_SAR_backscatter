#!/usr/bin/env python3
"""
Robust batch GAMMA runner with concurrent multi-process execution.

Each run ID is executed as a completely fresh Python subprocess running
``gamma_processing.py``.  The parent uses a small ThreadPool only to supervise
those independent subprocesses; the actual GAMMA work is performed by the
separate processes.  This avoids persistent py_gamma state between IDs.

Failed jobs can be retried automatically after the parallel pass.  By default
one retry is performed serially, which is useful for transient failures caused
by CPU/RAM/I/O pressure when several GAMMA programs run at once.

Examples
--------
Five concurrent jobs, then retry any failures serially::

    python gamma_batch.py INPUT OUTPUT MLI_PAR 0009-0197 -j 5

More conservative concurrent run::

    python gamma_batch.py INPUT OUTPUT MLI_PAR 0009-0197 -j 3

Disable automatic retry::

    python gamma_batch.py INPUT OUTPUT MLI_PAR 0009-0197 -j 5 --retries 0

Allow two OpenMP threads inside each GAMMA subprocess::

    python gamma_batch.py INPUT OUTPUT MLI_PAR 0009-0197 -j 3 --gamma-threads 2

Existing final TIFFs are skipped unless ``--overwrite`` is supplied.
After a job succeeds, its input ``P.{ID}.dem`` is deleted by default; use
``--keep-input-dem`` to retain it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        else:
            seen.add(item)
            unique.append(item)
    return unique, duplicates


def _build_processing_command(
    processing_script,
    input_dir,
    output_dir,
    mli_par,
    run_id,
    lat_ovr,
    lon_ovr,
):
    return [
        sys.executable,
        str(processing_script),
        str(input_dir),
        str(run_id),
        str(mli_par),
        "--output-dir",
        str(output_dir),
        "--lat-ovr",
        str(lat_ovr),
        "--lon-ovr",
        str(lon_ovr),
    ]


def _run_subprocess_job(job):
    (
        processing_script,
        input_dir,
        output_dir,
        mli_par,
        run_id,
        lat_ovr,
        lon_ovr,
        log_path,
        gamma_threads,
        attempt,
    ) = job

    started = time.perf_counter()
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = _build_processing_command(
        processing_script,
        input_dir,
        output_dir,
        mli_par,
        run_id,
        lat_ovr,
        lon_ovr,
    )

    env = os.environ.copy()
    if gamma_threads is not None:
        # GAMMA releases contain internally parallelised programs.  When
        # several independent jobs are run at once, limiting OpenMP-style
        # thread pools helps avoid CPU/RAM oversubscription.  Variables that
        # are irrelevant to a particular GAMMA build are simply ignored.
        thread_value = str(gamma_threads)
        env["OMP_NUM_THREADS"] = thread_value
        env["OPENBLAS_NUM_THREADS"] = thread_value
        env["MKL_NUM_THREADS"] = thread_value
        env["NUMEXPR_NUM_THREADS"] = thread_value

    with log_path.open("w") as log_file:
        log_file.write(
            f"[BATCH] run_id={run_id} attempt={attempt} "
            f"gamma_threads={gamma_threads}\n"
        )
        log_file.write("[BATCH] command: " + " ".join(command) + "\n\n")
        log_file.flush()

        completed = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )

    elapsed_s = time.perf_counter() - started

    if completed.returncode == 0:
        return {
            "run_id": run_id,
            "status": "successful",
            "error": None,
            "elapsed_s": elapsed_s,
            "log_path": str(log_path),
            "attempt": attempt,
        }

    return {
        "run_id": run_id,
        "status": "failed",
        "error": f"gamma_processing.py exited with code {completed.returncode}",
        "elapsed_s": elapsed_s,
        "log_path": str(log_path),
        "attempt": attempt,
    }


def _run_pass(
    run_ids,
    *,
    workers,
    attempt,
    processing_script,
    input_dir,
    output_dir,
    mli_par,
    lat_ovr,
    lon_ovr,
    gamma_threads,
    logs_dir,
):
    """Run one pass and return (passed_ids, failed_results)."""
    passed = []
    failed_results = []

    jobs = []
    for run_id in run_ids:
        suffix = "" if attempt == 1 else f".retry{attempt - 1}"
        log_path = logs_dir / f"gamma_{run_id}{suffix}.log"
        jobs.append(
            (
                str(processing_script),
                str(input_dir),
                str(output_dir),
                str(mli_par),
                run_id,
                lat_ovr,
                lon_ovr,
                str(log_path),
                gamma_threads,
                attempt,
            )
        )

    effective_workers = min(workers, len(jobs))
    pass_label = "parallel pass" if attempt == 1 else f"retry {attempt - 1}"

    print(
        f"\n[RUN] {pass_label}: {len(jobs)} job(s), "
        f"{effective_workers} worker(s)"
    )

    if not jobs:
        return passed, failed_results

    # Threads only supervise subprocess.run(); GAMMA itself executes in the
    # independent Python/GAMMA subprocesses created above.
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_to_id = {
            executor.submit(_run_subprocess_job, job): job[4]
            for job in jobs
        }

        completed_count = 0
        for future in as_completed(future_to_id):
            completed_count += 1
            run_id = future_to_id[future]

            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "run_id": run_id,
                    "status": "failed",
                    "error": f"batch supervisor error: {type(exc).__name__}: {exc}",
                    "elapsed_s": 0.0,
                    "log_path": None,
                    "attempt": attempt,
                }

            if result["status"] == "successful":
                passed.append(run_id)
                print(
                    f"[PASS {completed_count:>4}/{len(jobs)}] "
                    f"{run_id}  {result['elapsed_s'] / 60.0:.2f} min"
                )
            else:
                failed_results.append(result)
                print(
                    f"[FAIL {completed_count:>4}/{len(jobs)}] "
                    f"{run_id}: {result['error']}"
                )
                if result.get("log_path"):
                    print(f"       log: {result['log_path']}")

    return passed, failed_results


def _delete_successful_dems(input_dir, run_ids):
    """Delete P.{ID}.dem only for jobs that completed successfully."""
    deleted = []
    delete_errors = []

    for run_id in run_ids:
        dem_path = Path(input_dir) / f"P.{run_id}.dem"
        try:
            dem_path.unlink()
        except FileNotFoundError:
            # If it is already gone, do not turn a successful GAMMA run into
            # a failure. Record it as a warning instead.
            delete_errors.append((run_id, f"already missing: {dem_path}"))
        except OSError as exc:
            delete_errors.append((run_id, f"{type(exc).__name__}: {exc}"))
        else:
            deleted.append(run_id)
            print(f"[DELETE] {run_id}: removed input DEM {dem_path}")

    for run_id, message in delete_errors:
        print(f"[WARN] {run_id}: could not delete input DEM: {message}")

    return deleted, delete_errors


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
    retries=1,
    retry_workers=1,
    gamma_threads=1,
    delete_input_dem=True,
):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    mli_par = Path(mli_par).resolve()
    processing_script = Path(__file__).resolve().with_name("gamma_processing.py")

    if workers < 1:
        raise ValueError("workers must be >= 1")
    if retries < 0:
        raise ValueError("retries must be >= 0")
    if retry_workers < 1:
        raise ValueError("retry_workers must be >= 1")
    if gamma_threads is not None and gamma_threads < 1:
        raise ValueError("gamma_threads must be >= 1")

    output_dir.mkdir(parents=True, exist_ok=True)

    dem_par = input_dir / "P.dem_par"
    if not dem_par.exists():
        raise FileNotFoundError(dem_par)
    if not mli_par.exists():
        raise FileNotFoundError(mli_par)
    if not processing_script.exists():
        raise FileNotFoundError(processing_script)

    if mli_par.suffix.lower() in {".tif", ".tiff"}:
        raise ValueError(
            f"Expected a GAMMA MLI parameter file, not a GeoTIFF: {mli_par}"
        )

    run_ids = expand_run_ids(run_ids)
    run_ids, duplicates = _deduplicate_preserving_order(run_ids)

    successful = []
    skipped = []
    permanently_failed = []
    recovered = []
    deleted_dems = []
    delete_errors = []
    pending = []

    print("\n[SETUP] GAMMA batch")
    print(f"        input DEM directory: {input_dir}")
    print(f"        output directory:    {output_dir}")
    print(f"        MLI parameter file:  {mli_par}")
    print(f"        workers:             {workers}")
    print(f"        retries:             {retries}")
    print(f"        retry workers:       {retry_workers}")
    print(f"        GAMMA threads/job:   {gamma_threads}")
    print(f"        delete input DEMs:   {delete_input_dem}")
    print(f"        IDs:                 {' '.join(run_ids)}")

    if duplicates:
        print(
            "[WARN] Duplicate IDs ignored: "
            + " ".join(dict.fromkeys(duplicates))
        )

    for run_id in run_ids:
        dem_path = input_dir / f"P.{run_id}.dem"
        output_tif = output_dir / f"P.{run_id}.sim_sar.radar.tif"

        if not dem_path.exists():
            print(f"[FAIL] {run_id}: missing DEM: {dem_path}")
            permanently_failed.append(run_id)
            continue

        if output_tif.exists() and not overwrite:
            print(f"[SKIP] {run_id}: existing output: {output_tif}")
            skipped.append(run_id)
            successful.append(run_id)
            continue

        pending.append(run_id)

    if not pending:
        return _print_summary(
            successful,
            skipped,
            permanently_failed,
            recovered,
            0.0,
            0,
            deleted_dems,
            delete_errors,
        )

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    print(f"        worker logs:         {logs_dir}")

    started = time.perf_counter()
    jobs_run = 0

    passed, failures = _run_pass(
        pending,
        workers=workers,
        attempt=1,
        processing_script=processing_script,
        input_dir=input_dir,
        output_dir=output_dir,
        mli_par=mli_par,
        lat_ovr=lat_ovr,
        lon_ovr=lon_ovr,
        gamma_threads=gamma_threads,
        logs_dir=logs_dir,
    )
    jobs_run += len(pending)
    successful.extend(passed)
    if delete_input_dem and passed:
        deleted_now, delete_errors_now = _delete_successful_dems(input_dir, passed)
        deleted_dems.extend(deleted_now)
        delete_errors.extend(delete_errors_now)

    failed_ids = [result["run_id"] for result in failures]

    for retry_index in range(1, retries + 1):
        if not failed_ids:
            break

        print(
            f"\n[RETRY] {len(failed_ids)} failed ID(s) will be retried "
            f"with {retry_workers} worker(s): {' '.join(failed_ids)}"
        )

        retry_passed, retry_failures = _run_pass(
            failed_ids,
            workers=retry_workers,
            attempt=retry_index + 1,
            processing_script=processing_script,
            input_dir=input_dir,
            output_dir=output_dir,
            mli_par=mli_par,
            lat_ovr=lat_ovr,
            lon_ovr=lon_ovr,
            gamma_threads=gamma_threads,
            logs_dir=logs_dir,
        )
        jobs_run += len(failed_ids)

        successful.extend(retry_passed)
        recovered.extend(retry_passed)
        if delete_input_dem and retry_passed:
            deleted_now, delete_errors_now = _delete_successful_dems(
                input_dir, retry_passed
            )
            deleted_dems.extend(deleted_now)
            delete_errors.extend(delete_errors_now)
        failed_ids = [result["run_id"] for result in retry_failures]

    permanently_failed.extend(failed_ids)
    elapsed_s = time.perf_counter() - started

    return _print_summary(
        successful,
        skipped,
        permanently_failed,
        recovered,
        elapsed_s,
        jobs_run,
        deleted_dems,
        delete_errors,
    )


def _print_summary(
    successful,
    skipped,
    failed,
    recovered,
    elapsed_s,
    jobs_run,
    deleted_dems,
    delete_errors,
):
    # Keep deterministic output order without double-counting.
    successful = list(dict.fromkeys(successful))
    skipped = list(dict.fromkeys(skipped))
    failed = list(dict.fromkeys(failed))
    recovered = list(dict.fromkeys(recovered))
    deleted_dems = list(dict.fromkeys(deleted_dems))

    print("\n" + "=" * 72)
    print("[DONE] GAMMA batch summary")
    print("=" * 72)
    print(f"        successful: {len(successful)}")
    print(f"        skipped:    {len(skipped)}")
    print(f"        recovered:  {len(recovered)}")
    print(f"        failed:     {len(failed)}")
    print(f"        DEMs deleted: {len(deleted_dems)}")
    print(f"        delete warnings: {len(delete_errors)}")
    if elapsed_s > 0:
        print(f"        elapsed:    {elapsed_s / 60.0:.2f} min")
        if jobs_run:
            print(f"        attempts:   {jobs_run}")
            print(f"        rate:       {jobs_run / (elapsed_s / 60.0):.2f} attempts/min")
    if recovered:
        print("        retry PASS: " + " ".join(recovered))
    if failed:
        print("        failed IDs: " + " ".join(failed))

    return {
        "successful": successful,
        "skipped": skipped,
        "recovered": recovered,
        "failed": failed,
        "deleted_dems": deleted_dems,
        "delete_errors": delete_errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate sim_sar GeoTIFFs concurrently for P.{ID}.dem files, "
            "with optional serial retry of transient failures."
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
        help="Concurrent GAMMA subprocesses in the first pass (default: 1).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Retry count for failed IDs after the first pass (default: 1).",
    )
    parser.add_argument(
        "--retry-workers",
        type=int,
        default=1,
        help="Concurrent workers during retries (default: 1).",
    )
    parser.add_argument(
        "--gamma-threads",
        type=int,
        default=1,
        help=(
            "OpenMP/BLAS thread limit inherited by each GAMMA job "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--keep-input-dem",
        action="store_true",
        help=(
            "Keep P.{ID}.dem after a successful run. By default this "
            "storage-saving batch deletes the input DEM only after "
            "gamma_processing.py exits successfully."
        ),
    )

    args = parser.parse_args()

    result = run_gamma_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mli_par=args.mli_par,
        run_ids=args.ids,
        overwrite=args.overwrite,
        lat_ovr=args.lat_ovr,
        lon_ovr=args.lon_ovr,
        workers=args.workers,
        retries=args.retries,
        retry_workers=args.retry_workers,
        gamma_threads=args.gamma_threads,
        delete_input_dem=not args.keep_input_dem,
    )

    # Useful for shell pipelines / schedulers.
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
