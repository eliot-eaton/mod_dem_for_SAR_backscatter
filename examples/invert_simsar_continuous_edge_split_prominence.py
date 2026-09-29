#!/usr/bin/env python3
"""
Continuous-edge SimSAR inversion with separate MLI and SimSAR peak-prominence
thresholds.

This script is a thin wrapper around the existing
`invert_simsar_continuous_edge.py` in the same directory. It preserves the
existing inversion, plotting, excavation-gap handling, continuity tracking,
and scoring, but allows different prominence thresholds for:

    --mli-peak-prominence-db
    --simsar-peak-prominence-db

Place this file in the same `examples/` directory as
`invert_simsar_continuous_edge.py`, then run this file instead of the original.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


def load_base_module():
    """
    Load the existing invert_simsar_continuous_edge.py from the same folder.
    """
    this_file = Path(__file__).resolve()
    base_path = this_file.with_name("invert_simsar_continuous_edge.py")

    if not base_path.exists():
        raise FileNotFoundError(
            "Could not find the base script:\n"
            f"  {base_path}\n\n"
            "Put this file in the same examples/ directory as "
            "invert_simsar_continuous_edge.py."
        )

    spec = importlib.util.spec_from_file_location(
        "_invert_simsar_continuous_edge_base",
        base_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import base script: {base_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    base = load_base_module()

    parser = argparse.ArgumentParser(
        description=(
            "Rank modified DEM models using the existing continuous-edge "
            "inversion, with separate MLI and SimSAR peak-prominence "
            "thresholds."
        )
    )

    parser.add_argument(
        "mli_tif",
        type=Path,
    )
    parser.add_argument(
        "simsar_dir",
        type=Path,
    )
    parser.add_argument("id_start")
    parser.add_argument("id_end")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("peak_model_inversion_dense"),
    )
    parser.add_argument(
        "--simsar-pattern",
        default="P.{id}.sim_sar.radar.tif",
        help="Filename pattern containing {id}.",
    )
    parser.add_argument(
        "--median-size",
        type=int,
        default=15,
        help=(
            "Odd 2-D median-filter size used for the observed MLI. "
            "Default: 15."
        ),
    )
    parser.add_argument(
        "--simsar-median-size",
        type=int,
        default=3,
        help=(
            "Odd 2-D median-filter size applied to SimSAR before peak "
            "picking. Default: 3. Use 1 to disable."
        ),
    )
    parser.add_argument(
        "--peak-sigma",
        type=float,
        default=1.5,
        help=(
            "Gaussian smoothing sigma along each range profile, in pixels. "
            "Applied by the existing peak picker. Default: 1.5."
        ),
    )

    # ------------------------------------------------------------------
    # NEW: independent prominence thresholds
    # ------------------------------------------------------------------
    parser.add_argument(
        "--mli-peak-prominence-db",
        type=float,
        default=2.0,
        help=(
            "Minimum peak prominence for observed MLI peaks, in dB. "
            "Default: 2."
        ),
    )
    parser.add_argument(
        "--simsar-peak-prominence-db",
        type=float,
        default=2.0,
        help=(
            "Minimum peak prominence for SimSAR candidate peaks, in dB. "
            "Default: 2."
        ),
    )

    parser.add_argument(
        "--peak-distance-pixels",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--peak-mode",
        choices=("first", "most_prominent"),
        default="first",
    )
    parser.add_argument(
        "--simsar-max-jump-pixels",
        type=float,
        default=4.0,
        help=(
            "Maximum SimSAR peak movement per azimuth row. "
            "Larger jumps are rejected. Default: 4."
        ),
    )
    parser.add_argument(
        "--simsar-continuity-penalty",
        type=float,
        default=0.25,
        help=(
            "Cost per pixel of SimSAR row-to-row peak movement. "
            "Default: 0.25."
        ),
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=1.0,
        help=(
            "Fraction of filtered-MLI-valid rows on which a model must "
            "also produce a peak to be ranked. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--azimuth-min",
        type=int,
        default=base.DEFAULT_INVERSION_ROW_MIN,
        help=(
            "First azimuth row used in inversion. "
            f"Default: {base.DEFAULT_INVERSION_ROW_MIN}."
        ),
    )
    parser.add_argument(
        "--azimuth-max",
        type=int,
        default=base.DEFAULT_INVERSION_ROW_MAX,
        help=(
            "Last azimuth row used in inversion. "
            f"Default: {base.DEFAULT_INVERSION_ROW_MAX}."
        ),
    )
    parser.add_argument(
        "--interaction",
        choices=(
            "auto",
            "excavate_to_lower",
            "subtract_thickness",
            "fill_to_upper",
            "add_thickness",
        ),
        default="auto",
    )
    parser.add_argument(
        "--provenance-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--provenance-pattern",
        default="{id}.json",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--overview-grid-rows",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--overview-grid-cols",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    if "{id}" not in args.simsar_pattern:
        parser.error(
            "--simsar-pattern must contain '{id}'."
        )
    if "{id}" not in args.provenance_pattern:
        parser.error(
            "--provenance-pattern must contain '{id}'."
        )
    if args.mli_peak_prominence_db < 0:
        parser.error(
            "--mli-peak-prominence-db must be >= 0."
        )
    if args.simsar_peak_prominence_db < 0:
        parser.error(
            "--simsar-peak-prominence-db must be >= 0."
        )

    # ------------------------------------------------------------------
    # Keep references to the original functions, then intercept only the
    # prominence value passed to the MLI and SimSAR pickers.
    # ------------------------------------------------------------------

    original_pick_dense_profile_set = base.pick_dense_profile_set
    original_pick_continuous_simsar_edge = (
        base.pick_continuous_simsar_edge
    )

    def pick_dense_profile_set_with_mli_prominence(
        profiles_db,
        *,
        peak_sigma,
        prominence_db,
        min_distance_pixels,
        peak_mode,
        respect_internal_nodata_gaps=False,
        allow_boundary_peak=False,
    ):
        # Ignore the shared prominence value supplied internally by the base
        # workflow and substitute the MLI-specific threshold.
        return original_pick_dense_profile_set(
            profiles_db,
            peak_sigma=peak_sigma,
            prominence_db=args.mli_peak_prominence_db,
            min_distance_pixels=min_distance_pixels,
            peak_mode=peak_mode,
            respect_internal_nodata_gaps=respect_internal_nodata_gaps,
            allow_boundary_peak=allow_boundary_peak,
        )

    def pick_continuous_simsar_edge_with_sim_prominence(
        profiles_db,
        *,
        peak_sigma,
        prominence_db,
        min_distance_pixels,
        respect_internal_nodata_gaps,
        max_jump_pixels=4.0,
        continuity_penalty=0.25,
    ):
        # Ignore the shared prominence value supplied internally by the base
        # workflow and substitute the SimSAR-specific threshold.
        return original_pick_continuous_simsar_edge(
            profiles_db,
            peak_sigma=peak_sigma,
            prominence_db=args.simsar_peak_prominence_db,
            min_distance_pixels=min_distance_pixels,
            respect_internal_nodata_gaps=respect_internal_nodata_gaps,
            max_jump_pixels=max_jump_pixels,
            continuity_penalty=continuity_penalty,
        )

    base.pick_dense_profile_set = (
        pick_dense_profile_set_with_mli_prominence
    )
    base.pick_continuous_simsar_edge = (
        pick_continuous_simsar_edge_with_sim_prominence
    )

    run_ids = base.build_run_ids(
        args.id_start,
        args.id_end,
    )

    print("\n" + "=" * 80)
    print("SEPARATE MLI / SIMSAR PEAK PROMINENCE")
    print("=" * 80)
    print(
        f"MLI peak prominence:    "
        f"{args.mli_peak_prominence_db:g} dB"
    )
    print(
        f"SimSAR peak prominence: "
        f"{args.simsar_peak_prominence_db:g} dB"
    )
    print(
        "All other processing is inherited from "
        "invert_simsar_continuous_edge.py"
    )

    # The original workflow still requires its legacy shared argument.
    # Its value no longer controls the peak picking because the two picker
    # functions above substitute the separate thresholds. Passing the MLI
    # value here keeps the base script's diagnostic printout sensible.
    base.run_peak_inversion(
        mli_tif=args.mli_tif,
        simsar_dir=args.simsar_dir,
        run_ids=run_ids,
        output_dir=args.output_dir,
        simsar_pattern=args.simsar_pattern,
        median_size=args.median_size,
        simsar_median_size=args.simsar_median_size,
        peak_sigma=args.peak_sigma,
        peak_prominence_db=args.mli_peak_prominence_db,
        peak_distance_pixels=args.peak_distance_pixels,
        peak_mode=args.peak_mode,
        min_coverage=args.min_coverage,
        top_n=args.top_n,
        azimuth_min=args.azimuth_min,
        azimuth_max=args.azimuth_max,
        interaction=args.interaction,
        provenance_dir=args.provenance_dir,
        provenance_pattern=args.provenance_pattern,
        overview_grid_rows=args.overview_grid_rows,
        overview_grid_cols=args.overview_grid_cols,
        simsar_max_jump_pixels=args.simsar_max_jump_pixels,
        simsar_continuity_penalty=args.simsar_continuity_penalty,
    )


if __name__ == "__main__":
    main()
