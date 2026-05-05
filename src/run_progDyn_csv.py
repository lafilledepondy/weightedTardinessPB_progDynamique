from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Iterable

from progDyn import relax1, relax2, optimalOrRealisableOrInfesable
from readData_progDyn import readData

STATUS_MAP = {
    "Solution réalisable (donc optimale)": "Opt",
    "Solution NON réalisable (borne dual)": "NonReal",
}

# Put your input files here.
dataFile = [
    "data_aone/Toy_wt4.1.dat",
    "data_aone/Toy_wt4.2.dat",
    "data_aone/wt040/wt040_001.dat",
    "data_aone/wt040/wt040_002.dat",
    "data_aone/wt040/wt040_005.dat",
    "data_aone/wt050/wt050_001.dat",    
    "data_aone/wt050/wt050_002.dat",
    "data_aone/wt050/wt050_005.dat"    
]

# Choose which relaxation(s) to run: 1, 2, or [1, 2]
whichRelax = [2]


def _format_cpu_seconds(value: float) -> str:
    # Scientific notation formatted like 7.36e-4 (without zero-padded exponent).
    txt = f"{value:.2e}"
    txt = txt.replace("e-0", "e-").replace("e+0", "e+")
    return txt


def _resolve_data_file(project_root: Path, raw_name: str) -> Path:
    raw_path = Path(raw_name)

    if not raw_path.is_absolute():
        project_relative = project_root / raw_path
        if project_relative.exists():
            return project_relative.resolve()

    if raw_path.exists():
        return raw_path.resolve()

    if not raw_path.suffix:
        names_to_try = [raw_name, f"{raw_name}.dat"]
    else:
        names_to_try = [raw_name]

    search_roots = [
        project_root / "data_aone",
    ]

    for root in search_roots:
        if not root.exists():
            continue

        for candidate_name in names_to_try:
            if "/" in candidate_name:
                candidate = root / candidate_name
                if candidate.exists():
                    return candidate.resolve()
            else:
                matches = sorted(root.rglob(candidate_name))
                if len(matches) == 1:
                    return matches[0].resolve()
                if len(matches) > 1:
                    raise ValueError(
                        f"Multiple matches for '{raw_name}'. "
                        f"Please provide a more specific path."
                    )

    raise FileNotFoundError(
        f"Could not resolve data file '{raw_name}'. "
        "Use an explicit path or a unique filename/stem."
    )


def _run_relax(data_path: Path, relax_id: int) -> dict[str, str]:
    nb_items, T, processing_times, due_dates, penalties = readData(str(data_path))

    start = time.perf_counter()
    if relax_id == 1:
        L_tab, sequence = relax1(nb_items, T, processing_times, due_dates, penalties)
        dual_bound = L_tab[0]
    elif relax_id == 2:
        L_tab, sequence = relax2(nb_items, T, processing_times, due_dates, penalties)
        dual_bound = L_tab[0][0]
    else:
        raise ValueError(f"Unsupported relax id: {relax_id}")
    elapsed = time.perf_counter() - start

    status_raw = optimalOrRealisableOrInfesable(sequence, nb_items, T, processing_times)
    status = STATUS_MAP.get(status_raw, status_raw)

    return {
        "dataFile": data_path.stem,
        "relax": str(relax_id),
        "Dual Bound": str(dual_bound),
        "status": status,
        "CPU (s)": _format_cpu_seconds(elapsed),
    }


def export_prog_dyn_results(
    data_files: Iterable[str],
    output_csv: str = "results/progDyn_results.csv",
    sep: str = "&",
) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    output_path = (project_root / output_csv).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []

    relax_ids = whichRelax if isinstance(whichRelax, (list, tuple, set)) else [whichRelax]
    invalid_relax_ids = [rid for rid in relax_ids if rid not in (1, 2)]
    if invalid_relax_ids:
        raise ValueError(f"Invalid relax id(s): {invalid_relax_ids}. Use 1 and/or 2.")

    for data_file in data_files:
        data_path = _resolve_data_file(project_root, data_file)
        for relax_id in relax_ids:
            rows.append(_run_relax(data_path, relax_id))

    fieldnames = ["dataFile", "relax", "Dual Bound", "status", "CPU (s)"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=sep)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run progDyn relaxations on data files and export results to CSV."
    )
    parser.add_argument(
        "data_files",
        nargs="*",
        help=(
            "Data files to run. Examples: wt040_002, wt040_002.dat, "
            "data_aone/wt040/wt040_002.dat"
        ),
    )
    parser.add_argument(
        "--output",
        default="results/progDyn_results.csv",
        help="Output CSV path relative to project root (default: results/progDyn_results.csv)",
    )
    parser.add_argument(
        "--sep",
        default="&",
        help="CSV separator (default: '&')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_files = args.data_files if args.data_files else dataFile
    output_path = export_prog_dyn_results(selected_files, output_csv=args.output, sep=args.sep)
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()



