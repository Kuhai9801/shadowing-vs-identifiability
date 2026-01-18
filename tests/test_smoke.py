import os
import subprocess
import sys


def _assert_nonempty(path: str) -> None:
    assert os.path.exists(path), f"Missing output: {path}"
    assert os.path.getsize(path) > 0, f"Empty output: {path}"


def test_smoke_pipeline(tmp_path):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = tmp_path / "results"
    reports_dir = tmp_path / "reports"

    cmd = [
        sys.executable,
        os.path.join(root, "scripts", "run_pipeline.py"),
        "--mode",
        "smoke",
        "--device",
        "cpu",
        "--num_workers",
        "1",
        "--results_dir",
        str(results_dir),
        "--reports_dir",
        str(reports_dir),
    ]
    subprocess.run(cmd, check=True, cwd=root)

    smoke_results = results_dir / "smoke"
    smoke_reports = reports_dir / "smoke" / "figures"

    _assert_nonempty(str(smoke_results / "blockA_summary.csv"))
    _assert_nonempty(str(smoke_results / "blockA_success.csv"))
    _assert_nonempty(str(smoke_results / "blockA_hypotheses.csv"))
    _assert_nonempty(str(smoke_results / "blockB_summary.csv"))
    _assert_nonempty(str(smoke_results / "blockB_success.csv"))
    _assert_nonempty(str(smoke_results / "blockB_hypotheses.csv"))

    _assert_nonempty(str(smoke_reports / "money_plot1_collapse_K_over_bq.png"))
    _assert_nonempty(str(smoke_reports / "money_plot2_Lshape.png"))
    _assert_nonempty(str(smoke_reports / "minimax_bound_plot.png"))
    _assert_nonempty(str(smoke_reports / "horizon_plot.png"))
