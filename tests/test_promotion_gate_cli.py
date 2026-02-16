"""CLI tests for run_promotion_gate.py."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _write_inputs(tmp_path: Path, baseline_value: float, candidate_value: float) -> tuple[Path, Path, Path]:
    n = 700
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.RandomState(123)
    returns = 0.0007 + rng.randn(n) * 0.001
    prices = 100.0 * np.exp(np.cumsum(returns))

    prices_df = pd.DataFrame({"timestamp": ts, "close": prices})
    baseline_df = pd.DataFrame({"timestamp": ts, "forecast": baseline_value})
    candidate_df = pd.DataFrame({"timestamp": ts, "forecast": candidate_value})

    prices_path = tmp_path / "prices.csv"
    baseline_path = tmp_path / "baseline.csv"
    candidate_path = tmp_path / "candidate.csv"
    prices_df.to_csv(prices_path, index=False)
    baseline_df.to_csv(baseline_path, index=False)
    candidate_df.to_csv(candidate_path, index=False)
    return prices_path, baseline_path, candidate_path


def _base_cmd() -> list[str]:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_promotion_gate.py"
    return [sys.executable, str(script)]


def test_promotion_gate_cli_passes_and_writes_json(tmp_path):
    prices, baseline, candidate = _write_inputs(tmp_path, baseline_value=0.0, candidate_value=10.0)
    output = tmp_path / "gate_report.json"

    cmd = _base_cmd() + [
        "--prices-file",
        str(prices),
        "--baseline-file",
        str(baseline),
        "--candidate-file",
        str(candidate),
        "--timeframe",
        "1h",
        "--n-splits",
        "4",
        "--test-size",
        "100",
        "--purge",
        "24",
        "--embargo",
        "24",
        "--buffer-fraction",
        "0.0",
        "--taker-fee-bps",
        "0.0",
        "--slippage-bps",
        "0.0",
        "--min-mean-sharpe-delta",
        "0.01",
        "--min-outperform-fraction",
        "0.75",
        "--max-drawdown-worsening",
        "1.0",
        "--output-json",
        str(output),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert output.exists()

    payload = json.loads(output.read_text())
    assert payload["approved"] is True
    assert len(payload["splits"]) == 4


def test_promotion_gate_cli_rejects_worse_candidate(tmp_path):
    prices, baseline, candidate = _write_inputs(tmp_path, baseline_value=10.0, candidate_value=-10.0)
    cmd = _base_cmd() + [
        "--prices-file",
        str(prices),
        "--baseline-file",
        str(baseline),
        "--candidate-file",
        str(candidate),
        "--timeframe",
        "1h",
        "--n-splits",
        "4",
        "--test-size",
        "100",
        "--purge",
        "24",
        "--embargo",
        "24",
        "--buffer-fraction",
        "0.0",
        "--taker-fee-bps",
        "0.0",
        "--slippage-bps",
        "0.0",
        "--min-mean-sharpe-delta",
        "0.01",
        "--min-outperform-fraction",
        "0.75",
        "--max-drawdown-worsening",
        "1.0",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 2, proc.stderr + proc.stdout
    assert "Approved: False" in proc.stdout
