import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TOLERANCE = 0.05  # allowable relative/absolute deviation


def _compare_values(exp, act, tol, key):
    if exp is None:
        assert act is None, f"{key}: expected None, got {act}"
    elif isinstance(exp, list):
        assert isinstance(act, list), f"{key}: expected list, got {type(act)}"
        assert len(exp) == len(act), f"{key}: list length mismatch"
        for i, (e, a) in enumerate(zip(exp, act)):
            _compare_values(e, a, tol, f"{key}[{i}]")
    elif isinstance(exp, (int, float)):
        assert isinstance(act, (int, float)), f"{key}: expected numeric, got {type(act)}"
        diff = abs(act - exp)
        assert diff <= tol * max(1.0, abs(exp)), (
            f"{key}: diff {diff} exceeds tolerance {tol}"
        )
    else:
        assert act == exp, f"{key}: expected {exp}, got {act}"


def _compare_dicts(expected, actual, tol):
    assert expected.keys() == actual.keys(), "JSON keys mismatch"
    for k in expected:
        _compare_values(expected[k], actual[k], tol, k)


def main() -> int:
    repo = Path(__file__).parent
    main_py = repo / "main.py"
    smile_src = repo / "smile.wav"
    expected_scores = json.loads((repo / "example_scores.json").read_text())[0]
    expected_features = json.loads((repo / "example_features.json").read_text())[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        shutil.copy(smile_src, tmp / "smile.wav")
        subprocess.run([sys.executable, str(main_py)], cwd=tmp, check=True)

        # Ensure files exist
        for name in ["features.csv", "features.json", "scores.csv", "scores.json"]:
            assert (tmp / name).exists(), f"Missing output file: {name}"

        scores = json.loads((tmp / "scores.json").read_text())[0]
        features = json.loads((tmp / "features.json").read_text())[0]

    _compare_dicts(expected_scores, scores, TOLERANCE)
    _compare_dicts(expected_features, features, TOLERANCE)
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
