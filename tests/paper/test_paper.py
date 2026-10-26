"""Paper assets: every ledger row resolves to its source and matches it exactly (independent re-resolution, not the builder's helper); the builders run from a clean tree and
reproduce the committed ledger and tables byte for byte; the Part 5 report numbers are deterministic and complete.  pytest tests/paper -q"""
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LED = json.loads((ROOT / "paper/numbers.json").read_text())


def walk(rel, path):
    cur = json.loads((ROOT / rel).read_text())
    for t in path:
        cur = cur[t] if isinstance(cur, list) else cur[str(t)]
    return cur


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_every_row_matches_its_source():
    assert len(LED["rows"]) >= 150
    for r in LED["rows"]:
        if r["derive"] == "mean3_range":
            vals = [walk(r["file"].format(s=s), r["path"]) for s in (0, 1, 2)]
            assert r["value"] == sum(vals) / 3 and r["ci"] == [min(vals), max(vals)], r["id"]
            continue
        assert r["value"] == walk(r["file"], r["path"]), r["id"]
        if r["ci_path"]:
            assert r["ci"] == walk(r["file"], r["ci_path"]), r["id"]
        assert r["status"] in ("pre-registered", "descriptive", "post hoc") and r["part"], r["id"]
        assert r["n"] is None or isinstance(r["n"], int) and r["n"] > 0, r["id"]


def test_ledger_files_agree():
    md = (ROOT / "paper/numbers.md").read_text()
    rows = list(csv.DictReader(open(ROOT / "paper/numbers.csv")))
    assert [r["id"] for r in rows] == [r["id"] for r in LED["rows"]]
    for r, c in zip(LED["rows"], rows):
        assert f"`{r['id']}`" in md and float(c["value"]) == r["value"]
    assert all(q["match"] for q in LED["quotes"]), [q for q in LED["quotes"] if not q["match"]]
    assert "PENDING" not in md and any(r["id"] == "p5.1.posthoc.p" for r in LED["rows"])
    post = [r for r in LED["rows"] if r["status"] == "post hoc"]
    assert post and all("POST HOC" in r["name"] for r in post)


def test_clean_tree_rebuild_is_identical(tmp_path):
    (tmp_path / "scripts").mkdir()
    shutil.copytree(ROOT / "scripts/paper", tmp_path / "scripts/paper", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "results").symlink_to(ROOT / "results")
    for step in ("build_ledger", "build_tables", "build_figures"):
        subprocess.run([sys.executable, str(tmp_path / f"scripts/paper/{step}.py")], check=True, capture_output=True, cwd=tmp_path)
    for name in ("numbers.json", "numbers.md", "numbers.csv"):
        assert sha(tmp_path / "paper" / name) == sha(ROOT / "paper" / name), name
    for f in sorted((ROOT / "paper/tables").iterdir()):
        assert sha(tmp_path / "paper/tables" / f.name) == sha(f), f.name
    for k in ("F1_pipeline", "F2_success_vs_silent", "F3_rescues_harms", "F4_any_of_k", "F5_ps_distribution", "F6_first_noise"):
        for ext in ("pdf", "png"):
            assert (tmp_path / "paper/figures" / f"{k}.{ext}").stat().st_size > 5000, k


def test_report_numbers_deterministic_and_complete():
    outs = [ROOT / "results/part5/notes/p5_report_numbers.json", ROOT / "results/part5/notes/p5_report_numbers.md", ROOT / "results/part5/notes/p5_report.md"]
    cmd = [sys.executable, str(ROOT / "scripts/p5_report_numbers.py")]
    subprocess.run(cmd, check=True, capture_output=True, cwd=ROOT)
    h1 = [sha(p) for p in outs]
    subprocess.run(cmd, check=True, capture_output=True, cwd=ROOT)
    assert h1 == [sha(p) for p in outs]
    rep = outs[2].read_text()
    assert "{{" not in rep and "PENDING" not in rep
    nd = json.loads(outs[0].read_text())["no_decision_states"]
    assert len(nd) == 5 and all(r["successes"] == 0 and r["reason"] != "other" for r in nd)
    # the report quotes the ledger's headline numbers
    n = json.loads(outs[0].read_text())["numbers"]
    assert n["nc_gap"] == f"{100 * walk('results/part5/nc/results.json', ('A_R', 'gap', 'est')):.1f} pp"


def test_readme_part5_numbers_match_ledger():
    rd = (ROOT / "README.md").read_text()
    sec = rd[rd.index("## Part 5"):rd.index("## Part 1: DeltaDINO")]
    led = {r["id"]: r for r in LED["rows"]}
    for rid in ("p5.0.gap", "p5.1a.auc", "p5.1a.k8_gain", "p5.1b.effect"):
        d = led[rid]["display"]
        norm = lambda x: x.replace("+", "").replace(" pp", "")  # noqa: E731
        assert norm(d) in norm(sec), (rid, d)
    assert f"{100 * led['p5.0.any64']['value']:.1f}% vs {100 * led['p5.0.mean_p']['value']:.1f}%" in sec
