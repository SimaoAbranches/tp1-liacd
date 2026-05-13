"""
evaluate.py — harness de avaliação do pipeline de ponta a ponta

Uso:
    python evaluate.py --data events_validation.csv --output evaluation_report.json

Corre o pipeline completo sobre o dataset fornecido e calcula:
  - Consistência      : % de trajectórias sem sobreposição temporal
  - Cobertura         : % de eventos atribuídos a alguma trajectória
  - Completude        : % de trajectórias com entrada e saída válidas
  - Deteção anomalias : % das anomalias injectadas correctamente identificadas
  - Precisão numérica : % de valores numéricos nos insights verificáveis nos dados
  - Ausência alucinação: % de afirmações factuais no report verificáveis no metrics.json
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

NUMERIC_TOLERANCE  = 0.05
ANOMALY_ZONE_PATTERN = re.compile(r"Z_[A-Z]+\d*")


# ─── Correr o pipeline ───────────────────────────────────────────────────────────

def run_pipeline(data_path: str, work_dir: Path) -> dict:
    journeys = work_dir / "journeys.csv"
    metrics  = work_dir / "metrics.json"
    insights = work_dir / "insights.json"
    report   = work_dir / "weekly_report.md"

    steps = [
        ("stitcher",  [sys.executable, "src/stitcher.py",  "--input", data_path,       "--output", str(journeys)]),
        ("analytics", [sys.executable, "src/analytics.py", "--input", str(journeys),   "--output", str(metrics)]),
        ("insights",  [sys.executable, "src/insights.py",  "--input", str(metrics),    "--output", str(insights), "--strategy", "b"]),
        ("report",    [sys.executable, "src/report.py",    "--input", str(insights),   "--output", str(report), "--metrics", str(metrics)]),
    ]

    step_results = {}
    for name, cmd in steps:
        print(f"  [{name}]", end=" ", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            print(f"✗ ERRO")
            for line in proc.stderr.strip().splitlines()[-5:]:
                print(f"    {line}")
            step_results[name] = {"ok": False, "error": proc.stderr[-300:]}
            break
        print("✓")
        step_results[name] = {"ok": True}

    return {
        "step_results": step_results,
        "journeys": str(journeys),
        "metrics":  str(metrics),
        "insights": str(insights),
        "report":   str(report),
    }


# ─── Consistência ────────────────────────────────────────────────────────────────

def eval_consistency(journeys_path: str) -> dict:
    df = pd.read_csv(journeys_path, parse_dates=["entry_time", "exit_time"])
    violations, n_trajs = 0, 0
    for (_, _), group in df.groupby(["person_id", "visit_date"]):
        n_trajs += 1
        g = group.sort_values("entry_time").reset_index(drop=True)
        for i in range(len(g) - 1):
            if pd.notna(g.iloc[i]["exit_time"]) and g.iloc[i+1]["entry_time"] < g.iloc[i]["exit_time"]:
                violations += 1
                break
    score = 1.0 - (violations / n_trajs) if n_trajs > 0 else 0.0
    return {
        "score": round(score, 4),
        "pct":   f"{score*100:.2f}%",
        "violations": violations,
        "total_trajectories": n_trajs,
    }


# ─── Cobertura ───────────────────────────────────────────────────────────────────

def eval_coverage(data_path: str, journeys_path: str) -> dict:
    events   = pd.read_csv(data_path)
    journeys = pd.read_csv(journeys_path)
    n_entry  = len(events[events["event_type"] == "entry"])
    n_rows   = len(journeys)
    score    = min(n_rows / n_entry, 1.0) if n_entry > 0 else 0.0
    return {
        "score": round(score, 4),
        "pct":   f"{score*100:.2f}%",
        "entry_events":  n_entry,
        "journeys_rows": n_rows,
    }


# ─── Completude ──────────────────────────────────────────────────────────────────

def eval_completeness(journeys_path: str) -> dict:
    df = pd.read_csv(journeys_path, parse_dates=["entry_time"])

    def prefix(z):
        z = str(z)
        return "Z_CK" if z.startswith("Z_CK") else z.rstrip("0123456789")

    n_complete, n_total = 0, 0
    for (_, _), group in df.groupby(["person_id", "visit_date"]):
        n_total += 1
        g = group.sort_values("entry_time")
        if prefix(g.iloc[0]["zone_id"]) == "Z_E" and prefix(g.iloc[-1]["zone_id"]) in ("Z_E", "Z_CK"):
            n_complete += 1

    score = n_complete / n_total if n_total > 0 else 0.0
    return {
        "score":    round(score, 4),
        "pct":      f"{score*100:.2f}%",
        "complete": n_complete,
        "total":    n_total,
    }


# ─── Deteção de anomalias ────────────────────────────────────────────────────────

def eval_anomaly_detection(insights_path: str, metrics_path: str) -> dict:
    with open(insights_path, "r", encoding="utf-8") as f:
        insights_full = json.load(f)
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Anomalias reais calculadas pelo analytics (z > 2σ)
    known = [
        {"zone_id": a["zone_id"], "hour": a["hour_of_day"]}
        for a in metrics.get("anomalies", {}).get("anomalies_zone_hour", [])
    ]
    if not known:
        return {"score": None, "note": "Nenhuma anomalia no metrics.json"}

    # Zonas+horas mencionadas nos insights gerados
    detected = set()
    for key in ["estrategia_A_zero_shot", "estrategia_B_few_shot"]:
        for ins in insights_full.get(key, {}).get("insights", []):
            if ins.get("categoria") == "anomalia":
                text  = ins.get("observacao", "") + " " + ins.get("titulo", "")
                zones = ANOMALY_ZONE_PATTERN.findall(text)
                hours = [int(h) for h in re.findall(r"\b([0-9]{1,2})h\b", text)]
                for z in zones:
                    for h in hours:
                        detected.add((z, h))
                    if not hours:
                        detected.add((z, None))

    found = sum(
        1 for a in known
        if any(z == a["zone_id"] and (h == a["hour"] or h is None)
               for z, h in detected)
    )
    score = found / len(known)
    return {
        "score":          round(score, 4),
        "pct":            f"{score*100:.2f}%",
        "detected":       found,
        "total_anomalies": len(known),
    }


# ─── Precisão numérica ───────────────────────────────────────────────────────────

def eval_numeric_precision(insights_path: str, metrics_path: str) -> dict:
    with open(insights_path, "r", encoding="utf-8") as f:
        insights_full = json.load(f)
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    all_numbers = set()
    def flatten(obj):
        if isinstance(obj, (int, float)):
            all_numbers.add(float(obj))
        elif isinstance(obj, dict):
            for v in obj.values(): flatten(v)
        elif isinstance(obj, list):
            for v in obj: flatten(v)
    flatten(metrics)

    def verifiable(n):
        return any(m != 0 and abs(n - m) / abs(m) <= NUMERIC_TOLERANCE for m in all_numbers)

    total, verified = 0, 0
    best = insights_full.get("estrategia_B_few_shot") or insights_full.get("estrategia_A_zero_shot", {})
    for ins in best.get("insights", []):
        for n_str in re.findall(r"\b\d+(?:[.,]\d+)?\b", ins.get("observacao", "")):
            try:
                n = float(n_str.replace(",", "."))
                if n < 2: continue
                total += 1
                if verifiable(n): verified += 1
            except ValueError:
                pass

    score = verified / total if total > 0 else 1.0
    return {
        "score":    round(score, 4),
        "pct":      f"{score*100:.2f}%",
        "verified": verified,
        "total":    total,
    }


# ─── Ausência de alucinação ──────────────────────────────────────────────────────

def eval_hallucination(report_path: str, metrics_path: str) -> dict:
    report_text = Path(report_path).read_text(encoding="utf-8")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    checks = []

    # Zonas mencionadas devem existir nos dados
    known_zones = set(metrics.get("zones", {}).get("zone_stats", {}).keys())
    for zone in ANOMALY_ZONE_PATTERN.findall(report_text):
        checks.append({"type": "zone", "value": zone, "ok": zone in known_zones})

    # Datas mencionadas devem estar no intervalo do dataset
    dr    = metrics.get("dataset_info", {}).get("date_range", {})
    start = dr.get("start", "")
    end   = dr.get("end", "")
    for d in re.findall(r"\d{4}-\d{2}-\d{2}", report_text):
        checks.append({"type": "date", "value": d, "ok": start <= d <= end})

    if not checks:
        return {"score": 1.0, "pct": "100%", "total_checks": 0}

    passed = sum(1 for c in checks if c["ok"])
    score  = passed / len(checks)
    return {
        "score":        round(score, 4),
        "pct":          f"{score*100:.2f}%",
        "passed":       passed,
        "total_checks": len(checks),
        "failures":     [c for c in checks if not c["ok"]][:5],
    }


# ─── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("\n══════════════════════════════════════════════════")
    print("  Harness de avaliação — Retail Intelligence Pipeline")
    print("══════════════════════════════════════════════════\n")

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)

        print("Passo 1 — A correr pipeline...")
        p = run_pipeline(args.data, work_dir)
        pipeline_ok = all(v["ok"] for v in p["step_results"].values())

        print("\nPasso 2 — A calcular métricas...")
        results = {}

        try:
            print("  Consistência...",      end=" "); r = eval_consistency(p["journeys"]);                          print(r["pct"]); results["consistency"]   = r
            print("  Cobertura...",         end=" "); r = eval_coverage(args.data, p["journeys"]);                  print(r["pct"]); results["coverage"]      = r
            print("  Completude...",        end=" "); r = eval_completeness(p["journeys"]);                         print(r["pct"]); results["completeness"]  = r
            print("  Deteção anomalias...", end=" "); r = eval_anomaly_detection(p["insights"], p["metrics"]);      print(r.get("pct", "N/D")); results["anomaly"]   = r
            print("  Precisão numérica...", end=" "); r = eval_numeric_precision(p["insights"], p["metrics"]);      print(r["pct"]); results["numeric"]       = r
            print("  Ausência alucinação...",end=" "); r = eval_hallucination(p["report"], p["metrics"]);           print(r["pct"]); results["hallucination"] = r
        except FileNotFoundError as e:
            print(f"\n  Ficheiro em falta: {e}")

        # Score global ponderado
        weights = {"consistency": 0.25, "coverage": 0.15, "completeness": 0.20,
                   "anomaly": 0.20, "numeric": 0.10, "hallucination": 0.10}
        ws, wt = 0.0, 0.0
        for k, w in weights.items():
            s = results.get(k, {}).get("score")
            if s is not None:
                ws += s * w; wt += w
        global_score = ws / wt if wt > 0 else 0.0

        report = {
            "dataset":      args.data,
            "pipeline_ok":  pipeline_ok,
            "pipeline_steps": p["step_results"],
            "metrics":      results,
            "global_score": round(global_score, 4),
            "global_pct":   f"{global_score*100:.2f}%",
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n══════════════════════════════════════════════════")
    print(f"  Score global: {report['global_pct']}")
    print(f"  Pipeline OK:  {'✓' if pipeline_ok else '✗'}")
    print(f"══════════════════════════════════════════════════")
    print(f"\nRelatório guardado em: {args.output}\n")


if __name__ == "__main__":
    main()