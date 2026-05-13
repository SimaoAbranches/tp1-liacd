"""
stitcher.py — reconstrução de trajetórias individuais a partir de eventos anónimos

Abordagem: os eventos são processados por ordem cronológica
e para cada novo evento é tomada a decisão de associá-lo à trajetória aberta
que faz mais sentido. Se não encontrarmos nada compatível,
abrimos uma nova trajetória.

Executar com:
    python src/stitcher.py --input data/events.csv --output output/journeys.csv
"""

import argparse
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


#Configuração de heurísticas
# 2 minutos
MAX_GAP_SECONDS = 120
AGE_MISMATCH_PENALTY = 30

ZONE_ENTRY_PREFIXES = ("Z_E",)
ZONE_EXIT_PREFIXES  = ("Z_E", "Z_CK")

ZONE_TYPE_DWELL = {
    "Z_E":  (5,   30),
    "Z_C":  (60,  300),
    "Z_CK": (10,  60),
    "Z_N":  (10,  90),
    "Z_S":  (30,  600),
}

AGE_ADJACENCY = {
    "child":       ["child", "teenager"],
    "teenager":    ["child", "teenager", "young_adult"],
    "young_adult": ["teenager", "young_adult", "adult"],
    "adult":       ["young_adult", "adult", "middle_aged"],
    "middle_aged": ["adult", "middle_aged", "senior"],
    "senior":      ["middle_aged", "senior"],
}


def get_zone_prefix(zone_id: str) -> str:
    if str(zone_id).startswith("Z_CK"):
        return "Z_CK"
    return str(zone_id).rstrip("0123456789")


@dataclass
class ZoneVisit:
    zone_id:    str
    entry_time: pd.Timestamp
    exit_time:  Optional[pd.Timestamp] = None
    dwell_s:    int = 0
    gender:     str = "?"
    age_range:  str = "?"

@dataclass
class Trajectory:
    person_id:  str
    visits:     list = field(default_factory=list)
    last_event: Optional[pd.Timestamp] = None
    last_zone:  Optional[str] = None
    gender:     str = "?"
    age_range:  str = "?"
    is_closed:  bool = False

    def age_compatibility(self, age: str) -> float:
        if self.age_range == "?" or age == "?":
            return 0.5
        if self.age_range == age:
            return 1.0
        if age in AGE_ADJACENCY.get(self.age_range, []):
            return 0.5
        return 0.0

    def gender_compatibility(self, gender: str) -> float:
        if self.gender == "?" or gender == "?":
            return 0.5
        return 1.0 if self.gender == gender else 0.0


def is_currently_in_zone(traj: Trajectory) -> bool:
    """
    Verifica se a trajetória tem uma visita aberta.
    Uma pessoa não pode entrar numa nova zona sem ter saído da anterior.
    """
    if not traj.visits:
        return False
    return traj.visits[-1].exit_time is None


def update_demographics(traj: Trajectory, gender: str, age_range: str):
    """Atualiza género e idade usando o primeiro valor não desconhecido."""
    if traj.gender == "?" and gender != "?":
        traj.gender = gender
    if traj.age_range == "?" and age_range != "?":
        traj.age_range = age_range


# Score de compatibilidade

def compute_compatibility_score(traj: Trajectory, event: pd.Series) -> Optional[float]:
    """
    Calcula um score de compatibilidade entre a trajetória e o evento.
    Retorna None se for impossível associar.
    """
    if traj.is_closed or traj.last_event is None:
        return None

    gap = (event["timestamp"] - traj.last_event).total_seconds()

    if gap < 0 or gap > MAX_GAP_SECONDS:
        return None

    time_score = 1.0 - (gap / MAX_GAP_SECONDS)

    gender_score = traj.gender_compatibility(event.get("gender", "?"))
    age_score    = traj.age_compatibility(event.get("age_range", "?"))

    # Género diferente
    if gender_score == 0.0:
        return -1.0

    if age_score == 0.0:
        age_score = -AGE_MISMATCH_PENALTY / 100

    # Bónus de sequência
    sequence_bonus = 0.0
    if traj.last_zone:
        prev_prefix = get_zone_prefix(traj.last_zone)
        curr_prefix = get_zone_prefix(event["zone_id"])
        forward_transitions = {
            "Z_E":  ["Z_N", "Z_S", "Z_C"],
            "Z_N":  ["Z_S", "Z_C", "Z_N", "Z_E"],
            "Z_S":  ["Z_N", "Z_S", "Z_C"],
            "Z_C":  ["Z_CK", "Z_E"],
            "Z_CK": ["Z_E"],
        }
        if curr_prefix in forward_transitions.get(prev_prefix, []):
            sequence_bonus = 0.1

    return (time_score * 0.55) + (gender_score * 0.25) + (age_score * 0.10) + sequence_bonus


# Algoritmo principal

def stitch(events_df: pd.DataFrame) -> list:
    open_trajectories  = []
    closed_trajectories = []

    df = events_df.sort_values("timestamp").reset_index(drop=True)
    print(f"Processar {len(df)} eventos...")

    for _, event in df.iterrows():
        event_type = event["event_type"]
        zone       = event["zone_id"]
        ts         = event["timestamp"]
        gender     = event.get("gender", "?")
        age        = event.get("age_range", "?")
        duration   = event.get("duration_s", 0) or 0

        # Fechar trajetórias antigas
        cutoff  = ts - pd.Timedelta(seconds=MAX_GAP_SECONDS)
        expired = [t for t in open_trajectories
                   if t.last_event and t.last_event < cutoff]
        for t in expired:
            t.is_closed = True
            closed_trajectories.append(t)
        open_trajectories = [t for t in open_trajectories if not t.is_closed]

        if event_type == "entry":
            best_traj, best_score = None, -999

            for traj in open_trajectories:
                # Invariante de consistência: só associar a trajetórias sem visita aberta
                if is_currently_in_zone(traj):
                    continue
                score = compute_compatibility_score(traj, event)
                if score is not None and score > best_score:
                    best_score = score
                    best_traj  = traj

            if best_traj is not None and best_score > 0.1:
                visit = ZoneVisit(zone_id=zone, entry_time=ts, gender=gender, age_range=age)
                best_traj.visits.append(visit)
                best_traj.last_event = ts
                best_traj.last_zone  = zone
                update_demographics(best_traj, gender, age)
            else:
                new_traj = Trajectory(
                    person_id=f"P_{uuid.uuid4().hex[:6].upper()}",
                    last_event=ts,
                    last_zone=zone,
                    gender=gender,
                    age_range=age,
                )
                visit = ZoneVisit(zone_id=zone, entry_time=ts, gender=gender, age_range=age)
                new_traj.visits.append(visit)
                open_trajectories.append(new_traj)

        elif event_type == "exit":
            best_traj, best_score = None, -999

            for traj in open_trajectories:
                score = compute_compatibility_score(traj, event)
                if traj.last_zone == zone and score is not None:
                    score += 0.5  # bónus por match de zona
                if score is not None and score > best_score:
                    best_score = score
                    best_traj  = traj

            if best_traj is not None and best_score > 0.0:
                if best_traj.visits:
                    last_visit = best_traj.visits[-1]
                    if last_visit.exit_time is None:
                        last_visit.exit_time = ts
                best_traj.last_event = ts
                best_traj.last_zone  = zone
                update_demographics(best_traj, gender, age)

                if any(zone.startswith(p) for p in ZONE_EXIT_PREFIXES):
                    best_traj.is_closed = True
                    closed_trajectories.append(best_traj)
                    open_trajectories.remove(best_traj)

        elif event_type == "linger":
            for traj in open_trajectories:
                if traj.last_zone == zone and traj.visits:
                    last_visit = traj.visits[-1]
                    if last_visit.zone_id == zone and last_visit.exit_time is None:
                        last_visit.dwell_s = int(duration)
                        traj.last_event    = ts
                        break

    # Fechar trajetórias que ficaram abertas no fim
    for t in open_trajectories:
        t.is_closed = True
        closed_trajectories.append(t)

    print(f"  → {len(closed_trajectories)} trajectórias reconstruídas")
    return closed_trajectories


# Conversão para DataFrame

def trajectories_to_df(trajectories: list) -> pd.DataFrame:
    rows = []
    for traj in trajectories:
        for visit in traj.visits:
            exit_time = visit.exit_time
            if exit_time is None and visit.dwell_s > 0:
                exit_time = visit.entry_time + pd.Timedelta(seconds=visit.dwell_s)
            rows.append({
                "person_id":   traj.person_id,
                "zone_id":     visit.zone_id,
                "entry_time":  visit.entry_time,
                "exit_time":   exit_time,
                "dwell_s":     visit.dwell_s,
                "gender":      traj.gender,
                "age_range":   traj.age_range,
                "visit_date":  visit.entry_time.date(),
                "hour_of_day": visit.entry_time.hour,
            })
    return pd.DataFrame(rows)


# Métricas de qualidade

def print_quality_metrics(journeys_df: pd.DataFrame, original_df: pd.DataFrame):
    print("\nMétricas de qualidade")

    violations = 0
    n_trajs    = 0
    for (_, _), group in journeys_df.groupby(["person_id", "visit_date"]):
        n_trajs += 1
        g = group.sort_values("entry_time").reset_index(drop=True)
        for i in range(len(g) - 1):
            if pd.notna(g.iloc[i]["exit_time"]) and g.iloc[i+1]["entry_time"] < g.iloc[i]["exit_time"]:
                violations += 1
                break

    consistency = 1.0 - (violations / max(n_trajs, 1))
    print(f"  Consistência:  {consistency:.1%}  ({violations} violações em {n_trajs} trajectórias)")

    n_entry  = len(original_df[original_df["event_type"] == "entry"])
    n_rows   = len(journeys_df)
    coverage = n_rows / max(n_entry, 1)
    print(f"  Cobertura:     ≈ {coverage:.1%}  ({n_rows} visitas / {n_entry} eventos entry)")

    n_complete = 0
    for (_, _), g in journeys_df.groupby(["person_id", "visit_date"]):
        g = g.sort_values("entry_time")
        first = str(g.iloc[0]["zone_id"])
        last  = str(g.iloc[-1]["zone_id"])
        first_prefix = "Z_CK" if first.startswith("Z_CK") else first.rstrip("0123456789")
        last_prefix  = "Z_CK" if last.startswith("Z_CK") else last.rstrip("0123456789")
        if first_prefix == "Z_E" and last_prefix in ("Z_E", "Z_CK"):
            n_complete += 1
    completeness = n_complete / max(n_trajs, 1)
    print(f"  Completude:    {completeness:.1%}  ({n_complete}/{n_trajs} trajectórias com entrada+saída válidas)")

    gaps = []
    for (_, _), g in journeys_df.groupby(["person_id", "visit_date"]):
        g = g.sort_values("entry_time")
        for i in range(len(g) - 1):
            t1 = g.iloc[i]["exit_time"]
            t2 = g.iloc[i+1]["entry_time"]
            if pd.notna(t1) and pd.notna(t2):
                gaps.append((t2 - t1).total_seconds())
    if gaps:
        arr = np.array(gaps)
        print(f"  Gaps temporais entre zonas:")
        print(f"    mediana={np.median(arr):.0f}s  p90={np.percentile(arr, 90):.0f}s  max={arr.max():.0f}s")



# Entry point

def main():
    parser = argparse.ArgumentParser(
        description="Reconstrói trajectórias individuais a partir de eventos anónimos"
    )
    parser.add_argument("--input",  required=True, help="Caminho para events.csv")
    parser.add_argument("--output", required=True, help="Caminho para journeys.csv")
    args = parser.parse_args()

    print(f"\nA ler eventos de: {args.input}")
    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    df.columns = df.columns.str.strip().str.lower()

    print(f"{len(df)} eventos carregados ({df['timestamp'].min().date()} a {df['timestamp'].max().date()})")

    print("\nA reconstruir trajectórias...")
    trajectories = stitch(df)

    print("\nA gerar journeys.csv...")
    journeys_df = trajectories_to_df(trajectories)

    print_quality_metrics(journeys_df, df)

    journeys_df.to_csv(args.output, index=False)
    print(f"Ficheiro guardado em: {args.output}")
    print(f"{len(journeys_df)} linhas  |  {journeys_df['person_id'].nunique()} pessoas únicas\n")


if __name__ == "__main__":
    main()