"""
analytics.py — transforma journeys.csv num metrics.json rico e completo

O princípio fundamental deste módulo está no enunciado e vale a pena repetir:
a LLM nunca faz contas. Nunca. Toda a matemática acontece aqui.
Quando o insights.py chamar a LLM, ela vai receber factos prontos —
médias, desvios, rankings, anomalias — e a sua única tarefa é
transformar esses factos em linguagem natural.

Este ficheiro calcula cinco famílias de métricas:
  1. Tráfego geral (visitantes por dia, por hora, tempo médio de visita)
  2. Performance por zona (tráfego, dwell, taxa de paragem, sequências)
  3. Funil de cliente (da entrada à caixa, onde se perde tráfego)
  4. Segmentação demográfica (género e idade por hora e por zona)
  5. Anomalias (desvios > 2σ no dia 7 face à média dos primeiros 6 dias)

Executar com:
    python src/analytics.py --input output/journeys.csv --output output/metrics.json
"""

import argparse
import json
import warnings
from collections import Counter, defaultdict
from itertools import islice

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")



def get_zone_prefix(zone_id: str) -> str:
    """Z_S3 → Z_S,  Z_CK → Z_CK,  Z_N10 → Z_N"""
    if str(zone_id).startswith("Z_CK"):
        return "Z_CK"
    return str(zone_id).rstrip("0123456789")


def is_checkout_zone(zone_id: str) -> bool:
    p = get_zone_prefix(zone_id)
    return p in ("Z_C", "Z_CK")


def safe_round(val, decimals=2):
    """Arredonda de forma segura — lida com NaN e None sem explodir."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), decimals)


def top_n(counter_dict: dict, n: int = 10) -> list[dict]:
    """Converte um dicionário de contagens no top-N como lista de dicts."""
    sorted_items = sorted(counter_dict.items(), key=lambda x: x[1], reverse=True)
    return [{"key": k, "count": int(v)} for k, v in sorted_items[:n]]



def calc_traffic_metrics(df: pd.DataFrame) -> dict:
    """
    Métricas gerais de afluência à loja.

    'Visitante único por dia' = person_id distinto que teve pelo menos
    um evento nesse dia. Usamos visit_date que já vem do stitcher.
    """
    print("  [1/5] Métricas de tráfego...")

    # Visitantes únicos por dia
    daily = (
        df.groupby("visit_date")["person_id"]
        .nunique()
        .reset_index(name="unique_visitors")
        .sort_values("visit_date")
    )

    visitors_per_day = {
        str(row["visit_date"]): int(row["unique_visitors"])
        for _, row in daily.iterrows()
    }

    # Visitantes únicos por hora (agregado ao longo de toda a semana)
    hourly = (
        df.groupby("hour_of_day")["person_id"]
        .nunique()
        .reset_index(name="unique_visitors")
        .sort_values("hour_of_day")
    )
    visitors_per_hour = {
        int(row["hour_of_day"]): int(row["unique_visitors"])
        for _, row in hourly.iterrows()
    }

    # Hora de pico (hora com mais visitantes em média por dia)
    hourly_by_day = (
        df.groupby(["visit_date", "hour_of_day"])["person_id"]
        .nunique()
        .reset_index(name="visitors")
    )
    mean_by_hour = hourly_by_day.groupby("hour_of_day")["visitors"].mean()
    peak_hour = int(mean_by_hour.idxmax())

    # Tempo médio de visita à loja (do primeiro entry ao último exit por pessoa por dia)
    # Agrupamos por person_id + visit_date para capturar a visita completa
    visit_spans = df.groupby(["person_id", "visit_date"]).agg(
        first_entry=("entry_time", "min"),
        last_exit=("exit_time",  "max"),
    ).reset_index()
    visit_spans["duration_min"] = (
        (visit_spans["last_exit"] - visit_spans["first_entry"])
        .dt.total_seconds() / 60
    )
    # Filtrar durações implausíveis (< 1 min ou > 3 horas — provavelmente artefactos)
    valid_spans = visit_spans[
        (visit_spans["duration_min"] >= 1) &
        (visit_spans["duration_min"] <= 180)
    ]
    avg_visit_min = safe_round(valid_spans["duration_min"].mean())
    median_visit_min = safe_round(valid_spans["duration_min"].median())

    # Dia mais e menos movimentado
    busiest_day  = max(visitors_per_day, key=visitors_per_day.get)
    quietest_day = min(visitors_per_day, key=visitors_per_day.get)

    total_visitors_week = int(df["person_id"].nunique())

    return {
        "total_visitors_week":  total_visitors_week,
        "visitors_per_day":     visitors_per_day,
        "visitors_per_hour":    visitors_per_hour,
        "peak_hour":            peak_hour,
        "avg_visit_duration_min":    avg_visit_min,
        "median_visit_duration_min": median_visit_min,
        "busiest_day":  busiest_day,
        "quietest_day": quietest_day,
        "daily_avg_visitors": safe_round(daily["unique_visitors"].mean()),
        "daily_std_visitors": safe_round(daily["unique_visitors"].std()),
    }



def calc_zone_metrics(df: pd.DataFrame) -> dict:
    """
    Para cada zona calculamos:
    - tráfego total (visitas) e visitantes únicos
    - dwell time médio (apenas para linhas com dwell_s > 0, ou seja, com linger)
    - taxa de paragem = visitantes com dwell > 0 / visitantes totais
    - ranking de zonas por tráfego

    As sequências de zonas mais frequentes são calculadas a partir
    das trajectórias individuais — concatenamos as zonas por ordem
    de entrada e extraímos bi-gramas e tri-gramas.
    """
    print("  [2/5] Métricas por zona...")

    # Tráfego e dwell por zona
    zone_stats = {}
    for zone_id, group in df.groupby("zone_id"):
        total_visits    = len(group)
        unique_visitors = group["person_id"].nunique()

        # Dwell: só contamos linhas onde houve linger (dwell_s > 0)
        dwell_rows = group[group["dwell_s"] > 0]["dwell_s"]
        avg_dwell  = safe_round(dwell_rows.mean()) if len(dwell_rows) > 0 else 0
        has_linger_count = int((group["dwell_s"] > 0).sum())
        stop_rate  = safe_round(has_linger_count / unique_visitors) if unique_visitors > 0 else 0

        zone_stats[zone_id] = {
            "total_visits":     int(total_visits),
            "unique_visitors":  int(unique_visitors),
            "avg_dwell_s":      avg_dwell,
            "stop_rate":        stop_rate,        # 0.0 – 1.0
            "zone_type":        get_zone_prefix(zone_id),
        }

    # Ranking de zonas por tráfego
    ranked_zones = sorted(zone_stats.items(), key=lambda x: x[1]["total_visits"], reverse=True)
    traffic_ranking = [{"zone": z, "visits": s["total_visits"]} for z, s in ranked_zones]

    # Sequências mais frequentes (bi-gramas e tri-gramas entre zonas)
    bigrams  = Counter()
    trigrams = Counter()

    for (person, date), group in df.groupby(["person_id", "visit_date"]):
        zones = list(group.sort_values("entry_time")["zone_id"])
        for i in range(len(zones) - 1):
            bigrams[(zones[i], zones[i+1])] += 1
        for i in range(len(zones) - 2):
            trigrams[(zones[i], zones[i+1], zones[i+2])] += 1

    top_bigrams  = [{"sequence": list(k), "count": v}
                    for k, v in bigrams.most_common(10)]
    top_trigrams = [{"sequence": list(k), "count": v}
                    for k, v in trigrams.most_common(10)]

    return {
        "zone_stats":       zone_stats,
        "traffic_ranking":  traffic_ranking,
        "top_bigrams":      top_bigrams,
        "top_trigrams":     top_trigrams,
    }


# ─── 3. Funil de cliente ─────────────────────────────────────────────────────────

def calc_funnel_metrics(df: pd.DataFrame) -> dict:
    """
    O funil responde a uma pergunta simples:
    de todos os que entraram, quantos chegaram a cada zona?

    A ordem do funil é: Z_E → Z_N → Z_S → Z_C/Z_CK
    Calculamos também o perfil dos visitantes que não chegaram à caixa
    (potencial para perceber quem desiste e porquê).
    """
    print("  [3/5] Funil de cliente...")

    all_visitors = set(df["person_id"].unique())

    # Visitantes que passaram por cada tipo de zona
    def visitors_in_prefix(prefix: str) -> set:
        mask = df["zone_id"].apply(lambda z: get_zone_prefix(z) == prefix)
        return set(df[mask]["person_id"].unique())

    entered     = visitors_in_prefix("Z_E")
    in_corridor = visitors_in_prefix("Z_N")
    in_section  = visitors_in_prefix("Z_S")
    at_checkout = visitors_in_prefix("Z_C") | visitors_in_prefix("Z_CK")

    n_entered     = len(entered)
    n_corridor    = len(in_corridor & entered)
    n_section     = len(in_section & entered)
    n_checkout    = len(at_checkout & entered)

    # Taxa de conversão para caixa (dos que entraram pela porta)
    conversion_rate = safe_round(n_checkout / n_entered) if n_entered > 0 else 0

    # Perfil dos visitantes que NÃO chegaram à caixa
    did_not_checkout = entered - at_checkout
    dnc_df = df[df["person_id"].isin(did_not_checkout)]

    gender_dnc   = dnc_df.drop_duplicates("person_id")["gender"].value_counts().to_dict()
    age_dnc      = dnc_df.drop_duplicates("person_id")["age_range"].value_counts().to_dict()

    # Onde se perde o tráfego? Última zona de quem não chegou à caixa
    last_zone_dnc = (
        dnc_df.sort_values("entry_time")
        .groupby("person_id")["zone_id"]
        .last()
        .value_counts()
        .head(10)
        .to_dict()
    )
    last_zone_dnc = {str(k): int(v) for k, v in last_zone_dnc.items()}

    # Tempo médio de permanência dos que chegaram vs não chegaram à caixa
    def avg_time_in_store(person_ids: set) -> "float | None":
        sub = df[df["person_id"].isin(person_ids)]
        spans = sub.groupby(["person_id", "visit_date"]).agg(
            first=("entry_time", "min"), last=("exit_time", "max")
        )
        spans["dur"] = (spans["last"] - spans["first"]).dt.total_seconds() / 60
        valid = spans[(spans["dur"] > 0) & (spans["dur"] < 180)]
        return safe_round(valid["dur"].mean())

    avg_time_checkout     = avg_time_in_store(entered & at_checkout)
    avg_time_no_checkout  = avg_time_in_store(did_not_checkout)

    return {
        "funnel": {
            "entered_store":    int(n_entered),
            "reached_corridor": int(n_corridor),
            "reached_section":  int(n_section),
            "reached_checkout": int(n_checkout),
        },
        "conversion_rate_to_checkout": conversion_rate,
        "did_not_checkout": {
            "count":              int(len(did_not_checkout)),
            "gender_breakdown":   {str(k): int(v) for k, v in gender_dnc.items()},
            "age_breakdown":      {str(k): int(v) for k, v in age_dnc.items()},
            "last_zone_before_leaving": last_zone_dnc,
        },
        "avg_visit_min_buyers":      avg_time_checkout,
        "avg_visit_min_non_buyers":  avg_time_no_checkout,
    }



def calc_demographic_metrics(df: pd.DataFrame) -> dict:
    """
    Distribui visitantes por género e faixa etária, cruzando com:
    - hora do dia (para perceber quando entra cada perfil)
    - zona (para perceber onde cada perfil passa mais tempo)

    Apenas contamos cada person_id uma vez por combinação para
    não inflar os números de quem visita muitas zonas.
    """
    print("  [4/5] Segmentação demográfica...")

    # Um registo por pessoa (para distribuições globais)
    persons = df.drop_duplicates("person_id")[["person_id", "gender", "age_range"]].copy()

    gender_dist  = persons["gender"].value_counts().to_dict()
    age_dist     = persons["age_range"].value_counts().to_dict()

    # Distribuição de género por hora do dia
    # (usamos a primeira hora de entrada da pessoa nesse dia)
    first_hour = df.groupby(["person_id", "visit_date"])["hour_of_day"].min().reset_index()
    first_hour = first_hour.merge(persons[["person_id", "gender", "age_range"]], on="person_id")

    gender_by_hour = (
        first_hour.groupby(["hour_of_day", "gender"])["person_id"]
        .nunique()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )
    gender_by_hour = {int(k): {str(g): int(v) for g, v in vals.items()}
                      for k, vals in gender_by_hour.items()}

    age_by_hour = (
        first_hour.groupby(["hour_of_day", "age_range"])["person_id"]
        .nunique()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )
    age_by_hour = {int(k): {str(a): int(v) for a, v in vals.items()}
                   for k, vals in age_by_hour.items()}

    # Dwell time médio por segmento demográfico e por zona
    # Nota: df já tem colunas gender e age_range — não fazemos merge para evitar
    # duplicados (gender_x / gender_y). Usamos directamente as colunas existentes.
    dwell_df = df[df["dwell_s"] > 0][["person_id", "gender", "age_range", "zone_id", "dwell_s"]].copy()

    dwell_gender_zone = (
        dwell_df.groupby(["gender", "zone_id"])["dwell_s"]
        .mean()
        .round(1)
        .reset_index()
    )
    dwell_by_gender = defaultdict(dict)
    for _, row in dwell_gender_zone.iterrows():
        dwell_by_gender[str(row["gender"])][str(row["zone_id"])] = float(row["dwell_s"])

    dwell_age_zone = (
        dwell_df.groupby(["age_range", "zone_id"])["dwell_s"]
        .mean()
        .round(1)
        .reset_index()
    )
    dwell_by_age = defaultdict(dict)
    for _, row in dwell_age_zone.iterrows():
        dwell_by_age[str(row["age_range"])][str(row["zone_id"])] = float(row["dwell_s"])

    return {
        "global": {
            "gender_distribution": {str(k): int(v) for k, v in gender_dist.items()},
            "age_distribution":    {str(k): int(v) for k, v in age_dist.items()},
        },
        "gender_by_hour": gender_by_hour,
        "age_by_hour":    age_by_hour,
        "avg_dwell_by_gender_and_zone": dict(dwell_by_gender),
        "avg_dwell_by_age_and_zone":    dict(dwell_by_age),
    }



def calc_anomaly_metrics(df: pd.DataFrame) -> dict:
    """
    Metodologia do enunciado:
      - Para cada zona + hora, calcular média e desvio-padrão dos primeiros 6 dias
      - Identificar onde o dia 7 se desvia mais de 2σ

    O output inclui as anomalias ordenadas por magnitude do desvio,
    para que a LLM saiba imediatamente quais são as mais graves.

    Também calculamos anomalias a nível de dia inteiro (tráfego diário)
    para capturar padrões mais grosseiros.
    """
    print("  [5/5] Deteção de anomalias...")

    # Garantir que visit_date é datetime.date para podermos ordenar
    df = df.copy()
    df["visit_date"] = pd.to_datetime(df["visit_date"]).dt.date

    all_dates = sorted(df["visit_date"].unique())

    if len(all_dates) < 2:
        return {"error": "Dados insuficientes para deteção de anomalias (< 2 dias)"}

    # Os primeiros N-1 dias são a baseline; o último é o dia sob análise
    baseline_dates = all_dates[:-1]   # dias 1 a 6
    target_date    = all_dates[-1]    # dia 7

    # Contagem de visitantes únicos por zona + hora + dia
    hourly_zone = (
        df.groupby(["visit_date", "zone_id", "hour_of_day"])["person_id"]
        .nunique()
        .reset_index(name="visitors")
    )

    baseline_df = hourly_zone[hourly_zone["visit_date"].isin(baseline_dates)]
    target_df   = hourly_zone[hourly_zone["visit_date"] == target_date]

    # Média e desvio-padrão da baseline por zona + hora
    stats = (
        baseline_df.groupby(["zone_id", "hour_of_day"])["visitors"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    stats["std"] = stats["std"].fillna(0)  # std = NaN quando só há 1 dia

    # Juntar com o dia alvo
    merged = target_df.merge(stats, on=["zone_id", "hour_of_day"], how="left")
    merged["mean"] = merged["mean"].fillna(0)
    merged["std"]  = merged["std"].fillna(0)

    # Calcular z-score (desvio em número de sigmas)
    # Quando std = 0 e o valor muda, é uma anomalia perfeita — pontuamos alto
    def z_score(row):
        if row["std"] == 0:
            return 10.0 if row["visitors"] != row["mean"] else 0.0
        return abs(row["visitors"] - row["mean"]) / row["std"]

    merged["z_score"]    = merged.apply(z_score, axis=1)
    merged["direction"]  = merged.apply(
        lambda r: "acima" if r["visitors"] > r["mean"] else "abaixo", axis=1
    )

    # Filtrar anomalias > 2σ
    anomalies_df = merged[merged["z_score"] > 2.0].sort_values("z_score", ascending=False)

    anomalies = []
    for _, row in anomalies_df.iterrows():
        anomalies.append({
            "zone_id":        str(row["zone_id"]),
            "hour_of_day":    int(row["hour_of_day"]),
            "date_analyzed":  str(target_date),
            "visitors_day7":  int(row["visitors"]),
            "baseline_mean":  safe_round(row["mean"]),
            "baseline_std":   safe_round(row["std"]),
            "z_score":        safe_round(row["z_score"]),
            "direction":      str(row["direction"]),
            "baseline_days":  [str(d) for d in baseline_dates],
        })

    # Anomalias de tráfego diário total (visão macro)
    daily_visitors = (
        df.groupby("visit_date")["person_id"].nunique().reset_index(name="visitors")
    )
    daily_visitors["visit_date"] = pd.to_datetime(daily_visitors["visit_date"]).dt.date

    baseline_daily = daily_visitors[daily_visitors["visit_date"].isin(baseline_dates)]
    target_daily   = daily_visitors[daily_visitors["visit_date"] == target_date]

    daily_mean = baseline_daily["visitors"].mean()
    daily_std  = baseline_daily["visitors"].std() or 0
    target_visitors = int(target_daily["visitors"].values[0]) if len(target_daily) > 0 else 0

    daily_z = abs(target_visitors - daily_mean) / daily_std if daily_std > 0 else 0

    return {
        "analysis_target_date": str(target_date),
        "baseline_dates": [str(d) for d in baseline_dates],
        "anomalies_zone_hour": anomalies,
        "n_anomalies_found":   len(anomalies),
        "daily_traffic_anomaly": {
            "target_date":     str(target_date),
            "visitors":        target_visitors,
            "baseline_mean":   safe_round(daily_mean),
            "baseline_std":    safe_round(daily_std),
            "z_score":         safe_round(daily_z),
            "is_anomalous":    daily_z > 2.0,
        },
    }



def build_metrics(df: pd.DataFrame) -> dict:
    """
    Chama os cinco módulos e monta o dicionário final.
    Adiciona também metadados sobre o dataset para a LLM ter contexto.
    """
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"]  = pd.to_datetime(df["exit_time"])

    dates = sorted(df["visit_date"].unique())

    metadata = {
        "dataset_info": {
            "total_events":        len(df),
            "total_unique_persons": int(df["person_id"].nunique()),
            "total_zones":         int(df["zone_id"].nunique()),
            "date_range":          {"start": str(dates[0]), "end": str(dates[-1])},
            "days_covered":        len(dates),
            "store_hours":         "09:00–21:00",
        }
    }

    metrics = {}
    metrics.update(metadata)
    metrics["traffic"]     = calc_traffic_metrics(df)
    metrics["zones"]       = calc_zone_metrics(df)
    metrics["funnel"]      = calc_funnel_metrics(df)
    metrics["demographics"] = calc_demographic_metrics(df)
    metrics["anomalies"]   = calc_anomaly_metrics(df)

    return metrics


# ─── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Calcula todas as métricas analíticas a partir do journeys.csv"
    )
    parser.add_argument("--input",  required=True, help="Caminho para journeys.csv")
    parser.add_argument("--output", required=True, help="Caminho para metrics.json")
    args = parser.parse_args()

    print(f"\nA ler trajectórias de: {args.input}")
    df = pd.read_csv(args.input, parse_dates=["entry_time", "exit_time"])
    df.columns = df.columns.str.strip().str.lower()

    print(f"  → {len(df)} visitas a zonas  |  {df['person_id'].nunique()} pessoas únicas")
    print(f"  → {df['zone_id'].nunique()} zonas  |  {df['visit_date'].nunique()} dias\n")

    print("A calcular métricas...")
    metrics = build_metrics(df)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nFicheiro guardado em: {args.output}")

    # Resumo rápido para o developer
    n_anomalies = metrics["anomalies"].get("n_anomalies_found", 0)
    conv_rate   = metrics["funnel"].get("conversion_rate_to_checkout", 0)
    peak        = metrics["traffic"].get("peak_hour", "?")
    print(f"\n── Resumo ────────────────────────────────────────────")
    print(f"  Visitantes esta semana : {metrics['traffic']['total_visitors_week']}")
    print(f"  Hora de pico           : {peak}h")
    print(f"  Taxa de conversão      : {conv_rate:.1%}")
    print(f"  Anomalias encontradas  : {n_anomalies}")
    print(f"──────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()