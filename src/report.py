"""
report.py — gera o briefing semanal em Markdown a partir do insights.json

O relatório é desenhado para ser lido por um gestor de loja sem formação técnica.

estrutura:
  1. Resumo executivo (máx. 150 palavras)
  2. Performance de tráfego
  3. Análise de zonas
  4. Funil de clientes
  5. Anomalias da semana
  6. Recomendações para a próxima semana (máx. 5, ordenadas por urgência)

Executar com:
    python src/report.py --input output/insights.json --output output/weekly_report.md --strategy b --metrics output/metrics.json

Opções:
    --strategy   a|b  (qual estratégia usar, default: b)
    --metrics    caminho para metrics.json (opcional, enriquece o relatório com dados brutos)
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


# Helpers de formatação

def fmt_num(n) -> str:
    """Formata números com separador de milhares"""
    if n is None:
        return "N/D"
    try:
        return f"{int(n):,}".replace(",", ".")
    except (ValueError, TypeError):
        return str(n)


def fmt_pct(n) -> str:
    """0.382 → '38,2%'"""
    if n is None:
        return "N/D"
    try:
        return f"{float(n) * 100:.1f}%".replace(".", ",")
    except (ValueError, TypeError):
        return str(n)


def fmt_min(n) -> str:
    """Formata minutos de forma legível"""
    if n is None:
        return "N/D"
    try:
        mins = float(n)
        if mins >= 60:
            h   = int(mins // 60)
            m   = int(mins % 60)
            return f"{h}h {m}min"
        return f"{mins:.0f} min"
    except (ValueError, TypeError):
        return str(n)


def urgency_label(urgencia: str) -> str:
    """Converte o campo urgencia para emoji + texto legível."""
    mapping = {
        "imediata":     "Imediata",
        "esta_semana":  "Esta semana",
        "proximo_mes":  "Próximo mês",
    }
    return mapping.get(urgencia, urgencia)


def day_name(date_str: str) -> str:
    """'2026-03-10' → 'segunda-feira, 10 de março'"""
    try:
        dt = datetime.strptime(str(date_str), "%Y-%m-%d")
        days = ["segunda-feira", "terça-feira", "quarta-feira",
                "quinta-feira", "sexta-feira", "sábado", "domingo"]
        months = ["janeiro", "fevereiro", "março", "abril", "maio", "junho",
                  "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]
        return f"{days[dt.weekday()]}, {dt.day} de {months[dt.month - 1]}"
    except (ValueError, TypeError):
        return str(date_str)


# Secções do relatório

def section_header(metrics: dict) -> str:
    """Cabeçalho do documento com datas e contexto."""
    info = metrics.get("dataset_info", {}) if metrics else {}
    start = info.get("date_range", {}).get("start", "")
    end   = info.get("date_range", {}).get("end", "")

    date_range = ""
    if start and end:
        date_range = f"{day_name(start)} → {day_name(end)}"

    generated = datetime.now().strftime("%d/%m/%Y às %H:%M")

    return f"""# Relatório Semanal — Loja de Retalho

**Período:** {date_range or "Semana analisada"}  
**Gerado automaticamente em:** {generated}

---
"""


def section_executive_summary(insights_data: dict) -> str:
    """
    Secção 1 — Resumo executivo.
    Usa o resumo_executivo gerado pela LLM, ou constrói um a partir
    dos insights com maior confiança se o resumo estiver vazio.
    """
    resumo = insights_data.get("resumo_executivo", "")

    if isinstance(resumo, list):
        resumo = "\n".join(f"- {item}" for item in resumo)

    if not resumo:
        insights = insights_data.get("insights", [])
        top3 = sorted(insights, key=lambda x: x.get("confianca", 0), reverse=True)[:3]
        resumo = "\n".join(f"- {ins.get('titulo', '')}" for ins in top3)

    return f"""## 1. Resumo Executivo

{resumo}

---
"""


def section_traffic(insights_data: dict, metrics: dict) -> str:
    """Secção 2 — Performance de tráfego."""
    lines = ["## 2. Performance de Tráfego\n"]

    # Dados brutos do metrics.json
    if metrics:
        t = metrics.get("traffic", {})
        total   = fmt_num(t.get("total_visitors_week"))
        media   = fmt_num(t.get("daily_avg_visitors"))
        pico    = t.get("peak_hour")
        busiest = t.get("busiest_day")
        quietest= t.get("quietest_day")
        dur     = fmt_min(t.get("avg_visit_duration_min"))

        lines.append(f"| Métrica | Valor |")
        lines.append(f"|---------|-------|")
        lines.append(f"| Visitantes esta semana | **{total}** |")
        lines.append(f"| Média diária | {media} visitantes/dia |")
        lines.append(f"| Hora de maior afluência | {pico}h |")
        lines.append(f"| Dia mais movimentado | {day_name(busiest)} |")
        lines.append(f"| Dia menos movimentado | {day_name(quietest)} |")
        lines.append(f"| Duração média de visita | {dur} |")
        lines.append("")

        # Tabela de visitantes por dia
        vpd = t.get("visitors_per_day", {})
        if vpd:
            lines.append("**Afluência por dia:**\n")
            lines.append("| Dia | Visitantes |")
            lines.append("|-----|-----------|")
            for date, n in sorted(vpd.items()):
                lines.append(f"| {day_name(date)} | {fmt_num(n)} |")
            lines.append("")

    # Insights de tráfego da LLM
    traffic_insights = [
        ins for ins in insights_data.get("insights", [])
        if ins.get("categoria") == "trafego"
    ]
    if traffic_insights:
        lines.append("**Análise:**\n")
        for ins in traffic_insights:
            lines.append(f"> **{ins.get('titulo', '')}**  ")
            lines.append(f"> {ins.get('observacao', '')}  ")
            implicacao = ins.get('implicacao', '')
            if implicacao:
                lines.append(f"> *{implicacao}*")
            lines.append("")

    lines.append("---")
    return "\n".join(lines) + "\n"


def section_zones(insights_data: dict, metrics: dict) -> str:
    """Secção 3 — Análise de zonas."""
    lines = ["## 3. Análise de Zonas\n"]

    # Top 3 e bottom 3 do metrics.json
    if metrics:
        ranking = metrics.get("zones", {}).get("traffic_ranking", [])
        zone_stats = metrics.get("zones", {}).get("zone_stats", {})

        if ranking:
            top3  = ranking[:3]
            bot3  = [z for z in ranking[-3:] if z["zone"] not in [t["zone"] for t in top3]]

            lines.append("### Zonas com melhor performance\n")
            lines.append("| Zona | Visitas | Dwell médio | Taxa de paragem |")
            lines.append("|------|---------|-------------|-----------------|")
            for item in top3:
                zid   = item["zone"]
                stats = zone_stats.get(zid, {})
                dwell = f"{stats.get('avg_dwell_s', 0):.0f}s"
                stop  = fmt_pct(stats.get("stop_rate"))
                lines.append(f"| **{zid}** | {fmt_num(item['visits'])} | {dwell} | {stop} |")
            lines.append("")

            if bot3:
                lines.append("### Zonas com menor tráfego\n")
                lines.append("| Zona | Visitas | Dwell médio |")
                lines.append("|------|---------|-------------|")
                for item in bot3:
                    zid   = item["zone"]
                    stats = zone_stats.get(zid, {})
                    dwell = f"{stats.get('avg_dwell_s', 0):.0f}s"
                    lines.append(f"| {zid} | {fmt_num(item['visits'])} | {dwell} |")
                lines.append("")

    # Insights de zona da LLM
    zone_insights = [
        ins for ins in insights_data.get("insights", [])
        if ins.get("categoria") == "zona"
    ]
    if zone_insights:
        lines.append("### Observações e recomendações\n")
        for ins in zone_insights:
            lines.append(f"**{ins.get('titulo', '')}**\n")
            lines.append(f"{ins.get('observacao', '')}\n")
            implicacao = ins.get('implicacao', '')
            if implicacao:
                lines.append(f"*Implicação:* {implicacao}\n")
            rec = ins.get('recomendacao', '')
            if rec:
                lines.append(f"*Recomendação:* {rec}\n")

    lines.append("---")
    return "\n".join(lines) + "\n"


def section_funnel(insights_data: dict, metrics: dict) -> str:
    """Secção 4 — Funil de clientes."""
    lines = ["## 4. Funil de Clientes\n"]

    if metrics:
        f = metrics.get("funnel", {})
        funnel = f.get("funnel", {})
        conv   = f.get("conversion_rate_to_checkout")
        dnc    = f.get("did_not_checkout", {})

        if funnel:
            entered   = funnel.get("entered_store", 0)
            corridor  = funnel.get("reached_corridor", 0)
            section   = funnel.get("reached_section", 0)
            checkout  = funnel.get("reached_checkout", 0)

            # Calcular percentagens em relação aos que entraram
            def pct_of_entered(n):
                return f"{n / entered * 100:.1f}%" if entered > 0 else "N/D"

            lines.append("```")
            lines.append(f"  Entradas na loja  →  {fmt_num(entered)} visitantes (100%)")
            lines.append(f"         ↓")
            lines.append(f"  Corredores (Z_N)  →  {fmt_num(corridor)} ({pct_of_entered(corridor)})")
            lines.append(f"         ↓")
            lines.append(f"  Secções (Z_S)     →  {fmt_num(section)} ({pct_of_entered(section)})")
            lines.append(f"         ↓")
            lines.append(f"  Caixa (Z_C/Z_CK)  →  {fmt_num(checkout)} ({pct_of_entered(checkout)})")
            lines.append("```")
            lines.append("")
            lines.append(f"**Taxa de conversão para caixa: {fmt_pct(conv)}**\n")

        # Perfil de quem não chegou à caixa
        if dnc:
            n_dnc = dnc.get("count", 0)
            lines.append(f"**{fmt_num(n_dnc)} visitantes saíram sem passar pela caixa.**\n")

            last_zones = dnc.get("last_zone_before_leaving", {})
            if last_zones:
                top_zone = max(last_zones, key=last_zones.get)
                lines.append(
                    f"A zona onde mais visitantes desistiram foi **{top_zone}** "
                    f"({fmt_num(last_zones[top_zone])} pessoas).\n"
                )

            # Duração média compradores vs não compradores
            dur_buy  = fmt_min(f.get("avg_visit_min_buyers"))
            dur_nobuy = fmt_min(f.get("avg_visit_min_non_buyers"))
            lines.append(f"| | Duração média de visita |")
            lines.append(f"|---|---|")
            lines.append(f"| Visitantes que compraram | {dur_buy} |")
            lines.append(f"| Visitantes que não compraram | {dur_nobuy} |")
            lines.append("")

    # Insights de funil da LLM
    funnel_insights = [
        ins for ins in insights_data.get("insights", [])
        if ins.get("categoria") == "funil"
    ]
    if funnel_insights:
        lines.append("### Análise\n")
        for ins in funnel_insights:
            lines.append(f"**{ins.get('titulo', '')}**\n")
            lines.append(f"{ins.get('observacao', '')}\n")
            implicacao = ins.get('implicacao', '')
            if implicacao:
                lines.append(f"*Implicação:* {implicacao}\n")
            rec = ins.get('recomendacao', '')
            if rec:
                lines.append(f"*Recomendação:* {rec}\n")

    lines.append("---")
    return "\n".join(lines) + "\n"


def section_anomalies(insights_data: dict, metrics: dict) -> str:
    """Secção 5 — Anomalias da semana."""
    lines = ["## 5. Anomalias da Semana\n"]

    # Dados brutos das anomalias
    if metrics:
        a = metrics.get("anomalies", {})
        n_total   = a.get("n_anomalies_found", 0)
        target    = a.get("analysis_target_date", "")
        top_anom  = a.get("anomalies_zone_hour", [])[:5]

        if n_total == 0:
            lines.append("Nenhuma anomalia significativa detetada esta semana.\n")
        else:
            lines.append(
                f"Foram detetadas **{n_total} anomalias** no dia {day_name(target)} "
                f"face à média dos dias anteriores.\n"
            )

            if top_anom:
                lines.append("**Top anomalias por magnitude:**\n")
                lines.append("| Zona | Hora | Visitantes | Média habitual | Desvio | Direção |")
                lines.append("|------|------|-----------|---------------|--------|---------|")
                for anom in top_anom:
                    lines.append(
                        f"| {anom['zone_id']} "
                        f"| {anom['hour_of_day']}h "
                        f"| {anom['visitors_day7']} "
                        f"| {anom['baseline_mean']} "
                        f"| {anom['z_score']}σ "
                        f"| {anom['direction']} |"
                    )
                lines.append("")

    # Insights de anomalia da LLM
    anomaly_insights = [
        ins for ins in insights_data.get("insights", [])
        if ins.get("categoria") == "anomalia"
    ]
    if anomaly_insights:
        lines.append("### Análise detalhada\n")
        for ins in anomaly_insights:
            urgencia = urgency_label(ins.get("urgencia", ""))
            lines.append(f"**{ins.get('titulo', '')}** — {urgencia}\n")
            lines.append(f"{ins.get('observacao', '')}\n")
            implicacao = ins.get('implicacao', '')
            if implicacao:
                lines.append(f"*Causa provável:* {implicacao}\n")
            rec = ins.get('recomendacao', '')
            if rec:
                lines.append(f"*Ação recomendada:* {rec}\n")
    elif metrics and metrics.get("anomalies", {}).get("n_anomalies_found", 0) > 0:
        lines.append("*O modelo não gerou análise textual para as anomalias detetadas.*\n")

    lines.append("---")
    return "\n".join(lines) + "\n"


def section_recommendations(insights_data: dict) -> str:
    """
    Secção 6 — Recomendações para a próxima semana.
    Máximo 5, ordenadas por urgência (imediata > esta_semana > proximo_mes).
    """
    lines = ["## 6. Recomendações para a Próxima Semana\n"]

    all_insights = insights_data.get("insights", [])

    # Ordenar por urgência
    urgency_order = {"imediata": 0, "esta_semana": 1, "proximo_mes": 2}
    sorted_insights = sorted(
        all_insights,
        key=lambda x: (urgency_order.get(x.get("urgencia", "proximo_mes"), 2),
                       -x.get("confianca", 0))
    )

    # Top 5 com recomendação definida
    top5 = [ins for ins in sorted_insights if ins.get("recomendacao")][:5]

    if not top5:
        lines.append("*Nenhuma recomendação gerada.*\n")
    else:
        for i, ins in enumerate(top5, 1):
            urgencia = urgency_label(ins.get("urgencia", ""))
            titulo   = ins.get("titulo", "")
            rec      = ins.get("recomendacao", "")
            lines.append(f"### {i}. {titulo}")
            lines.append(f"**Urgência:** {urgencia}  ")
            lines.append(f"{rec}\n")

    lines.append("---")
    lines.append(
        "*Relatório gerado automaticamente pelo pipeline de retail intelligence. "
        "Os insights são baseados em dados de visão computacional e devem ser "
        "validados com contexto operacional antes de atuar.*"
    )
    return "\n".join(lines) + "\n"


# Orquestração principal

def build_report(insights_data: dict, metrics: dict) -> str:
    """Junta todas as secções num único documento Markdown."""
    sections = [
        section_header(metrics),
        section_executive_summary(insights_data),
        section_traffic(insights_data, metrics),
        section_zones(insights_data, metrics),
        section_funnel(insights_data, metrics),
        section_anomalies(insights_data, metrics),
        section_recommendations(insights_data),
    ]
    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(
        description="Gera o briefing semanal em Markdown a partir do insights.json"
    )
    parser.add_argument("--input",    required=True, help="Caminho para insights.json")
    parser.add_argument("--output",   required=True, help="Caminho para weekly_report.md")
    parser.add_argument("--strategy", default="b", choices=["a", "b"],
                        help="Qual estratégia de insights usar (default: b — few-shot)")
    parser.add_argument("--metrics",  default=None,
                        help="Caminho para metrics.json (opcional, enriquece o relatório)")
    args = parser.parse_args()

    print(f"\nA ler insights de: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        insights_full = json.load(f)

    # Seleciona a estratégia pedida
    strategy_key = (
        "estrategia_B_few_shot" if args.strategy == "b"
        else "estrategia_A_zero_shot"
    )
    insights_data = insights_full.get(strategy_key, {})

    # Se não houver a estratégia pedida, tenta a outra
    if not insights_data.get("insights"):
        fallback_key = "estrategia_A_zero_shot" if args.strategy == "b" else "estrategia_B_few_shot"
        insights_data = insights_full.get(fallback_key, {})
        if insights_data:
            print(f"  Estratégia {args.strategy} não encontrada — a usar fallback.")

    n_insights = len(insights_data.get("insights", []))
    print(f"  → {n_insights} insights carregados (estratégia {args.strategy})")

    # Carrega metrics.json
    metrics = None
    if args.metrics:
        try:
            with open(args.metrics, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            print(f"  Métricas carregadas de: {args.metrics}")
        except FileNotFoundError:
            print(f"  metrics.json não encontrado em {args.metrics} — relatório sem dados brutos.")

    print("\nA gerar relatório...")
    report = build_report(insights_data, metrics)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Relatório guardado em: {args.output}")
    print(f"  {len(report.splitlines())} linhas  |  {len(report)} caracteres\n")


if __name__ == "__main__":
    main()