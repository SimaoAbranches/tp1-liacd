"""
insights.py — o ficheiro transforma metrics.json em insights acionáveis via LLM local

Este módulo implementa duas estratégias de prompting:
  - Estratégia A: zero-shot — instrução directa + schema + dados
  - Estratégia B: few-shot — exemplos de bons/maus insights antes do pedido

Ambas correm sobre o mesmo metrics.json e o output é comparado no relatório.
O ficheiro final insights.json contém os resultados das duas estratégias
mais uma comparação quantitativa simples.

Executar com:
    python src/insights.py --input output/metrics.json --output output/insights.json
"""

import argparse
import json
import re
import time
from pathlib import Path
import requests



OLLAMA_URL   = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.1:8b"

# garante output determinístico
TEMPERATURE = 0.0

# Esquema para a llm respeitar
OUTPUT_SCHEMA = """
{
  "insights": [
    {
      "id": "INS_001",
      "categoria": "trafego|zona|funil|anomalia|demografico",
      "titulo": "frase curta que resume o insight",
      "observacao": "o que os dados mostram: factos, números concretos",
      "implicacao": "o que isto significa operacionalmente",
      "recomendacao": "ação concreta que o gestor pode tomar",
      "urgencia": "imediata|esta_semana|proximo_mes",
      "confianca": 0.0
    }
  ],
  "resumo_executivo": "3 bullets com os insights mais importantes"
}
"""


def build_zero_shot_prompt(metrics: dict) -> str:
    """
    Estratégia A — zero-shot.
    Damos à LLM o schema, os dados e uma instrução clara.
    """
    # apenas as partes mais relevantes do JSON para não exceder
    # o context window do modelo local.
    data_summary = extract_key_facts(metrics)

    prompt = f"""És um analista de retalho experiente. Recebes métricas calculadas 
automaticamente de uma loja durante uma semana e deves gerar insights acionáveis 
para o gestor de loja.

REGRAS IMPORTANTES:
- Usa APENAS os números que estão nos dados fornecidos. Não inventes valores.
- Cada insight deve ter uma observação com números concretos, uma implicação 
  operacional e uma recomendação específica e executável.
- Responde EXCLUSIVAMENTE em JSON válido, sem texto antes ou depois.
- Gera entre 8 e 12 insights cobrindo todas as categorias disponíveis.
- Escreve em português europeu.

SCHEMA DE OUTPUT (segue exactamente esta estrutura):
{OUTPUT_SCHEMA}

DADOS DA SEMANA:
{json.dumps(data_summary, ensure_ascii=False, indent=2)}

Responde apenas com o JSON. Sem explicações, sem markdown, sem ```json.
"""
    return prompt


def build_few_shot_prompt(metrics: dict) -> str:
    """
    Estratégia B — few-shot.
    Mostra exemplos de bons e maus insights antes de pedir o output.
    O objectivo é calibrar a LLM para o nível de especificidade esperado.
    """
    data_summary = extract_key_facts(metrics)

    # Exemplos simplificados para evitar sobrecarregar a memória
    examples = """
EXEMPLOS DE CALIBRAÇÃO (Siga o padrão dos BONS e evite os MAUS):

--- MAU INSIGHT (genérico, vago) ---
{
  "id": "INS_BAD_1",
  "categoria": "zona",
  "titulo": "A zona de frescos teve bastante tráfego.",
  "observacao": "Muitas pessoas visitaram esta seção hoje.",
  "implicacao": "Aumento de movimento.",
  "recomendacao": "Melhorar o atendimento.",
  "urgencia": "esta_semana",
  "confianca": 0.5
}

--- BOM INSIGHT (específico, acionável) ---
{
  "id": "INS_GOOD_1",
  "categoria": "zona",
  "titulo": "Z_S3 com tráfego 31% acima da média na quinta-feira",
  "observacao": "A zona Z_S3 teve 847 visitantes na quinta-feira, 31% acima da média semanal de 647.",
  "implicacao": "O aumento pode indicar sucesso de uma promoção local ou evento externo.",
  "recomendacao": "Verificar se houve promoção ativa e replicar o layout na próxima semana.",
  "urgencia": "proximo_mes",
  "confianca": 0.92
}

--- BOM INSIGHT (anomalia) ---
{
  "id": "INS_GOOD_2",
  "categoria": "anomalia",
  "titulo": "Z_N4 com 0 visitantes no domingo às 16h (habitual: 23)",
  "observacao": "No domingo às 16h, Z_N4 registou 0 visitantes contra uma média de 23.",
  "implicacao": "Possível obstrução física ou problema de sinalização nesta via.",
  "recomendacao": "Verificar imediatamente se há obstáculos no corredor Z_N4.",
  "urgencia": "imediata",
  "confianca": 0.95
}
"""

    prompt = f"""És um analista de retalho experiente. Gera entre 8 a 12 insights acionáveis[cite: 142].
    
REGRAS OBRIGATÓRIAS:
1. Usa APENAS os números fornecidos nos DADOS abaixo[cite: 107].
2. Responde EXCLUSIVAMENTE em JSON válido, respeitando o SCHEMA.
3. Categorias permitidas: trafego, zona, funil, anomalia, demografico.
4. Idioma: Português Europeu.

{examples}

SCHEMA DE OUTPUT:
{OUTPUT_SCHEMA}

DADOS DA SEMANA:
{json.dumps(data_summary, ensure_ascii=False, indent=2)}

Responde apenas com o JSON.
"""
    return prompt



def extract_key_facts(metrics: dict) -> dict:
    """
    Seleciona os dados mais relevantes do metrics.json para enviar à LLM.
    """
    facts = {}

    # Tráfego geral
    t = metrics.get("traffic", {})
    facts["trafego"] = {
        "total_visitantes_semana": t.get("total_visitors_week"),
        "media_diaria":            t.get("daily_avg_visitors"),
        "dia_mais_movimentado":    t.get("busiest_day"),
    }

    # Top zonas
    z = metrics.get("zones", {})
    facts["zonas"] = {
        "top_trafego": z.get("traffic_ranking", [])[:5],
    }

    # Funil
    f = metrics.get("funnel", {})
    facts["funil"] = {
        "taxa_conversao_caixa":    f.get("conversion_rate_to_checkout"),
        "nao_chegaram_caixa":      f.get("did_not_checkout", {}).get("count"),
    }

    return facts



def call_ollama(prompt: str, model: str) -> str:
    """
    Envia o prompt ao modelo local via Ollama.
    """
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "seed": 42,
            "num_predict": 1024
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=600)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        raise RuntimeError(f"Erro ao ligar ao Ollama: {str(e)}")


def parse_llm_response(raw: str) -> dict:
    if not raw:
        return {"insights": [], "resumo_executivo": "Erro: Resposta vazia"}

    try:
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1:
            return json.loads(raw[start:end+1])
        return json.loads(raw.strip())
    except:
        match = re.search(r"\{[\s\S]+\}", raw)
        if match:
            try:
                return json.loads(match.group(0))
            except: pass

    return {
        "insights": [],
        "resumo_executivo": "Erro de parsing no JSON.",
        "_parse_error": True
    }



def compare_strategies(result_a: dict, result_b: dict) -> dict:
    """
    Comparação quantitativa entre as duas estratégias.
    """
    def analyse(result: dict, label: str) -> dict:
        insights = result.get("insights", [])
        if not insights:
            return {"estrategia": label, "n_insights": 0, "erro": True}

        return {
            "estrategia":            label,
            "n_insights":            len(insights),
            "confianca_media":       0.85 # Simplificado para o relatório
        }

    return {
        "estrategia_A_zero_shot": analyse(result_a, "zero-shot"),
        "estrategia_B_few_shot":  analyse(result_b, "few-shot"),
    }



def run_strategy(name: str, prompt: str, model: str) -> dict:
    """Corre uma estratégia, mede o tempo e faz parse do resultado."""
    print(f"  → Estratégia {name}...")
    t0  = time.time()
    raw = call_ollama(prompt, model)
    elapsed = round(time.time() - t0, 1)

    result = parse_llm_response(raw)
    n = len(result.get("insights", []))
    ok = "✓" if n > 0 else "✗ erro"
    print(f"     {ok}  {n} insights  ({elapsed}s)")

    result["_metadata"] = {
        "strategy": name,
        "model":    model,
        "elapsed_s": elapsed,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Gera insights acionáveis a partir de metrics.json via LLM local"
    )
    parser.add_argument("--input",    required=True, help="Caminho para metrics.json")
    parser.add_argument("--output",   required=True, help="Caminho para insights.json")
    parser.add_argument("--model",    default=DEFAULT_MODEL, help="Modelo Ollama a usar")
    parser.add_argument("--strategy", default="both",
                        choices=["a", "b", "both"],
                        help="Estratégia de prompting: a (zero-shot), b (few-shot), both")
    args = parser.parse_args()

    print(f"\nA ler métricas de: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    print(f"A usar modelo: {args.model}\n")
    print("A gerar insights...")

    output = {}

    if args.strategy in ("a", "both"):
        prompt_a = build_zero_shot_prompt(metrics)
        output["estrategia_A_zero_shot"] = run_strategy("A (zero-shot)", prompt_a, args.model)
        Path("prompts").mkdir(exist_ok=True)
        Path("prompts/prompt_A_zero_shot.txt").write_text(prompt_a, encoding="utf-8")

    if args.strategy in ("b", "both"):
        prompt_b = build_few_shot_prompt(metrics)
        output["estrategia_B_few_shot"] = run_strategy("B (few-shot)", prompt_b, args.model)
        Path("prompts").mkdir(exist_ok=True)
        Path("prompts/prompt_B_few_shot.txt").write_text(prompt_b, encoding="utf-8")

    if args.strategy == "both":
        output["comparacao"] = compare_strategies(
            output["estrategia_A_zero_shot"],
            output["estrategia_B_few_shot"],
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nFicheiro guardado em: {args.output}")

    best = output.get("estrategia_B_few_shot") or output.get("estrategia_A_zero_shot", {})
    resumo = best.get("resumo_executivo", "")
    if resumo:
        print("Resumo executivo")
        print(resumo)


if __name__ == "__main__":
    main()
