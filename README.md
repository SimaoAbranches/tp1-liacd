# Retail Intelligence Pipeline: From Raw Detections to Real Intelligence

O objetivo deste projeto é o desenvolvimento de um sistema capaz de converter dados brutos de sensores de visão computacional, que registam presenças anónimas em zonas de uma loja, em inteligência de negócio acionável.
O desafio central reside na fragmentação dos dados: os sensores captam eventos, mas não a identidade contínua dos clientes. O pipeline desenvolvido aborda este problema através de uma arquitetura modular que separa o processamento algorítmico da interpretação semântica realizada por modelos de linguagem de larga escala (LLMs).

**UC:** Interação com Modelos de Larga Escala 
**Autor:** Simão Abranches nº53682
**Modelo LLM:** qwen2.5:0.5b (via Ollama)


## Estrutura do Projeto

O repositório segue a estrutura organizada exigida para a submissão técnica:

* `src/`: Scripts Python do pipeline (`stitcher`, `analytics`, `insights`, `report`).
* `data/`: Ficheiro de entrada `events.csv`.
* `output/`: Resultados gerados (`journeys.csv`, `metrics.json`, `insights.json`, `weekly_report.md`).
* `prompts/`: Estratégias de prompting (Zero-shot e Few-shot) em formato `.txt`.
* `evaluate.py`: Script de avaliação de performance e consistência.


## Como Executar o Pipeline

O pipeline foi desenhado de forma modular. Para gerar o relatório final, executa os seguintes comandos no terminal:

1. **Reconstrução de Trajetórias (Stitching):**
   ```bash
   python src/stitcher.py --input data/events.csv --output output/journeys.csv
2. **Pipeline Analítico:**
   ```bash
   python src/analytics.py --input output/journeys.csv --output output/metrics.json
3. **Motor de Insights (LLM):**
   ```bash
   python src/insights.py --input output/metrics.json --output output/insights.json
4. **Geração do Relatório Final (Markdown):**
   ```bash
   python src/report.py --input output/insights.json --output output/weekly_report.md --metrics output/metrics.json

**Avaliação e Reprodutibilidade**

Para validar a performance e a consistência do pipeline, utiliza o seguinte comando:
```bash
python evaluate.py --data data/events.csv --output output/evaluation_report.json
```
## Configuração do Modelo Local

Este projeto utiliza o Ollama para garantir que os dados permanecem locais e privados.

Modelo: qwen2.5:0.5b

Comando para descarregar: ollama pull qwen2.5:0.5b
