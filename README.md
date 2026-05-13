# Retail Intelligence Pipeline: From Raw Detections to Real Intelligence

Este projeto implementa um pipeline completo de análise de dados para retalho, capaz de transformar eventos anónimos de visão computacional em relatórios estratégicos para gestão.

**UC:** Interação com Modelos de Larga Escala (LIACD)  
**Autor:** Simão Abranches  
**Modelo LLM:** qwen2.5:0.5b (via Ollama)

---

## 🏗 Estrutura do Projeto

O repositório segue a estrutura organizada exigida para a submissão técnica:

* `src/`: Scripts Python do pipeline (`stitcher`, `analytics`, `insights`, `report`).
* `data/`: Ficheiro de entrada `events.csv`.
* `output/`: Resultados gerados (`journeys.csv`, `metrics.json`, `insights.json`, `weekly_report.md`).
* `prompts/`: Estratégias de prompting (Zero-shot e Few-shot) em formato `.txt`.
* `evaluate.py`: Script de avaliação de performance e consistência.

---

## 🚀 Como Executar o Pipeline

O pipeline foi desenhado de forma modular. Para gerar o relatório final, executa os seguintes comandos no terminal:

1. **Reconstrução de Trajetórias (Stitching):**
   ```bash
   python src/stitcher.py --input data/events.csv
