# In RAG We Trust?

Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by grounding responses in external knowledge. However, ensuring trustworthiness regarding **factuality**, **robustness to noise**, and **accountability** remains a challenge

This project presents an experimental pipeline to evaluate RAG systems across these three key dimensions using the HotpotQA dataset. The framework evaluates multiple open-weights models (including Ministral 3, Llama 3.1, and Qwen 3) using various prompting strategies to measure their resilience against poisoned and irrelevant documents.

## Architecture

The system consists of a modular Python framework:
* **`classes/pipeline.py`**: Orchestrates the experiments, handling data preparation, RAG inference, and evaluation loops.
* **`classes/rag_evaluator.py`**: Computes metrics using LLM-as-a-judge (Gemma 3), METEOR, F1 Score and embedding similarities.
* **`classes/simple_rag.py`**: Implements the RAG logic with specific prompting strategies (Baseline, Verify, Critical, Hedge).
* **`classes/document_poisoner.py`** (implied): Injects false information and distractors into the context.
* **`classes/ollama_client.py`**: To orchestrate the connection with Ollama which runs inference on the LLM with the relative prompt.
* **`classes/data_loader.py`**: Designed to retrieve and restructure the HotpotQA dataset, it transforms raw multi-document data into a cleaner format where relevant facts are isolated from noise
  
## Features & Methodology
The framework evaluates RAG systems across three dimensions:

1.  **Factuality:** Alignment with real-world facts and truthfulness.
2.  **Robustness:** Ability to ignore noisy or irrelevant retrieved documents.
3.  **Accountability:** Accuracy of citations/attributions provided by the model.

### Prompt Strategies
The system compares four distinct prompting strategies:

| Strategy | Description |
| :--- | :--- |
| **Baseline** | Standard RAG instruction: "Answer based on provided documents." |
| **Critical** | Role-plays as a security analyst; explicitly ignores suspicious info. |
| **Hedge** | Explicit instruction to output `I_DECLINE_TO_ANSWER` if conflicts exist. |
| **Verify** | Chain-of-Thought approach requiring step-by-step consistency checks. |

### Chart Showcase

* **`trustworthiness.ipynb`**: Jupyter Notebook that shows the charts of the evaluated results.

### Report of the Project

[Link to Report](https://github.com/echo85/rag-trustworthiness/blob/main/report/report.pdf)

### Generation & Evaluation CLI

* **`cli_generation_parallel.py`**: The entry point for parallel data generation. It handles "poisoning" documents (Stage 1) and running RAG inference (Stage 2).
* **`cli_evaluation.py`**: The entry point for evaluating the generated RAG outputs against ground truth metrics.
  
## 🛠️ Installation & Requirements

### Prerequisites
* Python 3.10+
* **Ollama**: This project requires a running instance of Ollama for inference and judging.
    * Target Models: `granite4:3b`, `llama3.1:8b`, `ministral-3:14b`, `mistral-small3.2:24b`, `qwen3:30b-instruct` (they are configurables on cli_generation_parallel.py)
    * Judge/Poisoner Models: `gemma3:12b` (Poisoner), `gemma3:27b` (LLM Judge)

Command to install the requirements + downloading llm and run the RAG Generation (Tested on Runpod.io with Nvidia A40 and 16 workers)
```bash
chmod +x ./install.sh
install.sh 

