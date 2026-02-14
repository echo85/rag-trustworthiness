from classes.ollama_client import OllamaClient
from classes.config import Config
from classes.utils import Utils
from classes.rag_evaluator import RAGEvaluator
from classes.pipeline import ExperimentPipeline
import argparse 
import os
import logging
def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run RAG evaluation experiments.")
    parser.add_argument(
        "--path", 
        type=str, 
        default="./data/data_20260208_185104.json",
        help="Path to the custom JSON data file"
    )
    args = parser.parse_args()

    utils = Utils()
    config = Config()
    config.judge_model = "gemma3:27b"
    config.judge_method = "llm"
    results = utils.load(args.path)

    ollama = OllamaClient(config.ollama_base_url)
    if not ollama.check_connection():
        print("⚠ Warning: Could not connect to Ollama. Make sure it's running.")
    
    # EVALUATION METRICS
    evaluator = RAGEvaluator(ollama, config.judge_model, config.judge_method)
    print("✓ Evaluator initialized")
    
    # Instantiate pipeline and run experiments
    pipeline = ExperimentPipeline()
    print("RUNNING EVALUATION EXPERIMENTS")
    results = pipeline.run_full_evaluation(evaluator, results)
    print(f"\n✓ Completed {len(results)} Evaluations")
    out_path = utils.save(results,False,os.path.splitext(os.path.basename(args.path))[0])
    print(f"Saved results to {out_path}")

if __name__ == '__main__':
    main()