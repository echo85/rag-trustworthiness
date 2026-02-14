import argparse
import concurrent.futures
import math
from typing import List, Dict
import logging

from classes.ollama_client import OllamaClient
from classes.config import Config
from classes.utils import Utils
from classes.data_loader import HotpotQADataLoader
from classes.document_poisoner import DocumentPoisoner
from classes.pipeline import ExperimentPipeline
from classes.simple_rag import SimpleRAG

# --- WORKER: PHASE 1 (PREPARATION) ---
def process_preparation_chunk(chunk_id: int, 
                              data_slice: List[Dict], 
                              config: Config, 
                              base_url: str) -> List[Dict]:
    """
    Generates poisoned documents and fake answers using the poison model.
    """
    if not data_slice:
        return []

    print(f"   [Prep-Worker {chunk_id}] Preparing {len(data_slice)} original samples...")
    
    client = OllamaClient(base_url)
    poisoner = DocumentPoisoner(client, config.poison_model)
    pipeline = ExperimentPipeline()

    # Pre-generate all document variations
    prepared_data = pipeline.prepare_dataset(
        poisoner, 
        data_slice, 
        config.poison_ratios, 
        config.distractors
    )
    
    print(f"   [Prep-Worker {chunk_id}] ✓ Done. Generated {len(prepared_data)} test cases.")
    return prepared_data

# --- WORKER: PHASE 2 (INFERENCE) ---
def process_inference_chunk(chunk_id: int, 
                            model_name: str, 
                            prepared_slice: List[Dict], 
                            config: Config, 
                            base_url: str) -> List[Dict]:
    """
    Runs RAG generation on ready-made data.
    """
    if not prepared_slice:
        return []

    print(f"   [RAG-Worker {chunk_id}] Processing {len(prepared_slice)} test cases...")
    
    client = OllamaClient(base_url)
    rag_system = SimpleRAG(client, [model_name], config.strategies)
    pipeline = ExperimentPipeline()

    # Run RAG only
    results = pipeline.run_rag_generation(rag_system, prepared_slice)
    
    print(f"   [RAG-Worker {chunk_id}] ✓ Batch finished.")
    return results

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run Optimized Parallel Generation")
    parser.add_argument("--model", type=str, help="Specific model to run")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel threads (default: 4)")
    args = parser.parse_args()

    # CONFIGURATION
    utils = Utils()
    config = Config()
    
    if args.model:
        target_models = [args.model]
    else:
        target_models = ["granite4:3b", "llama3.1:8b", "ministral-3:14b", "mistral-small3.2:24b","qwen3:30b-instruct"]

    config.poison_model = "gemma3:12b"
    config.num_samples = 30
    config.poison_ratios = [0.0, 0.3]
    config.distractors = [0, 8, 20]
    config.strategies = ["baseline", "verify", "hedge", "critical"]
    
    if not OllamaClient(config.ollama_base_url).check_connection():
        print(f"❌ Error: Could not connect to Ollama at {config.ollama_base_url}")
        return

    # LOAD DATA
    loader = HotpotQADataLoader(split='validation', limit=config.num_samples)
    all_data = loader.load()
    print(f"✓ Loaded {len(all_data)} original samples")
    print(f"\n{'='*60}")
    print(f"PHASE 1: PREPARING DATASET (Poisoning with {config.poison_model})")
    print(f"{'='*60}")
    
    # Split raw data for preparation
    chunk_size = math.ceil(len(all_data) / args.workers)
    prep_chunks = [all_data[i:i + chunk_size] for i in range(0, len(all_data), chunk_size)]
    
    full_prepared_dataset = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_chunk = {
            executor.submit(process_preparation_chunk, i, chunk, config, config.ollama_base_url): i 
            for i, chunk in enumerate(prep_chunks)
        }
        for future in concurrent.futures.as_completed(future_to_chunk):
            try:
                full_prepared_dataset.extend(future.result())
            except Exception as exc:
                print(f"❌ Prep Worker generated exception: {exc}")

    print(f"\n✓ Dataset Prepared. Total test cases: {len(full_prepared_dataset)}")

    # RAG GENERATION (Target Models)
    final_results = []
    
    # Split prepared data for RAG workers
    # (Recalculate chunks based on the expanded dataset size)
    rag_chunk_size = math.ceil(len(full_prepared_dataset) / args.workers)
    rag_chunks = [full_prepared_dataset[i:i + rag_chunk_size] for i in range(0, len(full_prepared_dataset), rag_chunk_size)]

    for current_model in target_models:
        print(f"\n{'='*60}")
        print(f"🚀 PHASE 2: PROCESSING MODEL: {current_model}")
        print(f"{'='*60}")

        model_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_chunk = {
                executor.submit(process_inference_chunk, i, current_model, chunk, config, config.ollama_base_url): i 
                for i, chunk in enumerate(rag_chunks)
            }
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                try:
                    model_results.extend(future.result())
                except Exception as exc:
                    print(f"❌ RAG Worker generated exception: {exc}")

        final_results.extend(model_results)
        print(f"✓ Completed {current_model}. (Saved checkpoint)")

    # FINAL SAVE
    print(f"\n✓ All models finished. Saving merged file...")
    out_path = utils.save(final_results, True)
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()