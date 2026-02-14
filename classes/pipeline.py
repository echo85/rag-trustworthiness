from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class ExperimentPipeline:
    """Pipeline for running RAG trustworthiness experiments"""
    
    def run_evaluation_accountability(self, sample: Dict) -> Dict:
        """Run single experiment evaluation"""
        question = sample["question"]
        correct_answer = sample["correct_answer"]
        documents = sample["documents"]
        rag_output = sample["rag_output"]
        logger.info("Evaluating Accountability...")
        evaluation = {}
        for(model, output_model) in rag_output.items(): 
            evaluation[model] = {}
            for(strategy, output) in output_model.items(): 
                evaluation[model][strategy] = {}
                evaluation[model][strategy]['accountability'] = self.evaluator.evaluate_accountability(
                    output["answer"], output["citations"], documents
                )
        return {
            "question": question,
            "correct_answer": correct_answer,
            "distractors_numbers": sample["distractors_numbers"],
            "documents": sample["documents"],
            "rag_output": rag_output,
            "evaluation": evaluation
        }
    
    def run_evaluation_robustness(self, sample: Dict) -> Dict:
        """Run single experiment evaluation"""
        question = sample["question"]
        correct_answer = sample["correct_answer"]
        documents = sample["documents"]
        rag_output = sample["rag_output"]
        logger.info("Evaluating Robustness...")
        evaluation = {}
        for(model, output_model) in rag_output.items(): 
            evaluation[model] = {}
            for(strategy, output) in output_model.items(): 
                evaluation[model][strategy] = {}
                evaluation[model][strategy]['robustness'] = self.evaluator.evaluate_robustness(
                    output["answer"], correct_answer, documents)
        return {
            "question": question,
            "correct_answer": correct_answer,
            "distractors_numbers": sample["distractors_numbers"],
            "documents": sample["documents"],
            "rag_output": rag_output,
            "evaluation": evaluation
        }

    def run_evaluation_factuality(self, sample: Dict) -> Dict:
        """Run single experiment evaluation"""
        question = sample["question"]
        correct_answer = sample["correct_answer"]
        fake_answer = sample["fake_answer"]
        documents = sample["documents"]
        rag_output = sample["rag_output"]
        logger.info("Evaluating Models...")
        evaluation = {}
        for(model, output_model) in rag_output.items(): 
            evaluation[model] = {}
            for(strategy, output) in output_model.items(): 
                evaluation[model][strategy] = {}
                evaluation[model][strategy]['factuality'] = self.evaluator.evaluate_factuality(
                    question, output["answer"], correct_answer, fake_answer, documents
                )
        
        return {
            "question": question,
            "fake_answer": fake_answer,
            "correct_answer": correct_answer,
            "poison_ratio": sample["poison_ratio"],
            "documents": documents,
            "rag_output": rag_output,
            "evaluation": evaluation
        }

    def prepare_dataset(self, poisoner, samples: List[Dict], poison_ratios: List[float], distractors: List[int]) -> List[Dict]:
        """
        Stage 1: Pre-generate all poisoned documents and fake answers using the poisoner model.
        This runs ONLY ONCE before the RAG generation loop.
        """
        self.poisoner = poisoner
        prepared_data = []
        
        total_ops = len(samples) * (len(poison_ratios) + len(distractors))
        current_op = 0
        
        logger.info(f"Starting Dataset Preparation for {len(samples)} samples...")

        for sample_idx, sample in enumerate(samples):
            question = sample["question"]
            correct_answer = sample["answer"]
            context = sample["context"]
            distractor_docs = sample.get("distractors", [])

            # 1. Factuality (Poisoning) Cases
            for poison_ratio in poison_ratios:
                current_op += 1
                logger.info(f"Prep [{current_op}/{total_ops}] Sample {sample_idx+1} (Poison: {poison_ratio:.0%})")
                
                # Generate poisoned docs
                documents = self.poisoner.poison_documents(
                    question, correct_answer, context, poison_ratio
                )
                # Generate fake answer
                fake_answer = self.poisoner.generate_fake_answer(
                    question, correct_answer, documents
                ).strip()

                prepared_data.append({
                    "experiment": "factuality",
                    "question": question,
                    "correct_answer": correct_answer,
                    "fake_answer": fake_answer,
                    "poison_ratio": poison_ratio,
                    "documents": documents,
                    "use_citations": False
                })

            # 2. Robustness & Accountability (Distractor) Cases
            for dist_num in distractors:
                current_op += 1
                logger.info(f"Prep [{current_op}/{total_ops}] Sample {sample_idx+1} (Distractors: {dist_num})")
                
                documents = self.poisoner.irrelevants_documents(
                    question, context, distractor_docs, dist_num
                )
                
                # Robustness Case (No citations)
                prepared_data.append({
                    "experiment": "robustness",
                    "question": question,
                    "correct_answer": correct_answer,
                    "distractors_numbers": dist_num,
                    "documents": documents,
                    "use_citations": False
                })
                
                # Accountability Case (With citations)
                prepared_data.append({
                    "experiment": "accountability",
                    "question": question,
                    "correct_answer": correct_answer,
                    "distractors_numbers": dist_num,
                    "documents": documents,
                    "use_citations": True
                })
        
        return prepared_data

    def run_rag_generation(self, rag_system, prepared_samples: List[Dict]) -> List[Dict]:
        """
        Stage 2: Run RAG inference on the pre-prepared dataset.
        This is called multiple times (once per target model).
        """
        self.rag_system = rag_system
        results = []
        
        total = len(prepared_samples)
        for i, sample in enumerate(prepared_samples):
            logger.info(f"RAG Inference [{i+1}/{total}] Exp: {sample['experiment']}")
            
            # Run RAG
            rag_output = self.rag_system.generate_multi_strategy_answers(
                sample["question"], 
                sample["documents"], 
                sample["use_citations"]
            )
            
            # Clone sample and attach results
            result = sample.copy()
            result["rag_output"] = rag_output
            results.append(result)
            
        return results

    def run_full_evaluation(self, evaluator, samples: List[Dict]) -> List[Dict]:
        """Run evaluation across all samples"""
        self.evaluator = evaluator
        results = []
        total = len(samples)
        current = 0
        
        logger.info(f"Starting evaluation: {len(samples)} samples")
        
        for sample_idx, sample in enumerate(samples):
            current += 1
            if sample["experiment"] == 'factuality':
                logger.info(f"[{current}/{total}] Eval Sample {sample_idx+1}, Poison ratio: {sample.get('poison_ratio', 0):.0%}")
                result = self.run_evaluation_factuality(sample)
            elif sample["experiment"] == 'accountability':
                result = self.run_evaluation_accountability(sample)
                logger.info(f"[{current}/{total}] Eval Sample {sample_idx+1}, Distractors: {sample.get('distractors_numbers', 0)}")
            else:
                result = self.run_evaluation_robustness(sample)
                logger.info(f"[{current}/{total}] Eval Sample {sample_idx+1}, Distractors: {sample.get('distractors_numbers', 0)}")

            result['experiment'] = sample["experiment"]
            results.append(result)
            self._debug(sample, result)
        
        return results

    def _debug(self, sample, result):
        for model in result["rag_output"].keys():
            logger.info(f"MODEL: {model}")
            logger.info(f"Question: {sample['question']}")
            logger.info(f"Correct Answer: {sample['correct_answer']}")
            if 'fake_answer' in result: 
                print("\n===FACTUALITY===\n")
                logger.info(f"Poison Ratio: {sample['poison_ratio']}")
                for strategy in result["rag_output"][model].keys():
                    logger.info(f"→ STRATEGY PROMPT {strategy}")
                    logger.info(f"Question: {sample['question']}")
                    fact_score = result["evaluation"][model][strategy]["factuality"]["is_correct"]
                    logger.info(f"  → Fake Answer: {result['fake_answer']}")
                    logger.info(f"  → Correct Answer: {sample['correct_answer']}")
                    logger.info(f"  → Generated Answer: {result['rag_output'][model][strategy]['answer']}\n")
                    logger.info(f"  → Documents Used: {result['rag_output'][model][strategy]['num_documents']}")
                    logger.info(f"  → Factuality: {'🟢' if fact_score else '🔴'}")
                    logger.info(f"       → Evaluation Metrics: {result['evaluation'][model][strategy]}")
                    logger.info(f"       → Factual Judge Answer: {result['evaluation'][model][strategy]['factuality']['judgment']}\n")
            else:
                if(sample['experiment'] == 'accountability'): 
                    print("\n===ACCOUNTABILITY===\n")
                else:
                    print("\n===ROBUSTNESS===\n")
                logger.info(f"MODEL: {model}")
                logger.info(f"Distractors: {sample['distractors_numbers']}")
                for strategy in result["rag_output"][model].keys():
                    logger.info(f"→ STRATEGY PROMPT {strategy}")
                    logger.info(f"Question: {sample['question']}")
                    logger.info(f"  → Correct Answer: {sample['correct_answer']}")
                    logger.info(f"  → Documents Used: {result['rag_output'][model][strategy]['num_documents']}")
                    logger.info(f"  → Generated Answer: {result['rag_output'][model][strategy]['answer']}\n")
                    logger.info(f"      → Evaluation Metrics: {result['evaluation'][model][strategy]}")