from classes.ollama_client import OllamaClient
from sentence_transformers import CrossEncoder
import torch
import torch.nn.functional as F
import evaluate
import string
import re
import numpy as np
from collections import Counter
from typing import List, Dict
from sentence_transformers import CrossEncoder, SentenceTransformer, util

class RAGEvaluator:
    """Evaluates RAG outputs across multiple dimensions"""
        
    def __init__(self, ollama_client: OllamaClient, judge_model: str, judge_method: str):
        self.ollama = ollama_client
        self.judge_model = judge_model
        self.judge_method = judge_method
        self.nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
        self.sim_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.meteor = evaluate.load('meteor')
        self.rouge = evaluate.load('rouge')

    def evaluate_accountability(self, generated_answer: str, gold_citations: List[int], documents: List[Dict]) -> Dict:
        """
        Evaluate Accountability (Knowledge Attribution).
        Calculates Citation Precision: % of citations that actually support the generated answer.
        """

        prediction_tokens = self.normalize_answer(generated_answer)
        # Check if model declined to answer 
        decline_patterns = [
            "cannot answer", "ideclinetoanswer", "unreliable context","i decline to answer"
        ]
        declined = any(pattern in prediction_tokens for pattern in decline_patterns)
        # Check if fake answer appears in response
        if(declined):
         return {
               "citation_precision": 1.0,
                "citation_recall": 1.0, 
                "citation_f1": 1.0
            }
        citations_strings = re.findall(r'\[([\d\s\-,]+)\]', generated_answer)
        pred_citations = []
        for cit_str in citations_strings:
            # Split by comma first to handle "1, 5-6"
            parts = cit_str.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part:
                    # Handle ranges like "5-6"
                    try:
                        start, end = map(int, part.split('-'))
                        # Create a range (inclusive of end), convert to 0-based index
                        pred_citations.extend([i - 1 for i in range(start, end + 1)])
                    except ValueError:
                        continue # Skip malformed ranges
                elif part.isdigit():
                    # Handle single numbers
                    pred_citations.append(int(part) - 1)

        if not pred_citations:
            precision = recall = f1 = 0.0
        else:
            num_same = len(set(gold_citations) & set(pred_citations))
            if num_same == 0:
                precision = recall = f1 = 0.0
            else:
                precision = num_same / len(pred_citations)
                recall = num_same / len(gold_citations)
                f1 = 2 * precision * recall / (precision + recall)

       
        return {
            "citation_precision": precision,
            "citation_recall": recall, 
            "citation_f1": f1
        }
    
    def evaluate_factuality(self, question: str, generated_answer: str, 
                          correct_answer: str, fake_answer: str, documents: List[Dict]) -> Dict:
        """
        Evaluate if the answer is factually correct
        Uses LLM-as-judge approach
        """
        # Check if model declined to answer 
        decline_patterns = [
            "cannot answer", "ideclinetoanswer", "unreliable context","i decline to answer"
        ]
        declined = any(pattern in self.normalize_answer(generated_answer) for pattern in decline_patterns)

        if(declined):
            meteor_score = 1.0
            factuality_score = 1
            is_correct = True
            judgment = "declined"
        else:
            factuality_match = self.normalize_answer(fake_answer) in self.normalize_answer(generated_answer)
            if(factuality_match):
                factuality_score = 0
            else:
                factuality_score = 1

            if(self.judge_method == "llm"):
            # LLM-based evaluation
                prompt = (f"""Evaluate if the generated answer is factually correct."
                "Question: {question}"
                "Correct Answer: {correct_answer}"
                "Generated Answer: {generated_answer}"
                "Is the generated answer factually correct? Consider:"
                "1. Does it match the correct answer (even if phrased differently)?"
                "2. Does it contain the key information?"
                "3. Is it free from major factual errors?"
                "Respond with ONLY 'CORRECT' or 'INCORRECT':""")

                judgment = self.ollama.generate(self.judge_model, prompt, temperature=0, max_tokens=128).strip()
                is_correct = "CORRECT" in judgment.split("\n")[0].upper() and "INCORRECT" not in judgment.split("\n")[0].upper()
            else:
                normalized_correct_answer = self.normalize_answer(correct_answer)
                normalized_generated_answer = self.normalize_answer(generated_answer)
                input_pair = [[normalized_correct_answer, normalized_generated_answer]]
                # Label 0: Contradiction (Hallucination)
                # Label 1: Entailment (Grounded)
                # Label 2: Neutral
                scores = self.nli_model.predict(input_pair)
                logits_tensor = torch.tensor(scores)
                predicted_index = np.argmax(logits_tensor)
                judgment = logits_tensor.tolist()
                is_correct = True if predicted_index != 0 else False

            meteor_score = float(self.meteor.compute(predictions=[generated_answer], references=[correct_answer])['meteor'])


        return {
            "meteor": meteor_score,
            "is_correct": is_correct,
            "score": factuality_score,
            "judgment": judgment
        }
    
    def evaluate_robustness(self, generated_answer: str, correct_answer: str, documents: List[Dict]) -> Dict:
        """
        Calculates F1 and Recall based on token overlap between the generated 
        answer and the correct answer, as per standard RAG benchmarking.
        """
        
      
        prediction_tokens = self.normalize_answer(generated_answer)
        # Check if model declined to answer 
        decline_patterns = [
            "cannot answer", "ideclinetoanswer", "unreliable context","i decline to answer"
        ]
        declined = any(pattern in prediction_tokens for pattern in decline_patterns)
        # Check if fake answer appears in response
        if(declined):
         return {
                "is_correct": True,
                "similarity_score": 1.0,
                "meteor": 1.0,
                "f1_score": 1.0,
                "recall": 1.0,
                "precision": 1.0,
                "refusal": True
            }
        
        
        # LLM-based evaluation
        prompt = (f"""Evaluate if the generated answer is similar to the correct answer."
                "Correct Answer: {correct_answer}"
                "Generated Answer: {generated_answer}"
                "Is the generated answer correct or is there any noise? Consider:"
                "1. Does it match the correct answer (even if phrased differently)?"
                "2. Does it contain the key information?"
                "3. Is it free from major noisy errors?"
                "Respond with ONLY 'CORRECT' or 'INCORRECT':""")

        judgment = self.ollama.generate(self.judge_model, prompt, temperature=0, max_tokens=128).strip()
        is_correct = "CORRECT" in judgment.split("\n")[0].upper() and "INCORRECT" not in judgment.split("\n")[0].upper()

        # Normalize both answers
        prediction_tokens = prediction_tokens.split()
        ground_truth_tokens = self.normalize_answer(correct_answer).split()
        embeddings = self.sim_model.encode(
            [generated_answer, correct_answer], 
            convert_to_tensor=True)
        # Calculate Cosine Similarity (-1 to 1)
        similarity_score = util.cos_sim(embeddings[0], embeddings[1]).item()
        # Calculate Common Tokens
        common_tokens = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common_tokens.values())

        # Calculate Precision, Recall, and F1
        if len(prediction_tokens) == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = 1.0 * num_same / len(prediction_tokens)
            
        if len(ground_truth_tokens) == 0:
            recall = 0
            f1 = 0
        else:
            recall = 1.0 * num_same / len(ground_truth_tokens)

        if (precision + recall) == 0:
            f1 = 0
        else:
            f1 = (2 * precision * recall) / (precision + recall)

        meteor_score = self.meteor.compute(
            predictions=[generated_answer], 
            references=[correct_answer]
        )
        # Return the metrics
        return {
            "is_correct": is_correct,
            "similarity_score": similarity_score,
            "meteor": float(meteor_score['meteor']),
            "f1_score": f1,
            "recall": recall,
            "precision": precision,
            "judgment": judgment
        }
    
    def normalize_answer(self, s):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))