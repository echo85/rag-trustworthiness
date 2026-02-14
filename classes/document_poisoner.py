from classes.ollama_client import OllamaClient
import random
from typing import List, Tuple, Dict

class DocumentPoisoner:
    """Generates poisoned/contradictory documents using LLM"""
    
    def __init__(self, ollama_client: OllamaClient, model: str):
        self.ollama = ollama_client
        self.model = model
    
    def generate_noisy_document(self, question: str) -> str:
        """Generate an irrelevant but topically related document"""
        prompt = f"""Given this question, create a SHORT document that is topically related but does NOT answer the question.

Question: {question}

Generate a 2-3 sentence document about a related topic that:
1. Mentions similar concepts or entities
2. Does NOT contain the answer
3. Could distract from finding the correct answer
4. Seems authoritative

Generate ONLY the document text, nothing else:"""
        
    def generate_fake_answer(self, question: str, correct_answer: str, documents: str) -> str:
        """Generate a plausible but incorrect answer"""
        
        filtered_documents = [doc for doc in documents if doc["is_poisoned"] == True]
        if len(filtered_documents) == 0:
            filtered_documents = documents

        context = "\n\n".join([
                f"[{i+1}] {doc['text']}" 
                for i, doc in enumerate(filtered_documents)
            ])

        prompt = f"""Given this question and correct answer, create a SHORT, PLAUSIBLE but INCORRECT answer that would mislead someone.
Question: {question}
Correct Answer: {correct_answer}    
Only use information from these poisoned documents:
Documents: {context}
Generate ONLY the misleading answer text, nothing else. Be concise like the correct answer."""
        
        return self.ollama.generate(self.model, prompt, temperature=0.8, max_tokens=200)
    
    def generate_counterfactual(self, question: str, correct_answer: str, 
                               context: str) -> str:
        """Generate a document with false information"""
        prompt = f"""Given this question and correct answer, create a SHORT, PLAUSIBLE but INCORRECT document that would mislead someone.
Question: {question}
Correct Answer: {correct_answer}
Original Context: {context}

Generate a 2-3 sentence document that:
1. Seems authoritative and factual
2. Contains WRONG information that contradicts the correct answer
3. Mentions specific but incorrect dates, names, or facts
4. Looks like it could come from Wikipedia or an encyclopedia

Generate ONLY the misleading document text, nothing else:"""
        return self.ollama.generate(self.model, prompt, temperature=0.8, max_tokens=200)
    
    def irrelevants_documents(self, 
                        question: str,
                        original_docs: List[Tuple[str, List[str]]], 
                        distractor_docs: List[Tuple[str, List[str]]],
                        num_distractors: int=8) -> List[Dict]:
        """
        Create a mixed set of original and poisoned documents
        
        Returns:
            List of documents with metadata indicating if poisoned
        """
        documents = []
        distractor_documents = min(num_distractors, len(distractor_docs))
        
        # Add original documents
        for doc_title, doc_text in original_docs:
            documents.append({
            "text": doc_text,
            "title": doc_title,
            "is_poisoned": False,
            "poison_type": None
            })
        
        # Add syntetic distractor documents
        if distractor_documents < num_distractors:
            for i in range(num_distractors-distractor_documents):
                documents.append({
                    "text":  self.generate_noisy_document(question),
                    "title": f"Generated_Document_{i+1}",
                    "is_poisoned": True,
                    "poison_type": "noisy"
                })
               
        # Add distractor documents from dataset
        for doc_title, doc_text in distractor_docs[:distractor_documents]:
            documents.append({
            "text": doc_text,
            "title": doc_title,
            "is_poisoned": True,
            "poison_type": "noisy"
            })
        
        # Shuffle to mix poisoned and real documents
        random.shuffle(documents)
        
        return documents
    
    def poison_documents(self, question: str, answer: str, 
                        original_docs: List[Tuple[str, List[str]]], 
                        poison_ratio: float) -> List[Dict]:
        """
        Create a mixed set of original and poisoned documents
        
        Returns:
            List of documents with metadata indicating if poisoned
        """
        documents = []
        # Add original documents
        for doc_title, doc_text in original_docs:
           
            documents.append({
                "text": doc_text,
                "title": doc_title,
                "is_poisoned": False,
                "poison_type": None
            })
        
        # Calculate number of poisoned docs to add
        if poison_ratio > 0:
            num_poison = max(1,int(len(documents) * poison_ratio / (1 - poison_ratio))) if poison_ratio < 1.0 else len(documents)
        else:
            num_poison = 0

        # Generate poisoned documents
        for i in range(num_poison):
            context = " ".join([d["text"] for d in documents[:2]])
            poisoned_text = self.generate_counterfactual(question, answer, context)
            documents.append({
                "text": poisoned_text.strip(),
                "title": f"Generated_Document_{i+1}",
                "is_poisoned": True,
                "poison_type": "counterfactual"
            })
        
        random.shuffle(documents)
        
        return documents