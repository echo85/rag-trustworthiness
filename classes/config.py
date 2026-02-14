from dataclasses import dataclass
from typing import List
from typing import Literal

@dataclass
class Config:
    """Configuration for the evaluation pipeline"""
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    poison_model: str = "llama3.1:8b"  # Model for generating poisoned docs
    rag_model: List[str] = None     # Model for RAG generation
    judge_model: str = "ministral-3:14b"    # Model for evaluation
    judge_method: Literal["nli","llm"] = "llm"
    # Dataset settings
    num_samples: int = 30  # Number of HotpotQA samples to use
    
    strategies: List[str] = None
    poison_ratios: List[float] = None  # [0.0, 0.33, 0.67, 1.0]
    distractors: List[float] = None  # [0.0, 0.33, 0.67, 1.0]
    num_documents_per_sample: int = 3  # Original golden documents
    
    # Evaluation settings
    temperature: float = 0.7
    max_tokens: int = 1024
    
    def __post_init__(self):
        if self.poison_ratios is None:
            self.poison_ratios = [0.0,0.3]
        if self.distractors is None:
            self.distractors = [8,30]
        if self.strategies is None:
            self.strategies = ["baseline","critical"]