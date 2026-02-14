from typing import List, Dict, Any
from datasets import load_dataset

class HotpotQADataLoader:
    """Load and process the HotpotQA dataset from Hugging Face."""
    
    def __init__(self, split: str = 'train', limit: int = None):
        self.split = split
        self.limit = limit
        self.dataset = None
        self.processed_samples = []
    
    def load(self) -> List[Dict[str, Any]]:
        self.dataset = load_dataset(
            "hotpot_qa", "distractor", 
            split=self.split
        )
        
        if self.limit:
            self.dataset = self.dataset.select(range(self.limit))
        
        self._process_samples()
        return self.processed_samples
    
    def _process_samples(self) -> None:
        self.processed_samples = []
        
        for item in self.dataset:
            # 1. Create Lookup Dictionary
            context_dict = dict(zip(item['context']['title'], item['context']['sentences']))
            gold_pointer_set = set(zip(item['supporting_facts']['title'], item['supporting_facts']['sent_id']))
            # 2. Prepare the data containers
            golden_sentences_text = [] # This will become your new 'context'
            supporting_facts_meta = [] # This keeps the [Title, ID] pairs
            distractor_text = []     # Will go into 'distractors'
            # 3. Build 'context' (Real Facts) 
            # We loop through supporting_facts to preserve the specific reasoning order
            # (Context Lookup Dictionary)
            context_dict = dict(zip(item['context']['title'], item['context']['sentences']))
            
            for title, sent_id in zip(item['supporting_facts']['title'], item['supporting_facts']['sent_id']):
                if title in context_dict and sent_id < len(context_dict[title]):
                    golden_sentences_text.append([title, context_dict[title][sent_id]])
                    supporting_facts_meta.append([title, sent_id])

            # 4. Build 'distractors' (The leftover sentences)
            # We iterate through ALL available documents and sentences
            for title, sentences in zip(item['context']['title'], item['context']['sentences']):
                for sent_idx, sentence in enumerate(sentences):
                    if (title, sent_idx) not in gold_pointer_set:
                        # If NOT gold, it's a distractor
                        distractor_text.append([title,sentence])
            
            # 4. Build the final dictionary
            sample = {
                "question": item['question'],
                "answer": item['answer'],
                "context": golden_sentences_text,     
                "distractors": distractor_text,    
                "supporting_facts": supporting_facts_meta, 
                "type": item['type'],
                "level": item['level']
            }
            self.processed_samples.append(sample)
