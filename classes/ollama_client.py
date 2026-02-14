
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        
    def generate(self, model: str, prompt: str, temperature: float = 0.7, 
                 max_tokens: int = 1024) -> str:
        """Generate text using Ollama"""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 8192,
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=600)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama: {e}")
            return ""
    
    def check_connection(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.Timeout:
            logger.error("request timeout - Ollama may not be running", flush=True)
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"request exception {e}", flush=True)
            return False