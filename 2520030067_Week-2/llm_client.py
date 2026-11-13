from abc import ABC, abstractmethod
from google import genai
import requests
# Common interface
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt):
        pass
# Gemini cloud provider
class GeminiProvider(LLMProvider):
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
    def generate(self, prompt):
        response = self.client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )
        return response.text
# Ollama local provider
class OllamaProvider(LLMProvider):
    def __init__(self, model="llama3.2:1b"):
        self.model = model
        self.url = "http://127.0.0.1:11434/api/generate"
    def generate(self, prompt):
        try:
            response = requests.post(
                self.url,
                json={
                    "model": "llama3.2:1b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )
            print("API Status:", response.status_code)
            if response.status_code != 200:
                print("API Error:", response.text)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            print("Ollama connection error:", e)
            raise
# Common client
class LLMClient:
    def __init__(self, provider):
        self.provider = provider
    def generate(self, prompt):
        return self.provider.generate(prompt)
