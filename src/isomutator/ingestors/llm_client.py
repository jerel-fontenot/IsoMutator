"""
ALGORITHM SUMMARY:
The AttackerLLMClient module utilizes the Strategy and Factory design patterns 
to polymorphically handle different LLM API schemas (Ollama vs. OpenAI-compatible).
1. The `LLMClientFactory` instantiates the correct adapter based on the `api_type` configuration.
2. Both clients implement `AttackerClientInterface` to ensure a standard `generate_json` contract.
3. Both clients perform regex-based markdown stripping to recover poorly formatted JSON.
4. Both clients implement an autonomous feedback loop, catching `JSONDecodeError` 
    exceptions and passing them back to the LLM for self-correction.
"""

import abc
import asyncio
import aiohttp
import json
import re

from isomutator.core.config import settings
from isomutator.core.log_manager import LogManager


class AttackerClientInterface(abc.ABC):
    """Abstract base class for all LLM API adapters."""
    
    def __init__(self, url: str, model: str):
        self.url = url
        self.model = model
        self.logger = LogManager.get_logger("isomutator.llm_client")
        
    @abc.abstractmethod
    async def generate_json(self, session: aiohttp.ClientSession, messages: list, max_retries: int = 3) -> dict:
        """Executes the API call and returns a validated JSON dictionary."""
        pass
        
    def _clean_json_response(self, response_text: str) -> str:
        """Extracts JSON from markdown code blocks using regex."""
        clean_text = response_text
        md_ticks = chr(96) * 3
        pattern = rf'{md_ticks}(?:json)?\s*(.*?)\s*{md_ticks}'
        
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            clean_text = match.group(1)
            self.logger.trace("Stripped markdown formatting from LLM response.")
        return clean_text


class OllamaClient(AttackerClientInterface):
    """Adapter for the Ollama /api/chat schema."""
    
    def __init__(self, url: str, model: str):
        # Append specific Ollama endpoint. Factory passes base URL.
        super().__init__(f"{url.rstrip('/')}/api/chat", model)
        self.logger.trace(f"OllamaClient initialized. Targeting: {self.url} (Model: {self.model})")

    async def generate_json(self, session: aiohttp.ClientSession, messages: list, max_retries: int = 3) -> dict:
        current_messages = messages.copy()
        
        # Small models like Llama 3.2 often ignore the format flag without explicit instructions.
        if not any(m.get("role") == "system" for m in current_messages):
            current_messages.insert(0, {
                "role": "system",
                "content": "You are a strict data-generation API. You must output ONLY valid, raw JSON. Do not include markdown formatting, conversational filler, or trailing commas."
            })
        
        for attempt in range(max_retries):
            # Strict Ollama Schema with strict token formatting
            payload = {
                "model": self.model,
                "format": "json",
                "messages": current_messages,
                "stream": False,
                "options": {
                    "temperature": 0.1  # Reduce hallucination variations
                }
            }

            try:
                self.logger.trace(f"Dispatching Ollama JSON request (Attempt {attempt + 1}/{max_retries})...")
                # --- THE FIX: Dynamic configuration timeout ---
                async with session.post(self.url, json=payload, timeout=aiohttp.ClientTimeout(total=settings.network_timeout)) as response:
                    if response.status == 422:
                        self.logger.error("HTTP 422 Unprocessable Entity. Schema mismatch from Ollama Target.")
                        return {}
                    if response.status != 200:
                        self.logger.warning(f"HTTP {response.status} from Ollama. Retrying in 2s...")
                        await asyncio.sleep(2)
                        continue

                    result_json = await response.json()
                    response_text = result_json.get("message", {}).get("content", "{}")
                    clean_text = self._clean_json_response(response_text)
                    
                    try:
                        parsed_data = json.loads(clean_text)
                        if attempt > 0:
                            self.logger.info(f"Successfully recovered JSON syntax on attempt {attempt + 1}.")
                        return parsed_data
                        
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"JSON Parse Error: {e}. Routing back to LLM...")
                        current_messages.append({"role": "assistant", "content": response_text})
                        current_messages.append({
                            "role": "user", 
                            "content": f"Your previous response failed JSON parsing with error: {e}. "
                                        f"Please output ONLY valid JSON. Remove trailing commas and ensure quotes are escaped."
                        })
                        
            except asyncio.TimeoutError:
                self.logger.warning(f"Ollama API request timed out after {settings.network_timeout}s.")
                return {}
            except Exception as e:
                self.logger.error(f"Network error during Ollama generation: {e}")
                
        self.logger.error("Exhausted all JSON correction retries. Generation failed.")
        return {}


class OpenAIClient(AttackerClientInterface):
    """Adapter for OpenAI-compatible /v1/chat/completions schemas (vLLM, TGI, etc)."""
    
    def __init__(self, url: str, model: str):
        super().__init__(f"{url.rstrip('/')}/v1/chat/completions", model)
        self.logger.trace(f"OpenAIClient initialized. Targeting: {self.url} (Model: {self.model})")

    async def generate_json(self, session: aiohttp.ClientSession, messages: list, max_retries: int = 3) -> dict:
        current_messages = messages.copy()
        
        if not any(m.get("role") == "system" for m in current_messages):
            current_messages.insert(0, {
                "role": "system",
                "content": "You are a strict data-generation API. You must output ONLY valid, raw JSON."
            })
        
        for attempt in range(max_retries):
            # Strict OpenAI Schema utilizing native json_object enforcement
            payload = {
                "model": self.model,
                "messages": current_messages,
                "temperature": 0.1,
                "stream": False,
                "response_format": {"type": "json_object"}
            }

            try:
                self.logger.trace(f"Dispatching OpenAI-compatible request (Attempt {attempt + 1}/{max_retries})...")
                async with session.post(self.url, json=payload, timeout=aiohttp.ClientTimeout(total=settings.network_timeout)) as response:
                    if response.status == 422:
                        self.logger.error("HTTP 422 Unprocessable Entity. Schema mismatch from OpenAI-Compatible Target.")
                        return {}
                    if response.status != 200:
                        self.logger.warning(f"HTTP {response.status} from target. Retrying in 2s...")
                        await asyncio.sleep(2)
                        continue

                    result_json = await response.json()
                    choices = result_json.get("choices", [])
                    response_text = "{}"
                    if choices and isinstance(choices, list):
                        response_text = choices[0].get("message", {}).get("content", "{}")
                        
                    clean_text = self._clean_json_response(response_text)
                    
                    try:
                        parsed_data = json.loads(clean_text)
                        if attempt > 0:
                            self.logger.info(f"Successfully recovered JSON syntax on attempt {attempt + 1}.")
                        return parsed_data
                        
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"JSON Parse Error: {e}. Routing back to LLM...")
                        current_messages.append({"role": "assistant", "content": response_text})
                        current_messages.append({
                            "role": "user", 
                            "content": f"Your previous response failed JSON parsing with error: {e}. "
                                        f"Please output ONLY valid JSON. Remove trailing commas and ensure quotes are escaped."
                        })
            
            except asyncio.TimeoutError:
                self.logger.warning(f"OpenAI API request timed out after {settings.network_timeout}s.")
                return {}            
            except Exception as e:
                self.logger.error(f"Network error during OpenAI generation: {e}")
                
        self.logger.error("Exhausted all JSON correction retries. Generation failed.")
        return {}


class LLMClientFactory:
    """Factory to instantiate the correct LLM adapter based on configuration."""
    
    @staticmethod
    def create(api_type: str, url: str, model: str) -> AttackerClientInterface:
        api_type = api_type.lower()
        if api_type == "ollama":
            return OllamaClient(url=url, model=model)
        elif api_type == "openai":
            return OpenAIClient(url=url, model=model)
        else:
            raise ValueError(f"Unsupported API type: {api_type}. Choose 'ollama' or 'openai'.")