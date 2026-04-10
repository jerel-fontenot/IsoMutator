"""
ALGORITHM SUMMARY:
The AttackerLLMClient encapsulates all network communications, retry logic, 
and JSON hardening for the remote Attacker LLM. 
1. It reads the target `attacker_url` dynamically from the IsoConfig singleton.
2. It executes asynchronous HTTP requests against the Ollama /api/chat endpoint.
3. It performs regex-based markdown stripping to recover poorly formatted JSON.
4. It implements an autonomous feedback loop, catching `JSONDecodeError` exceptions 
   and passing them back to the LLM for self-correction.

TECHNOLOGY QUIRKS:
- Composition: This class is designed to be injected into the various Mutators, 
  keeping the inheritance tree clean and allowing future Mutators to easily 
  swap out their "brain" without overriding complex networking logic.
"""

import asyncio
import aiohttp
import json
import logging
import re

from isomutator.core.config import settings
from isomutator.core.log_manager import LogManager

# Establish TRACE level logging if it does not exist
TRACE_LEVEL_NUM = 5
if not hasattr(logging, "TRACE"):
    logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
    logging.TRACE = TRACE_LEVEL_NUM

def trace(self, message, *args, **kws):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kws)

logging.Logger.trace = trace


class AttackerLLMClient:
    """Handles hardened JSON generation and error recovery for the Attacker LLM."""
    
    def __init__(self, model_name: str = "llama3.2"):
        self.logger = LogManager.get_logger("isomutator.llm_client")
        self.url = f"{settings.attacker_url}/api/chat"
        self.model = model_name
        self.logger.trace(f"AttackerLLMClient initialized. Targeting: {self.url} (Model: {self.model})")

    async def generate_json(self, session: aiohttp.ClientSession, messages: list, max_retries: int = 3) -> dict:
        """Executes the LLM call with built-in Markdown stripping and a feedback-driven retry loop."""
        current_messages = messages.copy()
        
        for attempt in range(max_retries):
            payload = {
                "model": self.model,
                "format": "json",
                "messages": current_messages,
                "stream": False
            }

            try:
                self.logger.trace(f"Dispatching JSON generation request (Attempt {attempt + 1}/{max_retries})...")
                async with session.post(self.url, json=payload, timeout=300.0) as response:
                    if response.status != 200:
                        self.logger.warning(f"HTTP {response.status} from Attacker LLM. Retrying in 2s...")
                        await asyncio.sleep(2)
                        continue

                    result_json = await response.json()
                    response_text = result_json.get("message", {}).get("content", "{}")
                    
                    # 1. Regex Markdown Stripping
                    clean_text = response_text
                    md_ticks = chr(96) * 3
                    pattern = rf'{md_ticks}(?:json)?\s*(.*?)\s*{md_ticks}'
                    
                    match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
                    if match:
                        clean_text = match.group(1)
                        self.logger.trace("Stripped markdown formatting from LLM response.")
                        
                    # 2. Strict JSON Parse
                    try:
                        parsed_data = json.loads(clean_text)
                        if attempt > 0:
                            self.logger.info(f"Successfully recovered JSON syntax on attempt {attempt + 1}.")
                        return parsed_data
                        
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"JSON Parse Error on attempt {attempt + 1}: {e}. Routing error back to LLM...")
                        current_messages.append({"role": "assistant", "content": response_text})
                        current_messages.append({
                            "role": "user", 
                            "content": f"Your previous response failed JSON parsing with error: {e}. "
                                       f"Please output ONLY valid JSON. Remove trailing commas and ensure quotes are escaped."
                        })
                        
            except Exception as e:
                self.logger.error(f"Network error during LLM generation: {e}")
                
        self.logger.error("Exhausted all JSON correction retries. Generation failed.")
        return {}