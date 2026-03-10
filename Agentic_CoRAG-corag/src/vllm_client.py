from typing import List, Dict, Union
from concurrent.futures import ThreadPoolExecutor
import time
import requests

from utils import AtomicCounter
from logger_config import logger


DEFAULT_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 3


def _normalize_base_url(api_base: str) -> str:
    return api_base.rstrip('/')


def get_vllm_model_id(host: str = "localhost", port: int = 8000, api_key: str = "token-123", api_base: str = None) -> str:
    if api_base:
        base_url = _normalize_base_url(api_base)
    else:
        base_url = f"http://{host}:{port}/v1"

    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    try:
        response = requests.get(f"{base_url}/models", headers=headers, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        models = response.json()
        return models['data'][0]['id']
    except Exception as e:
        # Fallback or re-raise with clear message
        raise RuntimeError(f"Failed to fetch models from vLLM: {e}")


class VllmClient:

    def __init__(self, model: str, host: str = 'localhost', port: int = 8000, api_key: str = 'token-123', api_base: str = None):
        super().__init__()
        self.model = model
        if api_base:
            self.base_url = _normalize_base_url(api_base)
        else:
            self.base_url = f"http://{host}:{port}/v1"
            
        self.api_key = api_key
        self.token_consumed: AtomicCounter = AtomicCounter()
        self.session = requests.Session()

    def _post_with_retries(self, endpoint: str, headers: Dict[str, str], data: Dict, timeout: int = DEFAULT_TIMEOUT) -> requests.Response:
        last_error = None
        url = f"{self.base_url}{endpoint}"
        for attempt in range(1, DEFAULT_MAX_RETRIES + 1):
            try:
                response = self.session.post(url, headers=headers, json=data, timeout=timeout)
                response.raise_for_status()
                return response
            except requests.RequestException as error:
                last_error = error
                status_code = getattr(error.response, 'status_code', 'N/A') if hasattr(error, 'response') else 'N/A'
                response_text = ''
                if getattr(error, 'response', None) is not None:
                    response_text = error.response.text[:500]
                if attempt == DEFAULT_MAX_RETRIES:
                    raise RuntimeError(
                        f"vLLM request failed after {DEFAULT_MAX_RETRIES} attempts. "
                        f"url={url}, status={status_code}, response={response_text or 'empty'}"
                    ) from error
                logger.warning(
                    f"vLLM request failed on attempt {attempt}/{DEFAULT_MAX_RETRIES}. "
                    f"url={url}, status={status_code}. Retrying..."
                )
                time.sleep(min(2 ** (attempt - 1), 4))
        raise RuntimeError(f"vLLM request failed unexpectedly: {last_error}")

    @staticmethod
    def _extract_chat_content(completion: Dict) -> str:
        choices = completion.get('choices') or []
        if not choices or 'message' not in choices[0] or 'content' not in choices[0]['message']:
            raise RuntimeError(f"Invalid chat completion response: {str(completion)[:500]}")
        return choices[0]['message']['content']

    @staticmethod
    def _extract_completion_text(completion: Dict) -> str:
        choices = completion.get('choices') or []
        if not choices or 'text' not in choices[0]:
            raise RuntimeError(f"Invalid completion response: {str(completion)[:500]}")
        return choices[0]['text']

    def call_chat(self, messages: List[Dict], return_str: bool = True, **kwargs) -> Union[str, Dict]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": messages,
            **kwargs
        }

        response = self._post_with_retries("/chat/completions", headers=headers, data=data)
        completion = response.json()
        
        if 'usage' in completion:
            self.token_consumed.increment(num=completion['usage']['total_tokens'])

        return self._extract_chat_content(completion) if return_str else completion

    def call_completion(self, prompt: str, return_str: bool = True, **kwargs) -> Union[str, Dict]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "prompt": prompt,
            **kwargs
        }

        response = self._post_with_retries("/completions", headers=headers, data=data)
        completion = response.json()
        
        if 'usage' in completion:
            self.token_consumed.increment(num=completion['usage']['total_tokens'])

        return self._extract_completion_text(completion) if return_str else completion

    def batch_call_chat(self, messages: List[List[Dict]], return_str: bool = True, num_workers: int = 4, **kwargs) -> List[Union[str, Dict]]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            return list(executor.map(
                lambda m: self.call_chat(m, return_str=return_str, **kwargs),
                messages
            ))
