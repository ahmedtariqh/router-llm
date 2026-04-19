from crewai import LLM
from pydantic import PrivateAttr

class SmartRouterLLM(LLM):
    _llms: list[LLM] = PrivateAttr(default=[])
    _current_index: int = PrivateAttr(default=0)

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, models: list[str], **kwargs):
        if not models:
            raise ValueError("You must provide a list of model strings.")
            
        super().__init__(model=models[0], **kwargs)
        self._llms = [LLM(model=m, **kwargs) for m in models]
        self._current_index = 0

    def supports_function_calling(self):
        return getattr(self._llms[self._current_index], "supports_function_calling", lambda: False)()

    def get_context_window_size(self):
        return getattr(self._llms[self._current_index], "get_context_window_size", lambda: 8192)()

    def call(self, messages: list[dict], **kwargs):
        max_attempts = len(self._llms)
        last_exception = None
        
        for _ in range(max_attempts):
            current_llm = self._llms[self._current_index]
            self._current_index = (self._current_index + 1) % len(self._llms)
            
            try:
                return current_llm.call(messages, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                is_rate_limit = "429" in error_msg or "rate limit" in error_msg
                is_quota = "insufficient_quota" in error_msg or "quota" in error_msg
                is_context = "context length" in error_msg or "contextwindow" in error_msg
                
                if is_rate_limit or is_quota or is_context:
                    last_exception = e
                    continue
                raise e
                
        raise RuntimeError(f"Router completely exhausted all {max_attempts} models. Last Error: {str(last_exception)}")