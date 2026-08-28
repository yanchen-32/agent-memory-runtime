from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        """Generate one response from a fully constructed prompt."""


class Agent(ABC):
    @abstractmethod
    def answer(self, query: str) -> str:
        raise NotImplementedError
