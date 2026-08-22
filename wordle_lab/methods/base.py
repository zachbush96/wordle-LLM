from __future__ import annotations

from abc import ABC, abstractmethod


class TrainerBackend(ABC):
    @abstractmethod
    def prepare(self, spec: dict): ...

    @abstractmethod
    def train(self): ...

    @abstractmethod
    def save(self): ...

    @abstractmethod
    def load_for_eval(self): ...
