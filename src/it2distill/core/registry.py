"""Registry pattern: components self-register by name, config selects them (Open/Closed)."""
from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def deco(cls: type[T]) -> type[T]:
            if name in self._items:
                raise KeyError(f"{self._kind} '{name}' already registered")
            self._items[name] = cls
            return cls
        return deco

    def create(self, name: str, **kwargs) -> T:
        if name not in self._items:
            raise KeyError(f"Unknown {self._kind} '{name}'. Available: {sorted(self._items)}")
        return self._items[name](**kwargs)

    def names(self) -> list[str]:
        return sorted(self._items)
