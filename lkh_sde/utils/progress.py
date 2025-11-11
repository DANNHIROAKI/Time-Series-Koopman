"""Simple progress iterator used as a fallback when tqdm is unavailable."""
from __future__ import annotations

from typing import Iterable, Iterator, TypeVar


T = TypeVar("T")


def progress(iterable: Iterable[T], description: str = "") -> Iterator[T]:
    for index, item in enumerate(iterable, start=1):
        if description and index == 1:
            print(description)
        yield item
