"""Minimal tqdm shim to satisfy imports when the real package is not installed.
This provides a `tqdm` function that yields its input iterator unchanged and
accepts common kwargs used in the repository (desc, total, leave).
"""
from typing import Iterable, Iterator, Any


def tqdm(iterable: Iterable[Any], **kwargs) -> Iterator[Any]:
    # Simple pass-through iterator; ignores tqdm-specific kwargs.
    for item in iterable:
        yield item


def trange(*args, **kwargs):
    # Returns an iterator over range(...) with tqdm semantics (pass-through here)
    return tqdm(range(*args), **kwargs)
