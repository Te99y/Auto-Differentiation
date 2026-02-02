from __future__ import annotations

from typing import Any

from .array import array
from .tensor import tensor


def as_tensor(x: Any) -> tensor:
    return x if isinstance(x, tensor) else tensor.const(x)


def as_array(x: Any) -> array:
    return x if isinstance(x, array) else array(x)
