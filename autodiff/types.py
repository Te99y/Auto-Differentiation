# autodiff/types.py
from __future__ import annotations
from typing import TYPE_CHECKING, TypeAlias

Number: TypeAlias = int | float

if TYPE_CHECKING:
    from .array import array
    from .tensor import tensor

    # Inputs users might pass into ops/linalg
    ArrayLike: TypeAlias = Number | list | array
    TensorLike: TypeAlias = Number | list | array | tensor
else:
    # runtime-safe fallbacks to avoid import cycles
    ArrayLike: TypeAlias = object
    TensorLike: TypeAlias = object
