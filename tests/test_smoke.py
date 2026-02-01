import autodiff


def test_public_api_surface() -> None:
    # Everything in __all__ should be present on the package
    expected_names = [
        "array", "tensor",
        "add", "sub", "mul", "div", "neg", "abs_", "exp", "log", "sin", "cos",
        "flatten", "reshape", "transpose", "swapaxes", "matmul",
        "jvp", "vjp",
    ]
    for name in expected_names:
        assert hasattr(autodiff, name), f"autodiff is missing {name}"


def test_direct_imports() -> None:
    # Verify direct imports work and are callable/constructible
    from autodiff import (
        array, tensor,
        add, sub, mul, div, neg, abs_, exp, log, sin, cos,
        flatten, reshape, transpose, swapaxes, matmul,
        jvp, vjp,
    )

    # Classes should be callable (constructors)
    assert callable(array)
    assert callable(tensor)

    # Functions should be callable
    for fn in [add, sub, mul, div, neg, abs_, exp, log, sin, cos,
               flatten, reshape, transpose, swapaxes, matmul,
               jvp, vjp]:
        assert callable(fn)
