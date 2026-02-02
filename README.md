# Auto Differentiation
[![CI](https://github.com/Te99y/Auto-Differentiation/actions/workflows/ci.yml/badge.svg)](https://github.com/Te99y/Auto-Differentiation/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A zero runtime dependency automatic differentiation engine supporting forward‑mode (JVP) and reverse‑mode (VJP) with a pure‑Python, list‑backed N‑dimensional array.

---

##  Features

- Dynamic computation graph built from tensor operations  
- Forward‑mode automatic differentiation via Jacobian–Vector Product (JVP)  
- Reverse‑mode automatic differentiation via Vector–Jacobian Product (VJP)  
- Broadcasting and elementwise operations  
- Linear algebra support: `@` (matrix multiplication), `transpose`, `swapaxes`, `flatten`, `reshape`  
- Does not rely on NumPy/JAX/PyTorch in runtime
- Tested against:
  - finite‑difference gradients  
  - NumPy reference implementations  
  - optional JAX comparisons  
- Continuous Integration with `pytest`

---

##  Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/your-username/autodiff.git
cd autodiff
pip install -e ".[dev]"
```

Optional: to enable JAX comparison tests

```bash
pip install -e ".[dev,jax]"
```

---

##  Quickstart

### Reverse‑mode autodiff (VJP / gradients)

```python
from autodiff import tensor, vjp

x = tensor(2.0)
y = (x * x).sin()   # y = sin(x^2)

grads = vjp(y, inputs={x: x.arr}, cotangent=None, pull_back=False)

print("y =", y.arr.value)
print("dy/dx =", grads[x].value)
```

**Expected result:**

```
y = [sin(4.0)]
dy/dx = [2*x*cos(x^2)] ≈ -2.614
```

---

### Forward‑mode autodiff (JVP)

```python
from autodiff import array, tensor, jvp

x = tensor(3.0)
y = x * x

tangent = jvp(
    y,
    inputs={x: x.arr},
    directions={x: array(1.0)},
    push_forward=False,
)

print("JVP =", tangent.value)  # should be 2*x = 6
```

---

##  API Overview

### Core objects
- **`array`** — list‑backed N‑dimensional numeric container  
- **`tensor`** — wraps array and builds a computation graph  

### Differentiation
- `jvp(f, inputs, directions, push_forward=False)`  
- `vjp(f, inputs, cotangent, pull_back=False)`  

### Operations

**Elementwise:**
- `+`, `-`, `*`, `/`, `abs`, `neg`  
- `sin`, `cos`, `exp`, `log`  

**Linear algebra:**
- `@` (matrix multiplication)  
- `transpose()`, `swapaxes()`  
- `flatten()`, `reshape()`  

---

## Testing

Run the test suite:

```bash
pytest
```

Tests include:
- Unit tests for array and utility functions  
- Gradient checks using finite differences (central difference)  
- Comparisons against NumPy  
- Optional comparisons against JAX (skipped if not installed)  

---

##  Project Structure

```
autodiff/
    __init__.py     # export to public
    ad.py           # jvp / vjp
    api.py          # convenient functions
    array.py        # numeric container
    linalg.py       # matrix operations
    ops.py          # elementary operations
    tensor.py       # computation graph
    types.py        # type aliasing
    utils.py        # list-based helpers
legacy/
    AutoDiff.py     # prototype
    test.py         # old tests
tests/
    test_ad.py
    test_array.py
    test_compare_jax.py
    test_linalg.py
    test_ops.py
    test_smoke.py
    test_utils.py
    utils.py
examples/
    scalar_grad.py
    jvp_vs_vjp.py
pyproject.toml
README.md
```

---

##  Design Notes

- The computation graph is dynamic, similar to PyTorch.  
- Gradients are propagated using:
  - topological ordering of the graph  
  - local derivative rules for each operation  
- Broadcasting is handled explicitly when accumulating gradients.  
- This project prioritizes clarity and correctness over performance.  

---

## ️ Limitations

- Pure Python (slow for large tensors)  
- No GPU support  
- No slicing or advanced indexing  
- Intended for learning and experimentation, not production ML workloads  

---

##  Roadmap

- Written autodiff tutorials
- Numerical stability improvements  
- More linear algebra ops (`sum`, `mean`)  
- Visualization of computation graph  
- Vectorized gradient checking  

---

## ️ Acknowledgements

Inspired by:
- Fang, Yu‑Hsueh; Lin, He‑Zhe; Liu, Jie‑Jyun; and Lin, Chih‑Jen.  
  *[A Step‑by‑step Introduction to the Implementation of Automatic Differentiation.](https://www.csie.ntu.edu.tw/~cjlin/papers/autodiff/autodiff.pdf)*  
  National Taiwan University; Mohamed bin Zayed University of Artificial Intelligence.  
 
- Hare, Jonathon.  
  *[An Introduction to Automatic Differentiation.](https://comp6248.ecs.soton.ac.uk/handouts/autograd-handouts.pdf)*  
  Vision, Learning and Control, University of Southampton.
