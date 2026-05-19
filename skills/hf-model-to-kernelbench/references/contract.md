# KernelBench task contract

A task is one self-contained Python module. When exec'd it must define exactly these three names. The harness exec's it, seeds RNG, then constructs and calls the model.

## Required symbols

| Symbol | Contract |
|---|---|
| `Model(nn.Module)` | Constructed as `Model(*get_init_inputs())`, then `.to(device).eval()`. Build the wrapped HF model from a **config**, never from pretrained weights. |
| `get_inputs()` | Returns a `list` of `forward` args. Tensors are moved to device by the harness; non-tensors pass through unchanged. |
| `get_init_inputs()` | Returns a `list` of constructor args for `Model` (often `[]`). |

## Invariants the bench enforces (the non-obvious ones)

| Invariant | Why it exists | How to satisfy |
|---|---|---|
| `forward` returns a **single tensor** | bench does `.float()`, `.abs()`, `.max()` on the output | If the HF model returns a dataclass/tuple, return one representative tensor (e.g. `out.pred_masks`, `out.last_hidden_state`). |
| **Deterministic** in `eval()` + `no_grad` | bench runs the same input twice and requires identical output | No dropout/sampling active in eval; no `torch.rand` inside `forward`. |
| **Output depends on inputs** | bench compares a second input set (different seed, same shapes) — a constant return is rejected | The representative tensor must be a real function of the inputs. |
| Positional `forward` order = `get_inputs()` order | harness calls `model(*get_inputs())` | Define `forward(self, a, b, ...)` in the same order the list is built. |
| Offline + seeded | harness seeds RNG *before* constructing model and inputs; no network allowed | Config-only init; build inputs with plain `torch` ops (the harness's seed makes them deterministic and varies them per seed). |
| Fixed shapes | bench reuses shapes across input seeds | `get_inputs()` returns fixed-shape tensors; only values vary by seed. |

## Skeleton

```python
import torch
import torch.nn as nn
from transformers import SomeConfig, SomeModel


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = SomeModel(SomeConfig())

    def forward(self, x, aux):
        out = self.net(x, aux)
        return out.last_hidden_state  # ONE tensor


def get_inputs():
    return [torch.randn(1, 3, 224, 224), torch.tensor([[1]], dtype=torch.int64)]


def get_init_inputs():
    return []
```

Constants in `get_inputs()` (built from literals, not RNG) stay identical across input seeds — that is fine and still defeats memoization as long as at least one returned tensor is seed-varying (e.g. a `torch.randn` input) and the output depends on it.
