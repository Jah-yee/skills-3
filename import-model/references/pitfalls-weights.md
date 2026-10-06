# Weight adapter pitfalls

This page covers Phase 2 traps that surface while you write
`weight_adapters.py`, map HF safetensor keys to the root module's parameter
names, and handle BF16 tensors. It describes these pitfalls:

- Weight names follow the root module's attribute paths
- Tied embeddings keep one copy of the shared weight
- Unconsumed checkpoint tensors load without an error
- Dtype mismatches raise unless `auto_cast` permits them
- `numpy.from_dlpack` does not support bfloat16
- Embedding row-count may exceed `vocab_size`

## Weight names follow the root module's attribute paths

`compile(weights=...)` looks up each parameter by its attribute path from the
root module that `_instantiate_module()` returns. When the root stores its
text model as `self.language_model`, every text-model weight is named
`language_model.<...>`, and the adapter maps HF's `model.layers.` to
`language_model.layers.` (`olmo3/weight_adapters.py`). Print the expected
names from the lazily built module:

```python
from max.experimental import functional as F

with F.lazy():
    model = MyModel(config, kv_params)
expected = {name for name, _ in model.parameters}
```

HF ships `lm_head.weight` at the top level. Map it to wherever your root
module holds the head (`language_model.lm_head.weight` in `olmo3`).

## Tied embeddings keep one copy of the shared weight

When `config.tie_word_embeddings=True`, create no `lm_head` parameter and
compute the logits from the embedding weight in `forward()`:

```python
if self.tie_word_embeddings:
    logits = h @ self.embed_tokens.weight.T
else:
    logits = self.lm_head(h)
```

The state dict then holds one tensor, `embed_tokens.weight`. If the
checkpoint also ships `lm_head.weight` for a tied model, drop it in the
adapter.

## Unconsumed checkpoint tensors load without an error

`compile(weights=...)` raises `KeyError` for a parameter with no tensor and
`ValueError` for a shape or dtype mismatch. A checkpoint tensor whose name
matches no parameter is left out without a warning. That hides these bugs:

- A rename typo sends a real tensor to a name nothing reads, while another
  rename fills the intended parameter with a tensor of the same shape.
- The checkpoint carries a component the module never built (QK-norm
  weights, a shared expert, attention biases), so the port computes without
  it.

Run the unconsumed-key audit in [state-dict-audit.md](state-dict-audit.md)
before serving.

## Dtype mismatches raise unless `auto_cast` permits them

A parameter's dtype comes from the module's default dtype
(`_module_default_dtype()`, the encoding's dtype unless overridden). A
checkpoint tensor in another dtype fails `compile()` with
`Loaded tensor (shape=..., dtype=...) not assignable to parameter`. Cast in
the adapter, or override `_prepare_state_dict()` in `model.py`.
`ModuleV3PipelineModelWithKVCache.load_model()` calls `compile()` without
`auto_cast`, so a pipeline port needs one of these casts. When you call
`compile()` yourself, `compile(auto_cast=True)` casts between `float32` and
`bfloat16`.

## `numpy.from_dlpack` does not support bfloat16

`np.from_dlpack(weight_data.data)` crashes with
`RuntimeError: Unsupported dtype in DLTensor` when the underlying
buffer is bfloat16 (or fp8, fp4, etc.). NumPy has no native bfloat16
dtype. For BF16 weight manipulation in a weight adapter (slicing,
reshaping, repacking), use torch as the DLPack bridge:

```python
import torch
from max.graph import Shape
from max.graph.weights import WeightData

t = torch.from_dlpack(weight_data.data)
t_sliced = t[:vocab_size].contiguous()
new_wd = WeightData(
    data=t_sliced,
    name=name,
    dtype=weight_data.dtype,
    shape=Shape((vocab_size, hidden_size)),
    quantization_encoding=weight_data.quantization_encoding,
)
```

`WeightData.shape` is a `max.graph.Shape`. Wrap the dimensions in
`Shape((...))`.

## Embedding row-count may exceed `vocab_size`

Some HF models size their `nn.Embedding` larger than
`config.vocab_size` to make room for special image/tile/audio tokens
that the LM head doesn't predict over. Mllama ships
`nn.Embedding(vocab_size + 8, hidden_size)` with
`lm_head: Linear(hidden_size, vocab_size)`. The embedding has 128264 rows,
and `lm_head` has 128256. A MAX `Embedding` sized at `vocab_size` fails the
shape check on 128264-row weights. The weight adapter must slice the first
`vocab_size` rows of `embed_tokens.weight`. The extra rows are reserved
special tokens that never appear in text-only generation, so truncation is
safe for that scope. Use the torch-DLPack pattern above, because NumPy
crashes on BF16.
