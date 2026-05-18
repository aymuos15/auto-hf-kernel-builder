# Integrating the kernel back into the model (#7, transformers-only)

First pass: transformers kernels hook only (no diffusers, no manual surgery). Model = `config.yaml: model.id`; nothing model-specific here.

- Register the built kernel as a transformers *kernel layer* (replaces the target block class's forward); load via `KernelConfig` passed to `<Model>.from_pretrained(..., kernel_config=...)`.
- Pattern: `references/transformers_injection_example.py`; adapt the target class to the model-select block.
- **Mandatory** (`config.yaml: integration.assert_kernel_invoked`): `use_kernels` has historically not fired the kernel in some transformers versions — prove it actually runs inside `forward` (call counter > 0) before trusting any e2e number. Pin that transformers version in `config.yaml: env:`.
- Report both block-level speedup vs `torch.compile` and end-to-end model latency. A block win that doesn't move the model is reported honestly — it informs the next block choice.
