This repo uses opencode headless to write Triton kernels that beat `torch.compile` and build with [kernel-builder](https://github.com/huggingface/kernels/tree/main/kernel-builder). The agent is pointed at a config; the base code must be in a KernelBench-style format.

Install opencode and get a free key from https://opencode.ai/zen :

```
curl -fsSL https://opencode.ai/install | bash
```

Then:

```
python3 src/cli.py config --level 3 --problem 4
python3 src/cli.py setup  --config configs/<name>/config.json
python3 src/cli.py solve  --config configs/<name>/config.json
```

Full CLI reference: `docs/cli.md`. Architecture / phase-by-phase walkthrough: `guide.md`.
