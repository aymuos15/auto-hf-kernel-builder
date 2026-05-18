# Canonical entrypoints. Everything specific is in config.yaml.
# Tiers: A provision (AGENTS.md SETUP, human) · B prep (once/target) · C loop.
.ONESHELL:
.DEFAULT_GOAL := help
SHELL := bash
CONFIG ?= configs/config.yaml

# Derive host/image/slug from config (config is the only source of truth).
HOST  := $(shell sed -nE 's/^[[:space:]]*ssh:[[:space:]]*"?ssh ([^" #]+).*/\1/p' $(CONFIG) | head -1)
IMAGE := $(shell sed -nE 's/^[[:space:]]*container:[[:space:]]*([^ #]+).*/\1/p' $(CONFIG) | head -1)
BASE  := $(shell sed -nE 's/^[[:space:]]*base_container:[[:space:]]*([^ #]+).*/\1/p' $(CONFIG) | head -1)
MODEL := $(shell sed -nE 's/^[[:space:]]*id:[[:space:]]*([^ #]+).*/\1/p' $(CONFIG) | head -1)
SLUG  := $(subst /,__,$(MODEL))

help:
	@echo "agentic-kernels — config: $(CONFIG)  model: $(MODEL)  host: $(HOST)"
	@echo "  make sync     rsync repo to the authoritative host"
	@echo "  make image    build derived container (kernels baked) — ONCE on host"
	@echo "  make verify   Spark de-risk: container (torch/triton/GPU) + host (Nix/kernel-builder)"
	@echo "  make prep     B: profile -> rank -> freeze -> headroom (once/target)"
	@echo "  make loop     C: autonomous kernel-opt loop (requires prep)"
	@echo "  make clean    remove regenerable local artifacts"

sync:
	rsync -az --delete --exclude .git --exclude targets --exclude __pycache__ \
	  -e ssh ./ $(HOST):'~/agentic-kernels/'

# One-time (per host / per base bump): bake the kernels client into the NGC
# base so the ephemeral loop container never reinstalls it. Idempotent
# (Docker layer cache). The loop just uses $(IMAGE) afterwards.
image: sync
	ssh $(HOST) 'cd ~/agentic-kernels && docker build --build-arg BASE=$(BASE) \
	  -t $(IMAGE) -f Dockerfile . && \
	  docker run --rm $(IMAGE) python3 -c "import kernels;print(\"image OK\",kernels.__version__)"'

# Spark de-risk (replaces the old derisk/ dir). TWO SURFACES:
#  container = torch/triton/GPU + kernels client ; host = Nix + kernel-builder.
verify: sync
	@echo "=== container surface ($(IMAGE)) ==="
	ssh $(HOST) 'docker run --rm --gpus all --ipc=host -v $$HOME/agentic-kernels:/work -w /work $(IMAGE) \
	  bash -lc "python3 - <<PY
	import torch, triton
	assert torch.cuda.is_available()
	print(\"container OK:\", torch.__version__, \"triton\", triton.__version__, \
	      torch.cuda.get_device_name(0), \".\".join(map(str,torch.cuda.get_device_capability(0))))
	PY
	python3 -c \"import kernels; print(\\\"kernels baked\\\", kernels.__version__)\" \
	  || { echo \"kernels NOT in image — run: make image\"; exit 1; }"'
	@echo "=== host surface (Nix / kernel-builder, aarch64-linux) ==="
	ssh $(HOST) 'command -v nix >/dev/null && \
	  nix --extra-experimental-features "nix-command flakes" \
	    eval github:huggingface/kernel-builder#packages.aarch64-linux.build2cmake.name \
	  && echo "host OK: kernel-builder usable for aarch64-linux"'

prep:
	python3 pipeline/driver.py --phase prep --config $(CONFIG)

loop:
	./launch.sh --config $(CONFIG)

clean:
	rm -rf pipeline/__pycache__ skills/*/scripts/__pycache__ \
	  targets/$(SLUG)/profile.json targets/$(SLUG)/*/inputs.pt \
	  targets/$(SLUG)/*/golden.pt targets/$(SLUG)/*/inductor \
	  targets/$(SLUG)/*/loop_history.json

.PHONY: help sync image verify prep loop clean
