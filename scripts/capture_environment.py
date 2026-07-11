#!/usr/bin/env python3
"""Capture the current reproduction environment without changing packages."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="environment_current.json")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    data: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model": args.model,
    }
    try:
        import torch
        data.update({
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except Exception as exc:
        data["torch_error"] = repr(exc)
    try:
        import transformers
        data["transformers"] = transformers.__version__
    except Exception as exc:
        data["transformers_error"] = repr(exc)
    try:
        import huggingface_hub
        data["huggingface_hub"] = huggingface_hub.__version__
    except Exception as exc:
        data["huggingface_hub_error"] = repr(exc)
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
        data["model_revision"] = getattr(cfg, "_commit_hash", None)
    except Exception as exc:
        data["model_revision_error"] = repr(exc)

    out = Path(args.output)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
