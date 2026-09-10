from __future__ import annotations

import json
import os
import urllib.request
from typing import Protocol

class TextBackend(Protocol):
    def generate(self, system: str, prompt: str) -> str: ...

class DemoBackend:

    def generate(self, system: str, prompt: str) -> str:
        if system.startswith("Distill"):
            record = json.loads(prompt)
            return json.dumps({
                "core": f"Demonstration memory for {record['tag']}.",
                "conditions": "Use only when the task matches this capability.",
                "procedure": "Inspect the observation and available context before answering.",
            })
        return "A"

class APIBackend:

    def __init__(self, base_url: str, model: str, key_env: str = "FEDMEVO_API_KEY",
                 max_tokens: int = 512, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.key = os.environ[key_env]
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(self, system: str, prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/chat/completions", data=body,
            headers={"Authorization": "Bearer " + self.key,
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)["choices"][0]["message"]["content"]

class LocalBackend:

    def __init__(self, model_path: str, device: str = "cpu", max_tokens: int = 512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        ).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        self.max_tokens = max_tokens

    def generate(self, system: str, prompt: str) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

def add_backend_arguments(parser):
    parser.add_argument("--backend", choices=["demo", "api", "local"], default="demo")
    parser.add_argument("--base-url", help="Chat-completions API root, including /v1")
    parser.add_argument("--model", help="API model identifier or local checkpoint directory")
    parser.add_argument("--key-env", default="FEDMEVO_API_KEY")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-tokens", type=int, default=512)

def make_backend(args):
    if args.backend == "demo":
        return DemoBackend()
    if not args.model:
        raise ValueError("--model is required for api and local backends")
    if args.backend == "api":
        if not args.base_url:
            raise ValueError("--base-url is required for the api backend")
        return APIBackend(args.base_url, args.model, args.key_env, args.max_tokens)
    return LocalBackend(args.model, args.device, args.max_tokens)
