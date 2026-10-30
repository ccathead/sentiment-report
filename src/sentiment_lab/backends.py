from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from .core import digest, object_hash


class BackendError(RuntimeError):
    def __init__(self, message, *, fatal=False, attempts=None):
        super().__init__(message)
        self.fatal = fatal
        self.attempts = attempts or []


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def image_data_url(path):
    suffix = Path(path).suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(suffix)
    if mime is None:
        raise ValueError("Unsupported API image format")
    return "data:" + mime + ";base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def api_payload(cfg, prompt, image_path):
    image_url = image_data_url(image_path) if image_path is not None else None
    if cfg["endpoint"] == "responses":
        content = [{"type": "input_text", "text": prompt}]
        if image_url:
            content.append({"type": "input_image", "image_url": image_url})
        body = {"model": cfg["model"], "input": [{"role": "user", "content": content}],
                "max_output_tokens": cfg.get("max_output_tokens", 512)}
    else:
        content = [{"type": "text", "text": prompt}]
        if image_url:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        body = {"model": cfg["model"], "messages": [{"role": "user", "content": content}],
                cfg.get("chat_token_parameter", "max_completion_tokens"): cfg.get("max_output_tokens", 512)}
    if cfg.get("temperature") is not None:
        body["temperature"] = cfg["temperature"]
    body.update(cfg.get("extra_body", {}))
    return body


def response_text(response, endpoint):
    if endpoint == "responses":
        return "".join(
            part.get("text", "") for item in response.get("output", []) if item.get("type") == "message"
            for part in item.get("content", []) if part.get("type") == "output_text"
        )
    choices = response.get("choices", [])
    if not choices:
        return ""
    content = choices[0].get("message", {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return ""


def clean_trace(value, key):
    """Remove accidental credential echoes and image payloads from persisted responses."""
    if isinstance(value, str):
        value = value.replace(key, "[REDACTED]") if key else value
        return re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[IMAGE_PAYLOAD]", value)
    if isinstance(value, list):
        return [clean_trace(v, key) for v in value]
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if k.lower() in {"authorization", "api_key", "access_token"}
                    else clean_trace(v, key)) for k, v in value.items()}
    return value


class DMXBackend:
    def __init__(self, cfg):
        self.cfg = cfg
        self.key = os.environ.get(cfg.get("api_key_env", "DMX_API_KEY"), "").strip()
        if not self.key:
            raise ValueError("DMX API key missing; set the configured environment variable or use --env-file .env")
        self.opener = urllib.request.build_opener(NoRedirect())
        self.last_call = None
        self.info = {"provider": "DMXAPI", "requested_model": cfg["model"],
                     "backend_snapshot": "unverified; inspect returned model and request id",
                     "endpoint": cfg["endpoint"]}

    def request(self, method, suffix, payload=None):
        attempts = []
        for n in range(self.cfg.get("max_retries", 2) + 1):
            if self.last_call is not None:
                pause = self.cfg.get("request_interval_seconds", 1) - (time.monotonic() - self.last_call)
                if pause > 0:
                    time.sleep(pause)
            request = urllib.request.Request(
                self.cfg["base_url"] + "/" + suffix, method=method,
                data=None if payload is None else json.dumps(payload).encode("utf-8"),
                headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json",
                         "User-Agent": "sentiment-report/0.1.0"},
            )
            started = time.monotonic()
            retry_after = None
            try:
                with self.opener.open(request, timeout=self.cfg.get("timeout_seconds", 120)) as result:
                    response = json.load(result)
                    attempts.append({"attempt": n + 1, "http_status": result.status,
                                     "seconds": time.monotonic() - started,
                                     "request_id": result.headers.get("x-request-id")})
                    self.last_call = time.monotonic()
                    return clean_trace(response, self.key), attempts
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                attempts.append({"attempt": n + 1, "http_status": exc.code,
                                 "seconds": time.monotonic() - started,
                                 "request_id": exc.headers.get("x-request-id")})
                retry_after = exc.headers.get("Retry-After")
                exc.close()
                if not retryable:
                    raise BackendError(f"HTTP {exc.code}; check model id, route, credentials and parameters",
                                       fatal=True, attempts=attempts) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
                attempts.append({"attempt": n + 1, "error_type": type(exc).__name__,
                                 "seconds": time.monotonic() - started})
            finally:
                self.last_call = time.monotonic()
            if n < self.cfg.get("max_retries", 2):
                delay = min(30, float(retry_after)) if retry_after and retry_after.isdigit() else min(30, 2 ** n)
                time.sleep(delay)
        raise BackendError("API retry budget exhausted", attempts=attempts)

    def infer(self, prompt, image_path):
        body = api_payload(self.cfg, prompt, image_path)
        response, attempts = self.request("POST", self.cfg["endpoint"], body)
        if "error" in response and response["error"]:
            raise BackendError("Provider returned an error object with HTTP success", fatal=True, attempts=attempts)
        return {"raw_answer": response_text(response, self.cfg["endpoint"]),
                "raw_response": response, "attempts": attempts, "usage": response.get("usage", {}),
                "returned_model": response.get("model"), "response_id": response.get("id"),
                "provider_status": response.get("status"),
                "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
                "request_body_sha256": object_hash(body)}


class HFBackend:
    def __init__(self, cfg):
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise ValueError("CUDA GPU not available; use the 4090 D machine for local models")
        revision = cfg.get("revision", "main")
        if not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise ValueError("Pin HF revision first with the lock-model command")
        self.cfg, self.torch = cfg, torch
        self.device = cfg.get("device", "cuda:0")
        torch.cuda.set_device(self.device)
        torch.manual_seed(cfg.get("seed", 20261030))
        torch.cuda.manual_seed_all(cfg.get("seed", 20261030))
        torch.backends.cudnn.benchmark = False
        self.dtype = getattr(torch, cfg.get("dtype", "float16"))
        kwargs = {}
        if cfg["architecture"] == "qwen2_5_vl":
            kwargs = {"min_pixels": cfg.get("min_pixels", 200704), "max_pixels": cfg.get("max_pixels", 1003520)}
        started = time.monotonic()
        self.processor = AutoProcessor.from_pretrained(cfg["model"], revision=revision, **kwargs)
        cls = LlavaForConditionalGeneration if cfg["architecture"] == "llava" else Qwen2_5_VLForConditionalGeneration
        self.model = cls.from_pretrained(cfg["model"], revision=revision, dtype=self.dtype,
                                         device_map={"": self.device}, low_cpu_mem_usage=True,
                                         attn_implementation="sdpa").eval()
        self.processor.tokenizer.padding_side = "left"
        if cfg["architecture"] == "llava":
            self.processor.patch_size = self.model.config.vision_config.patch_size
            self.processor.vision_feature_select_strategy = self.model.config.vision_feature_select_strategy
            self.processor.num_additional_image_tokens = 1
        torch.cuda.synchronize()
        template = self.processor.chat_template
        self.info = {"provider": "local-transformers", "model": cfg["model"], "revision": revision,
                     "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
                     "load_seconds": time.monotonic() - started,
                     "chat_template": template, "chat_template_sha256": object_hash(template),
                     "image_processor": self.processor.image_processor.to_dict(),
                     "do_sample": False, "num_beams": 1, "attention": "sdpa",
                     "determinism": "seeded greedy generation; hardware/library bitwise equality not guaranteed"}

    def infer(self, prompt, image_path):
        from PIL import Image, ImageOps

        content = []
        image = None
        if image_path is not None:
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[rendered], images=[image] if image is not None else None,
                                padding=True, return_tensors="pt")
        inputs = {k: v.to(device=self.device, dtype=self.dtype if v.is_floating_point() else v.dtype)
                  for k, v in inputs.items()}
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.cfg.get("max_new_tokens", 32),
                                         do_sample=False, num_beams=1, use_cache=True)
        self.torch.cuda.synchronize()
        new_ids = output[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return {"raw_answer": answer, "rendered_prompt": rendered,
                "input_tokens": inputs["input_ids"].shape[1], "generated_tokens": new_ids.shape[1],
                "peak_allocated_gib": self.torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gib": self.torch.cuda.max_memory_reserved() / 1024**3,
                "input_tensor_shapes": {k: list(v.shape) for k, v in inputs.items()}}


def make_backend(cfg):
    return HFBackend(cfg) if cfg["backend"] == "hf" else DMXBackend(cfg)
