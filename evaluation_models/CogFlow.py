#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union

import torch
from PIL import Image, ImageDraw

from vllm import LLM, SamplingParams


# ----------------------------
# 1) A simple SynVRs-style renderer (polyline + circle)
# ----------------------------
def render_synvrs_like(perception_text: str, img_size: int = 224, pad: int = 10) -> Optional[Image.Image]:
    """
    A very lightweight renderer for perception formats like:
      "(x1, y1) -- (x2, y2) -- (x3, y3)"
      "Circle: (cx, cy, r)"

    If parsing fails, return None.
    """
    lines = [ln.strip() for ln in perception_text.splitlines() if ln.strip()]
    points_all = []
    circles = []

    for ln in lines:
        if "Circle" in ln:
            m = re.findall(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)", ln)
            if m:
                cx, cy, rr = map(float, m[0])
                circles.append((cx, cy, rr))
            continue

        pts = re.findall(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)", ln)
        if len(pts) >= 2:
            pts_f = [(float(x), float(y)) for x, y in pts]
            points_all.append(pts_f)

    if not points_all and not circles:
        return None

    xs, ys = [], []
    for poly in points_all:
        for x, y in poly:
            xs.append(x)
            ys.append(y)
    for cx, cy, rr in circles:
        xs.extend([cx - rr, cx + rr])
        ys.extend([cy - rr, cy + rr])

    if not xs or not ys:
        return None

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = max(1e-6, x_max - x_min)
    h = max(1e-6, y_max - y_min)

    def to_xy(x, y):
        xx = pad + (x - x_min) / w * (img_size - 2 * pad)
        yy = pad + (y - y_min) / h * (img_size - 2 * pad)
        return xx, yy

    img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for poly in points_all:
        xy = [to_xy(x, y) for x, y in poly]
        draw.line(xy, width=2, fill=(0, 0, 0))

    for cx, cy, rr in circles:
        cx2, cy2 = to_xy(cx, cy)
        r_pix = rr / w * (img_size - 2 * pad)
        box = [cx2 - r_pix, cy2 - r_pix, cx2 + r_pix, cy2 + r_pix]
        draw.ellipse(box, outline=(0, 0, 0), width=2)

    return img


# ----------------------------
# 2) CLIP-based SVSR scorer (fallback)
# ----------------------------
class VSRScorer:
    """
    SVSR(I_hat, I): normalized cosine similarity using a frozen encoder.
    Paper uses FG-CLIP; here we provide a default CLIP fallback.
    """

    def __init__(self, device: str = "cuda", clip_name: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPProcessor, CLIPModel

        self.device = device
        self.processor = CLIPProcessor.from_pretrained(clip_name)
        self.model = CLIPModel.from_pretrained(clip_name).to(device)
        self.model.eval()

    @torch.inference_mode()
    def score(self, img_a: Image.Image, img_b: Image.Image) -> float:
        inputs = self.processor(images=[img_a, img_b], return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        cos = (feats[0] * feats[1]).sum().clamp(-1, 1)
        return float((cos + 1) * 0.5)


@dataclass
class GateConfig:
    tau: float = 0.75
    max_trials: int = 3
    perception_max_new_tokens: int = 128
    reasoning_max_new_tokens: int = 256
    stop_strings: Optional[List[str]] = None
    continue_prefix: str = "<REASONING>\n"

    def __post_init__(self):
        if self.stop_strings is None:
            self.stop_strings = ["<REASONING>"]


class CogFlow:
    """
    Class wrapper for CogFlow inference.

    It keeps the original two-stage logic:
      1. perception generation + visual gate scoring
      2. reasoning continuation

    Added compatibility method:
      - get_response(image, query)

    This makes it easy to use in code such as:
        from models import CogFlow
        model = CogFlow.CogFlow(model_path="/path/to/model")
        response = model.get_response(image_path, query)
    """

    def __init__(
        self,
        model_path: str,
        tp: int = 1,
        gpu_mem_util: float = 0.6,
        max_model_len: int = 2048,
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        tau: float = 0.75,
        max_trials: int = 3,
        perception_max_new_tokens: int = 128,
        reasoning_max_new_tokens: int = 256,
        stop_strings: Optional[Union[str, List[str]]] = None,
        continue_prefix: str = "<REASONING>\n",
        temperature: float = 0.7,
        top_p: float = 0.95,
        seed: int = 1234,
        clip_name: str = "openai/clip-vit-base-patch32",
        render_debug_dir: Optional[str] = None,
    ):
        self.model_path = model_path
        self.tp = tp
        self.gpu_mem_util = gpu_mem_util
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.clip_name = clip_name
        self.render_debug_dir = render_debug_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if isinstance(stop_strings, str):
            stop_strings = json.loads(stop_strings)
        elif stop_strings is None:
            stop_strings = ["<REASONING>"]

        self.gate_cfg = GateConfig(
            tau=tau,
            max_trials=max_trials,
            perception_max_new_tokens=perception_max_new_tokens,
            reasoning_max_new_tokens=reasoning_max_new_tokens,
            stop_strings=stop_strings,
            continue_prefix=continue_prefix,
        )

        if self.render_debug_dir:
            os.makedirs(self.render_debug_dir, exist_ok=True)

        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tp,
            gpu_memory_utilization=self.gpu_mem_util,
            max_model_len=self.max_model_len,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            enforce_eager=False,
        )

        self.vsr = VSRScorer(device=self.device, clip_name=self.clip_name) if self.device == "cuda" else None
        self.last_result: Optional[Dict[str, Any]] = None

    @staticmethod
    def load_jsonl(path: str) -> List[Dict[str, Any]]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    @staticmethod
    def save_jsonl(rows: List[Dict[str, Any]], path: str):
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _load_image(image: Union[str, Image.Image, None]) -> Optional[Image.Image]:
        if image is None:
            return None
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"image_path not found: {image}")
            return Image.open(image).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _generate_once(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        max_tokens: int = 256,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> str:
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=stop,
            detokenize=True,
            seed=self.seed if seed is None else seed,
        )

        if image is None:
            out = self.llm.generate(
                prompts=[prompt],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
        else:
            out = self.llm.generate(
                prompts=[{"prompt": prompt, "multi_modal_data": {"image": image}}],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
        return out[0].outputs[0].text

    def _score_perception(
        self,
        ex_id: Any,
        trial_idx: int,
        perception_text: str,
        orig_img: Optional[Image.Image],
    ) -> float:
        if orig_img is None:
            return -1.0

        rend = render_synvrs_like(perception_text)
        if rend is None or self.vsr is None:
            score = -1.0
        else:
            score = self.vsr.score(rend, orig_img)

        if self.render_debug_dir and rend is not None:
            rend.save(
                os.path.join(
                    self.render_debug_dir,
                    f"{ex_id}_trial{trial_idx}_score{score:.3f}.png",
                )
            )
        return score

    def run_one(
        self,
        prompt: str,
        image: Union[str, Image.Image, None] = None,
        ex_id: Any = None,
    ) -> Dict[str, Any]:
        orig_img = self._load_image(image)
        ex_id = 0 if ex_id is None else ex_id

        # Text-only fallback: skip visual gate and directly reason.
        if orig_img is None:
            final_text = self._generate_once(
                prompt=prompt,
                image=None,
                max_tokens=self.gate_cfg.reasoning_max_new_tokens,
                seed=self.seed + 999,
            )
            result = {
                "id": ex_id,
                "image_path": image if isinstance(image, str) else None,
                "tau": self.gate_cfg.tau,
                "chosen_trial": 0,
                "vsr_score": None,
                "perception": "",
                "final": final_text,
            }
            self.last_result = result
            return result

        best = {"score": -1e9, "perception": "", "trial": -1}
        accepted = None

        for t in range(1, self.gate_cfg.max_trials + 1):
            perception_text = self._generate_once(
                prompt=prompt,
                image=orig_img,
                max_tokens=self.gate_cfg.perception_max_new_tokens,
                stop=self.gate_cfg.stop_strings,
                seed=self.seed + t,
            )

            score = self._score_perception(
                ex_id=ex_id,
                trial_idx=t,
                perception_text=perception_text,
                orig_img=orig_img,
            )

            if score > best["score"]:
                best = {"score": score, "perception": perception_text, "trial": t}

            if score >= self.gate_cfg.tau:
                accepted = {"score": score, "perception": perception_text, "trial": t}
                break

        chosen = accepted if accepted is not None else best
        reasoning_prompt = prompt + "\n" + chosen["perception"] + "\n" + self.gate_cfg.continue_prefix
        final_text = self._generate_once(
            prompt=reasoning_prompt,
            image=orig_img,
            max_tokens=self.gate_cfg.reasoning_max_new_tokens,
            seed=self.seed + 999,
        )

        result = {
            "id": ex_id,
            "image_path": image if isinstance(image, str) else None,
            "tau": self.gate_cfg.tau,
            "chosen_trial": chosen["trial"],
            "vsr_score": chosen["score"],
            "perception": chosen["perception"],
            "final": final_text,
        }
        self.last_result = result
        return result

    def get_response(self, image: Union[str, Image.Image, None], query: str) -> str:
        """
        Compatibility interface for the second script.

        Args:
            image: image path, PIL image, or None
            query: text prompt/query

        Returns:
            final answer string
        """
        result = self.run_one(prompt=query, image=image)
        return result["final"]

    def run_jsonl(self, input_jsonl: str, output_jsonl: str):
        rows = self.load_jsonl(input_jsonl)
        outputs = []
        for i, ex in enumerate(rows):
            ex_id = ex.get("id", i)
            prompt = ex["prompt"]
            image_path = ex.get("image_path", None)
            rec = self.run_one(prompt=prompt, image=image_path, ex_id=ex_id)
            outputs.append(rec)
            print(
                f"[{ex_id}] chosen_trial={rec['chosen_trial']} "
                f"vsr={rec['vsr_score']}"
            )
        self.save_jsonl(outputs, output_jsonl)
        print(f"Saved -> {output_jsonl}")


# Optional alias to tolerate different capitalizations in old code.
Cogflow = CogFlow


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--input_jsonl", type=str, required=True,
                    help="Each line: {'id':..., 'prompt':..., 'image_path':...}")
    ap.add_argument("--output_jsonl", type=str, required=True)

    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem_util", type=float, default=0.6)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--tau", type=float, default=0.75)
    ap.add_argument("--max_trials", type=int, default=3)
    ap.add_argument("--perception_max_new_tokens", type=int, default=128)
    ap.add_argument("--reasoning_max_new_tokens", type=int, default=256)
    ap.add_argument("--stop_strings", type=str, default='["<REASONING>"]')
    ap.add_argument("--continue_prefix", type=str, default="<REASONING>\n")

    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--render_debug_dir", type=str, default=None)
    return ap


def main():
    args = build_parser().parse_args()
    model = CogFlow(
        model_path=args.model_path,
        tp=args.tp,
        gpu_mem_util=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        tau=args.tau,
        max_trials=args.max_trials,
        perception_max_new_tokens=args.perception_max_new_tokens,
        reasoning_max_new_tokens=args.reasoning_max_new_tokens,
        stop_strings=args.stop_strings,
        continue_prefix=args.continue_prefix,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        clip_name=args.clip_name,
        render_debug_dir=args.render_debug_dir,
    )
    model.run_jsonl(args.input_jsonl, args.output_jsonl)


if __name__ == "__main__":
    main()
