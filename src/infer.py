#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

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

    # polyline: parse "(x,y)" pairs
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

    # Normalize coords into canvas
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
        # map to [pad, img_size-pad]
        xx = pad + (x - x_min) / w * (img_size - 2 * pad)
        yy = pad + (y - y_min) / h * (img_size - 2 * pad)
        return xx, yy

    img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # draw polylines
    for poly in points_all:
        xy = [to_xy(x, y) for x, y in poly]
        draw.line(xy, width=2, fill=(0, 0, 0))

    # draw circles
    for cx, cy, rr in circles:
        cx2, cy2 = to_xy(cx, cy)
        # scale radius by x-axis
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
        feats = self.model.get_image_features(**inputs)  # (2, d)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        cos = (feats[0] * feats[1]).sum().clamp(-1, 1)
        # normalized to [0,1]
        return float((cos + 1) * 0.5)


# ----------------------------
# 3) Visual Gate inference (perception -> VSR -> reasoning)
# ----------------------------
@dataclass
class GateConfig:
    tau: float
    max_trials: int
    perception_max_new_tokens: int
    reasoning_max_new_tokens: int
    stop_strings: List[str]
    continue_prefix: str


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
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

    # gate
    ap.add_argument("--tau", type=float, default=0.75)
    ap.add_argument("--max_trials", type=int, default=3)
    ap.add_argument("--perception_max_new_tokens", type=int, default=128)
    ap.add_argument("--reasoning_max_new_tokens", type=int, default=256)
    ap.add_argument("--stop_strings", type=str, default='["<REASONING>"]')
    ap.add_argument("--continue_prefix", type=str, default="<REASONING>\n")

    # sampling
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1234)

    # vsr
    ap.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--render_debug_dir", type=str, default=None)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    stop_strings = json.loads(args.stop_strings)

    gate_cfg = GateConfig(
        tau=args.tau,
        max_trials=args.max_trials,
        perception_max_new_tokens=args.perception_max_new_tokens,
        reasoning_max_new_tokens=args.reasoning_max_new_tokens,
        stop_strings=stop_strings,
        continue_prefix=args.continue_prefix,
    )

    if args.render_debug_dir:
        os.makedirs(args.render_debug_dir, exist_ok=True)

    # vLLM
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=False,
    )

    # VSR scorer
    vsr = VSRScorer(device=device, clip_name=args.clip_name) if device == "cuda" else None

    rows = load_jsonl(args.input_jsonl)

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for i, ex in enumerate(rows):
            ex_id = ex.get("id", i)
            prompt = ex["prompt"]
            image_path = ex.get("image_path", None)

            if image_path is None or not os.path.exists(image_path):
                raise FileNotFoundError(f"[{ex_id}] image_path not found: {image_path}")

            orig_img = Image.open(image_path).convert("RGB")

            best = {"score": -1e9, "perception": "", "trial": -1}
            accepted = None

            # ---------------------
            # Stage-1: perception with visual gate
            # ---------------------
            for t in range(1, gate_cfg.max_trials + 1):
                sp = SamplingParams(
                    max_tokens=gate_cfg.perception_max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop=gate_cfg.stop_strings,
                    detokenize=True,
                    seed=args.seed + t,
                )

                out = llm.generate(
                    prompts=[{"prompt": prompt, "multi_modal_data": {"image": orig_img}}],
                    sampling_params=sp,
                    use_tqdm=False,
                )
                perception_text = out[0].outputs[0].text

                # render predicted perception
                rend = render_synvrs_like(perception_text)
                if rend is None or vsr is None:
                    score = -1.0
                else:
                    score = vsr.score(rend, orig_img)

                if score > best["score"]:
                    best = {"score": score, "perception": perception_text, "trial": t}

                if args.render_debug_dir and rend is not None:
                    rend.save(os.path.join(args.render_debug_dir, f"{ex_id}_trial{t}_score{score:.3f}.png"))

                if score >= gate_cfg.tau:
                    accepted = {"score": score, "perception": perception_text, "trial": t}
                    break

            chosen = accepted if accepted is not None else best

            # ---------------------
            # Stage-2: reasoning continuation
            # ---------------------
            reasoning_prompt = prompt + "\n" + chosen["perception"] + "\n" + gate_cfg.continue_prefix

            sp2 = SamplingParams(
                max_tokens=gate_cfg.reasoning_max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                detokenize=True,
                seed=args.seed + 999,
            )

            out2 = llm.generate(
                prompts=[{"prompt": reasoning_prompt, "multi_modal_data": {"image": orig_img}}],
                sampling_params=sp2,
                use_tqdm=False,
            )
            final_text = out2[0].outputs[0].text

            rec = {
                "id": ex_id,
                "image_path": image_path,
                "tau": gate_cfg.tau,
                "chosen_trial": chosen["trial"],
                "vsr_score": chosen["score"],
                "perception": chosen["perception"],
                "final": final_text,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(f"[{ex_id}] chosen_trial={chosen['trial']} vsr={chosen['score']:.3f} accepted={accepted is not None}")

    print(f"Saved -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
