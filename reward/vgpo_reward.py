# verl_custom_rewards.py

import re
import math
import asyncio
from typing import Any, Optional, Tuple, List

import numpy as np

# Optional dependency
try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ============================================================
# Utilities
# ============================================================
def extract_xml_block(text: str, tag: str) -> str:
    if not text:
        return ""
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def _to_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return ""


def _get_gt_text(ground_truth: Any, keys=("solution", "answer", "target")) -> str:
    if isinstance(ground_truth, str):
        return ground_truth
    if isinstance(ground_truth, dict):
        for k in keys:
            if k in ground_truth and ground_truth[k] is not None:
                return _to_text(ground_truth[k])
    return _to_text(ground_truth)


def parse_mc_choice(text: str) -> Optional[str]:
    if not text:
        return None

    ans = extract_xml_block(text, "ANSWER")
    if ans:
        m = re.search(r"\b([A-D])\b", ans, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    m = re.search(r"<ANSWER>\s*([A-D])\s*</ANSWER>", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([A-D])\b", text)
    if m:
        return m.group(1).upper()

    return None


def mm_watch_think_answer_format_score(
    data_source, solution_str, ground_truth, extra_info=None, **kwargs
) -> float:
    if not solution_str:
        return 0.0

    pattern = (
        r"^\s*<WATCHING>.*?</WATCHING>\s*"
        r"<THINKING>.*?</THINKING>\s*"
        r"<ANSWER>.*?</ANSWER>\s*$"
    )
    return 1.0 if re.match(pattern, solution_str, flags=re.IGNORECASE | re.DOTALL) else 0.0


# ============================================================
# VPR (line-set matching reward)
# ============================================================
class _ParaRewardCore:
    THETA_MAX_DEG = 6.0
    OVERLAP_MIN = 0.20
    W_ANGLE = 1.0
    W_OFFSET = 1.0
    W_OVERLAP = 2.0
    W_HAUSDORFF = 1.0
    COST_LARGE = 1e6
    COST_GATE = 1e5

    def extract_lines_by_points_only(self, text: str) -> List[List[Tuple[float, float]]]:
        if not text:
            return []

        watching = extract_xml_block(text, "WATCHING")
        if watching:
            text = watching

        if "Line:" not in text:
            return []

        if "Circle:" in text:
            text = text.split("Circle:", 1)[0]

        block = text.split("Line:", 1)[1]
        lines = [ln.strip() for ln in block.strip().split("\n") if ln.strip()]

        out = []
        for ln in lines:
            pts = re.findall(r"\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)", ln)
            if not pts:
                continue
            out.append([(float(x), float(y)) for x, y in pts])
        return out

    def _fit_and_norm_line(self, pts: List[Tuple[float, float]]):
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)

        if np.isclose(xs.var(), 0.0, atol=1e-12):
            A, B, C = 1.0, 0.0, -float(xs.mean())
        else:
            m, b = np.polyfit(xs, ys, 1)
            A, B, C = -float(m), 1.0, -float(b)

        norm = math.hypot(A, B)
        if norm == 0:
            A, B, C = 0.0, 1.0, 0.0
        else:
            A, B, C = A / norm, B / norm, C / norm

        if A < 0 or (np.isclose(A, 0.0) and B < 0):
            A, B, C = -A, -B, -C

        p0 = np.array(pts[0], dtype=float)
        p1 = np.array(pts[-1], dtype=float)
        dir_vec = p1 - p0

        if np.allclose(dir_vec, 0.0):
            for i in range(1, len(pts)):
                dir_vec = np.array(pts[i], dtype=float) - p0
                if not np.allclose(dir_vec, 0.0):
                    break

        if np.allclose(dir_vec, 0.0):
            dir_vec = np.array([-B, A], dtype=float)

        u = dir_vec / (np.linalg.norm(dir_vec) + 1e-12)
        return {"A": A, "B": B, "C": C, "p0": p0, "p1": p1, "dir": u}

    def _angle_diff(self, l1, l2) -> float:
        n1 = np.array([l1["A"], l1["B"]])
        n2 = np.array([l2["A"], l2["B"]])
        cosv = abs(float(np.dot(n1, n2))) / (np.linalg.norm(n1) * np.linalg.norm(n2) + 1e-12)
        cosv = max(0.0, min(1.0, cosv))
        return math.acos(cosv)

    def _overlap_ratio(self, a0, a1, b0, b1, u):
        t1 = sorted([float(np.dot(a0, u)), float(np.dot(a1, u))])
        t2 = sorted([float(np.dot(b0, u)), float(np.dot(b1, u))])
        inter = max(0.0, min(t1[1], t2[1]) - max(t1[0], t2[0]))
        union = (t1[1] - t1[0]) + (t2[1] - t2[0]) - inter
        return inter / union if union > 0 else 0.0

    def _hausdorff_2seg(self, a0, a1, b0, b1):
        def p2seg(p, s0, s1):
            ap = p - s0
            ab = s1 - s0
            denom = float(np.dot(ab, ab))
            if denom <= 1e-12:
                return float(np.linalg.norm(ap))
            t = float(np.dot(ap, ab) / denom)
            t = max(0.0, min(1.0, t))
            proj = s0 + t * ab
            return float(np.linalg.norm(p - proj))

        d = max(
            max(p2seg(a0, b0, b1), p2seg(a1, b0, b1)),
            max(p2seg(b0, a0, a1), p2seg(b1, a0, a1)),
        )
        return d

    def _build_cost_matrix(self, preds, gts):
        M = np.full((len(preds), len(gts)), self.COST_LARGE, dtype=float)
        theta_gate = math.radians(self.THETA_MAX_DEG)

        for i, p in enumerate(preds):
            for j, g in enumerate(gts):
                dtheta = self._angle_diff(p, g)
                if dtheta > theta_gate:
                    continue

                dC = abs(p["C"] - g["C"])
                u = g["dir"]
                overlap = self._overlap_ratio(p["p0"], p["p1"], g["p0"], g["p1"], u)
                if overlap < self.OVERLAP_MIN:
                    continue

                h = self._hausdorff_2seg(p["p0"], p["p1"], g["p0"], g["p1"])

                cost = (
                    self.W_ANGLE * dtheta
                    + self.W_OFFSET * dC
                    + self.W_OVERLAP * (1.0 - overlap)
                    + self.W_HAUSDORFF * h
                )
                M[i, j] = float(cost)
        return M

    def _match_and_metrics(self, M: np.ndarray, n_pred: int, n_gt: int):
        if _HAS_SCIPY:
            rows, cols = linear_sum_assignment(M)
            pairs = [(r, c) for r, c in zip(rows, cols) if M[r, c] < self.COST_GATE]
        else:
            pairs = []
            used_r, used_c = set(), set()
            flat = [(M[r, c], r, c) for r in range(n_pred) for c in range(n_gt)]
            flat.sort(key=lambda x: x[0])
            for cost, r, c in flat:
                if cost >= self.COST_GATE:
                    break
                if r in used_r or c in used_c:
                    continue
                used_r.add(r)
                used_c.add(c)
                pairs.append((r, c))

        TP = len(pairs)
        FP = n_pred - TP
        FN = n_gt - TP
        P = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        R = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        F1 = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0
        return pairs, {"TP": TP, "FP": FP, "FN": FN, "P": P, "R": R, "F1": F1}

    def score(self, pred_text: str, gt_text: str) -> float:
        pred_lines_pts = self.extract_lines_by_points_only(pred_text)
        gt_lines_pts = self.extract_lines_by_points_only(gt_text)

        pred_lines = [self._fit_and_norm_line(pts) for pts in pred_lines_pts if len(pts) >= 2]
        gt_lines = [self._fit_and_norm_line(pts) for pts in gt_lines_pts if len(pts) >= 2]

        if len(pred_lines) == 0 and len(gt_lines) == 0:
            return 1.0
        if len(pred_lines) == 0 or len(gt_lines) == 0:
            return 0.0

        C = self._build_cost_matrix(pred_lines, gt_lines)
        _, metrics = self._match_and_metrics(C, len(pred_lines), len(gt_lines))
        return float(metrics["F1"])


_PARA = _ParaRewardCore()


def para_reward_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    gt_text = _get_gt_text(ground_truth, keys=("solution", "gt", "ground_truth"))
    return _PARA.score(solution_str, gt_text)


import os
import hashlib
from PIL import Image, ImageDraw

# ============================================================
# VSR: Visual Similarity Reward (CLIP cosine similarity)
# ============================================================

_VSR_MODEL = None
_VSR_PROCESSOR = None
_VSR_DEVICE = None
_VSR_GT_CACHE = {}   # path -> embedding (np.ndarray)
_VSR_MAX_CACHE = 5000


def _get_first_image_path(extra_info, ground_truth) -> str:
    """
    Try to find GT image path from extra_info / ground_truth.
    """
    # 1) extra_info
    if isinstance(extra_info, dict):
        for k in ["raw_image", "image", "img", "images", "raw_images"]:
            if k in extra_info and extra_info[k] is not None:
                v = extra_info[k]
                if isinstance(v, str):
                    return v
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str):
                    return v[0]

    # 2) ground_truth
    if isinstance(ground_truth, dict):
        for k in ["image", "img", "images"]:
            if k in ground_truth and ground_truth[k] is not None:
                v = ground_truth[k]
                if isinstance(v, str):
                    return v
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str):
                    return v[0]

    return ""


def _load_vsr_model(model_name: str = "qihoo360/fg-clip-large"):
    """
    Lazy-load CLIP model and processor.
    """
    global _VSR_MODEL, _VSR_PROCESSOR, _VSR_DEVICE

    if _VSR_MODEL is not None and _VSR_PROCESSOR is not None:
        return

    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except Exception:
        _VSR_MODEL, _VSR_PROCESSOR, _VSR_DEVICE = None, None, None
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _VSR_DEVICE = device

    _VSR_PROCESSOR = CLIPProcessor.from_pretrained(model_name)
    _VSR_MODEL = CLIPModel.from_pretrained(model_name).to(device)
    _VSR_MODEL.eval()


def _encode_image_clip(pil_img: Image.Image) -> Optional[np.ndarray]:
    """
    Encode an image into a normalized embedding vector.
    """
    _load_vsr_model()
    if _VSR_MODEL is None or _VSR_PROCESSOR is None:
        return None

    try:
        import torch
        inputs = _VSR_PROCESSOR(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(_VSR_DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            feats = _VSR_MODEL.get_image_features(**inputs)  # [1, D]
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
        return feats[0].detach().cpu().float().numpy()
    except Exception:
        return None


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def _cache_put(cache: dict, key: str, value: np.ndarray):
    if key in cache:
        return
    if len(cache) >= _VSR_MAX_CACHE:
        # Simple eviction: pop one arbitrary item
        cache.pop(next(iter(cache)))
    cache[key] = value


def _parse_watching_geometry(watching: str):
    """
    Parse geometry from WATCHING text:
      Point:
        A: (x, y)
      Line:
        (x, y) -- (x, y) ...
      Circle:
        (cx, cy, r)

    Return dict with points, polylines, circles.
    """
    pts = []
    polylines = []
    circles = []

    # Points section
    # Example: A: (-8.05, -0.95)
    for m in re.finditer(r"([A-Za-z0-9_]+)\s*:\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)", watching):
        x = float(m.group(2))
        y = float(m.group(3))
        pts.append((x, y))

    # Line section: each line contains multiple (x,y)
    if "Line:" in watching:
        line_block = watching.split("Line:", 1)[1]
        if "Circle:" in line_block:
            line_block = line_block.split("Circle:", 1)[0]

        for ln in [s.strip() for s in line_block.strip().split("\n") if s.strip()]:
            p = re.findall(r"\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)", ln)
            if len(p) >= 2:
                poly = [(float(x), float(y)) for x, y in p]
                polylines.append(poly)
                pts.extend(poly)

    # Circle section
    if "Circle:" in watching:
        circ_block = watching.split("Circle:", 1)[1]
        for ln in [s.strip() for s in circ_block.strip().split("\n") if s.strip()]:
            m = re.search(r"\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)", ln)
            if m:
                cx, cy, r = float(m.group(1)), float(m.group(2)), float(m.group(3))
                circles.append((cx, cy, r))
                # add bbox points for scaling
                pts.append((cx - r, cy - r))
                pts.append((cx + r, cy + r))

    return {"points": pts, "polylines": polylines, "circles": circles}


def _render_watching_to_image(watching: str, size: int = 512, pad: int = 32) -> Optional[Image.Image]:
    """
    Render WATCHING geometry into a synthetic diagram image.
    """
    geo = _parse_watching_geometry(watching)
    pts = geo["points"]
    polylines = geo["polylines"]
    circles = geo["circles"]

    if len(pts) == 0 and len(polylines) == 0 and len(circles) == 0:
        return None

    xs = [p[0] for p in pts] if pts else [0.0]
    ys = [p[1] for p in pts] if pts else [0.0]

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    # Avoid degenerate bbox
    if abs(maxx - minx) < 1e-6:
        maxx += 1.0
        minx -= 1.0
    if abs(maxy - miny) < 1e-6:
        maxy += 1.0
        miny -= 1.0

    # Add margin
    dx = maxx - minx
    dy = maxy - miny
    minx -= 0.10 * dx
    maxx += 0.10 * dx
    miny -= 0.10 * dy
    maxy += 0.10 * dy

    def map_pt(x, y):
        # map to [pad, size-pad]
        ux = (x - minx) / (maxx - minx + 1e-12)
        uy = (y - miny) / (maxy - miny + 1e-12)
        px = pad + ux * (size - 2 * pad)
        py = pad + (1.0 - uy) * (size - 2 * pad)  # flip y
        return (px, py)

    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw circles
    for (cx, cy, r) in circles:
        x0, y0 = map_pt(cx - r, cy - r)
        x1, y1 = map_pt(cx + r, cy + r)
        draw.ellipse([x0, y0, x1, y1], outline=(0, 0, 0), width=2)

    # Draw lines
    for poly in polylines:
        pxy = [map_pt(x, y) for (x, y) in poly]
        if len(pxy) >= 2:
            draw.line(pxy, fill=(0, 0, 0), width=2)

    # Draw points
    # Use all unique points from lines + points parsing
    uniq = list({(round(x, 6), round(y, 6)) for (x, y) in pts})
    for (x, y) in uniq:
        px, py = map_pt(x, y)
        rr = 3
        draw.ellipse([px - rr, py - rr, px + rr, py + rr], fill=(0, 0, 0))

    return img


def vsr_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """
    Visual Similarity Reward:
      - GT embedding from dataset image
      - Pred embedding from rendered WATCHING diagram
      - reward = cosine similarity mapped to [0, 1]
    """
    if not solution_str:
        return 0.0

    # Require correct tag format; you can remove this if you want
    if mm_watch_think_answer_format_score(data_source, solution_str, ground_truth, extra_info, **kwargs) <= 0.0:
        return 0.0

    # 1) Get GT image path
    gt_path = _get_first_image_path(extra_info, ground_truth)
    if not gt_path:
        return 0.0
    if not os.path.exists(gt_path):
        return 0.0

    # 2) Render predicted image from WATCHING
    watching = extract_xml_block(solution_str, "WATCHING")
    if not watching:
        return 0.0

    pred_img = _render_watching_to_image(watching, size=512, pad=32)
    if pred_img is None:
        return 0.0

    # 3) Encode embeddings
    # GT embedding cache
    gt_key = gt_path
    if gt_key in _VSR_GT_CACHE:
        gt_emb = _VSR_GT_CACHE[gt_key]
    else:
        try:
            gt_img = Image.open(gt_path).convert("RGB")
        except Exception:
            return 0.0
        gt_emb = _encode_image_clip(gt_img)
        if gt_emb is None:
            return 0.0
        _cache_put(_VSR_GT_CACHE, gt_key, gt_emb)

    pred_emb = _encode_image_clip(pred_img)
    if pred_emb is None:
        return 0.0

    # 4) Cosine similarity -> map to [0, 1]
    sim = _cosine_sim(gt_emb, pred_emb)
    sim01 = (sim + 1.0) * 0.5
    sim01 = max(0.0, min(1.0, sim01))
    return float(sim01)

# ============================================================
# IntlzR: Knowledge Internalization Reward
# ============================================================

_INTLZR_MODEL = None
_INTLZR_TOKENIZER = None
_INTLZR_DEVICE = None


def _load_intlzr_model(model_path: str):
    global _INTLZR_MODEL, _INTLZR_TOKENIZER, _INTLZR_DEVICE

    if _INTLZR_MODEL is not None and _INTLZR_TOKENIZER is not None:
        return

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except Exception:
        _INTLZR_MODEL, _INTLZR_TOKENIZER, _INTLZR_DEVICE = None, None, None
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _INTLZR_DEVICE = device

    _INTLZR_TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    dtype = torch.float16 if device == "cuda" else torch.float32
    _INTLZR_MODEL = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)

    _INTLZR_MODEL.eval()


def intlzr_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs
) -> float:
    """
    Returns a learned reward score for the response string.
    Default output range: [0, 1] after sigmoid.
    """

    model_path = (
        kwargs.get("model_path")
        or os.environ.get("INTLZR_MODEL_PATH", "")
    )
    if not model_path:
        return 0.0

    if not solution_str:
        return 0.0

    _load_intlzr_model(model_path)
    if _INTLZR_MODEL is None or _INTLZR_TOKENIZER is None:
        return 0.0

    try:
        import torch

        max_len = int(kwargs.get("max_length", 1024))

        inputs = _INTLZR_TOKENIZER(
            solution_str,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        inputs = {k: v.to(_INTLZR_DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            out = _INTLZR_MODEL(**inputs)
            logits = out.logits

        score = float(logits.squeeze().detach().cpu().float().item())



        if not (score == score):
            return 0.0

        return float(score)

    except Exception:
        return 0.0


# ============================================================
# MathAccuracy / MathFormat (WATCHING/THINKING/ANSWER)
# ============================================================
def math_accuracy_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    gt_text = _get_gt_text(ground_truth, keys=("solution", "answer", "target"))
    gt_ans = extract_xml_block(gt_text, "ANSWER") or extract_xml_block(gt_text, "answer") or gt_text.strip()
    pred_ans = extract_xml_block(solution_str, "ANSWER") or extract_xml_block(solution_str, "answer") or solution_str.strip()

    try:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify

        gold_parsed = parse(gt_ans, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) == 0:
            return 1.0

        answer_parsed = parse(
            pred_ans,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        return float(verify(answer_parsed, gold_parsed))
    except Exception:
        return 0.0


def math_format_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    return mm_watch_think_answer_format_score(data_source, solution_str, ground_truth, extra_info, **kwargs)


# ============================================================
# MultiModalAccuracy (WATCHING/THINKING/ANSWER + fallback)
# ============================================================
def multimodal_acc_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    gt_text = _get_gt_text(ground_truth, keys=("solution", "answer", "target"))
    gt_ans = extract_xml_block(gt_text, "ANSWER") or extract_xml_block(gt_text, "answer") or gt_text.strip()
    pred_ans = extract_xml_block(solution_str, "ANSWER") or extract_xml_block(solution_str, "answer") or solution_str.strip()

    try:
        from math_verify import parse, verify
        if float(verify(parse(pred_ans), parse(gt_ans))) > 0:
            return 1.0
    except Exception:
        pass

    try:
        return 1.0 if pred_ans.strip() == gt_ans.strip() else 0.0
    except Exception:
        return 0.0





def _clip01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x != x:
        return 0.0
    return max(0.0, min(1.0, x))


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:


    w_vsr = float(kwargs.get("w_vsr", 1))
    w_vpr = float(kwargs.get("w_vpr", 1))
    w_acc = float(kwargs.get("w_acc", 1))
    w_fmt = float(kwargs.get("w_fmt", 1))
    w_intlzr = float(kwargs.get("w_intlzr", 1))

    normalize_weights = bool(kwargs.get("normalize_weights", True))
    skip_expensive_if_bad_format = bool(kwargs.get("skip_expensive_if_bad_format", True))

    fmt = _clip01(math_format_score(data_source, solution_str, ground_truth, extra_info, **kwargs))
    acc = _clip01(math_accuracy_score(data_source, solution_str, ground_truth, extra_info, **kwargs))

    if skip_expensive_if_bad_format and fmt <= 0.0:
        vsr = 0.0
        vpr = 0.0
    else:
        try:
            vsr = _clip01(vsr_score(data_source, solution_str, ground_truth, extra_info, **kwargs))
        except Exception:
            vsr = 0.0

        try:
            vpr = _clip01(para_reward_score(data_source, solution_str, ground_truth, extra_info, **kwargs))
        except Exception:
            vpr = 0.0

    try:
        intlzr = _clip01(intlzr_score(data_source, solution_str, ground_truth, extra_info, **kwargs))
    except Exception:
        intlzr = 0.0

    if normalize_weights:
        s = w_vsr + w_vpr + w_acc + w_fmt + w_intlzr
        if s > 1e-12:
            w_vsr /= s
            w_vpr /= s
            w_acc /= s
            w_fmt /= s
            w_intlzr /= s

    reward = (
        w_vsr * vsr +
        w_vpr * vpr +
        w_acc * acc +
        w_fmt * fmt +
        w_intlzr * intlzr
    )
    return float(_clip01(reward))
