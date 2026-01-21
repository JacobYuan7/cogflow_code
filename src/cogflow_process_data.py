import os
import json
import random
import argparse
import re
from datasets import Dataset


def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def extract_tag(text: str, tag: str) -> str:
    if not text:
        return ""
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def load_mm_dataset(data_dir, train_file, val_file, train_n=None, val_n=None, seed=42):
    rng = random.Random(seed)

    train_path = os.path.join(data_dir, train_file)
    val_path = os.path.join(data_dir, val_file)

    train_data_all = read_jsonl(train_path)
    val_data_all = read_jsonl(val_path)

    if train_n is not None:
        train_k = min(train_n, len(train_data_all))
        train_data_all = rng.sample(train_data_all, train_k)

    if val_n is not None:
        val_k = min(val_n, len(val_data_all))
        val_data_all = rng.sample(val_data_all, val_k)

    return train_data_all, val_data_all


def make_map_fn(split: str, data_source: str, image_root: str = None):
    def process_fn(example, idx):
       
        question = example.get("query", "")


        img = example.get("images", None)

        if img is None:
            images = []
        else:
            if image_root is not None and not os.path.isabs(img):
                img = os.path.join(image_root, img)
            images = [img]  


        if images and "<image>" not in question:
            question = "<image>\n" + question

        prompt = [{"role": "user", "content": question}]


        resp = example.get("response", "")

        watching = extract_tag(resp, "WATCHING")   
        thinking = extract_tag(resp, "THINKING")   
        ans = extract_tag(resp, "ANSWER")   

        ground_truth = {
            "answer": ans,
            "watch": watching,
            "think": thinking,
        }

        return {
            "data_source": data_source,
            "prompt": prompt,
            "images": images,
            "ability": "mathcog",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "split": split,
                "index": int(idx),
                "query": example.get("query", ""),
                "raw_image": example.get("images", None),
                "response": resp,
            }
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=str, default="DATA_PATH")
    parser.add_argument("--train_file", type=str, default="train.jsonl")
    parser.add_argument("--val_file", type=str, default="val.jsonl")
    parser.add_argument("--save_dir", type=str, default="data/mathcog")
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--train_n", type=int, default=None)
    parser.add_argument("--val_n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    train_data, val_data = load_mm_dataset(
        args.local_dir,
        args.train_file,
        args.val_file,
        train_n=args.train_n,
        val_n=args.val_n,
        seed=args.seed
    )

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    train_dataset = train_dataset.map(
        function=make_map_fn("train", data_source="Mathcog", image_root=args.image_root),
        with_indices=True
    )
    val_dataset = val_dataset.map(
        function=make_map_fn("val", data_source="Mathcog", image_root=args.image_root),
        with_indices=True
    )

    train_dataset.to_parquet(os.path.join(args.save_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(args.save_dir, "val.parquet"))

    print(f"Saved to {args.save_dir}")
    print(f"train={len(train_dataset)} val={len(val_dataset)}")
