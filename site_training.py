"""File-based federation for institutions with aligned, authorized research records.

The coordinator initializes one checkpoint and averages client updates. Each site
reads its own JSONL manifest locally and sends a trainable-weight update only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import AutoImageProcessor, AutoTokenizer, BertModel, ViTModel

from model import TemporalGraph

VISION = "google/vit-base-patch16-224-in21k"
LANGUAGE = "bert-base-uncased"


class SiteModel(torch.nn.Module):
    def __init__(self, vision, language):
        super().__init__()
        self.vision = ViTModel.from_pretrained(vision)
        self.language = BertModel.from_pretrained(language)
        for encoder in (self.vision, self.language):
            encoder.requires_grad_(False)
            encoder.eval()
        self.image_projection = torch.nn.Linear(self.vision.config.hidden_size, 32)
        self.text_projection = torch.nn.Linear(self.language.config.hidden_size, 32)
        self.graph = TemporalGraph(32)
        self.attention = torch.nn.Sequential(torch.nn.Linear(32, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
        self.head = torch.nn.Linear(32, 1)

    def predict_record(self, record, folder, processor, tokenizer):
        features = []
        if record.get("image"):
            with Image.open(folder / record["image"]) as image:
                pixels = processor(images=image.convert("RGB"), return_tensors="pt")
            features.append(self.image_projection(self.vision(**pixels).last_hidden_state[:, 0]))
        if record.get("transcript", "").strip():
            tokens = tokenizer(record["transcript"], return_tensors="pt", truncation=True, max_length=256)
            features.append(self.text_projection(self.language(**tokens).last_hidden_state[:, 0]))
        if record.get("visits"):
            visits = torch.tensor(record["visits"], dtype=torch.float32)
            if visits.ndim != 2 or visits.shape[1] != 3 or not visits.isfinite().all():
                raise ValueError("Expected finite visits as [time, standardized feature A, standardized feature B].")
            if (visits[1:, 0] < visits[:-1, 0]).any():
                raise ValueError("Visits must be chronological.")
            features.append(self.graph(visits.unsqueeze(0)))
        if not features:
            return None
        branches = torch.stack(features, dim=1)
        weights = self.attention(branches).softmax(dim=1)
        return self.head((branches * weights).sum(dim=1)).flatten()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_trainable(model, path):
    save_file({n: p.detach().contiguous() for n, p in model.named_parameters() if p.requires_grad}, str(path))


def run(args):
    torch.set_num_threads(1)
    if args.command == "aggregate":
        meta = [json.loads(p.with_suffix(".json").read_text()) for p in args.updates]
        if any(m["parent"] != digest(args.base) or m["count"] <= 0 for m in meta):
            raise ValueError("Expected nonempty updates from the same round.")
        if len({m['site'] for m in meta}) != len(meta):
            raise ValueError("A site appears twice.")
        base = load_file(str(args.base))
        averaged = {k: torch.zeros_like(v) for k, v in base.items()}
        for path, m in zip(args.updates, meta):
            state = load_file(str(path))
            if state.keys() != base.keys() or m["sha256"] != digest(path):
                raise ValueError("Update contents mismatch.")
            for k, v in state.items():
                if v.shape != base[k].shape or not v.isfinite().all():
                    raise ValueError("Invalid update tensor.")
                averaged[k].add_(v, alpha=m["count"]/sum(i["count"] for i in meta))
        save_file(averaged, str(args.out))
        return
    torch.manual_seed(23)
    model = SiteModel(args.vision, args.language)
    if args.command == "init":
        save_trainable(model, args.out)
        return
    state = load_file(str(args.base))
    if set(state) != {n for n,p in model.named_parameters() if p.requires_grad}:
        raise ValueError("Checkpoint does not match model.")
    model.load_state_dict(state, strict=False)
    processor = AutoImageProcessor.from_pretrained(args.vision)
    tokenizer = AutoTokenizer.from_pretrained(args.language)
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Provide one aligned record per participant.")
    opt = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.01)
    scored = 0
    errors = []
    correct = 0
    for row in rows:
        with torch.set_grad_enabled(args.command == "client"):
            logit = model.predict_record(row, args.manifest.parent, processor, tokenizer)
            if logit is None:
                continue
            if row["label"] not in (0,1):
                raise ValueError("Binary reference label required.")
            y = torch.tensor([float(row["label"])])
            if args.command == "client":
                opt.zero_grad()
                torch.nn.functional.binary_cross_entropy_with_logits(logit, y).backward()
                opt.step()
            prob = logit.detach().sigmoid().item()
            correct += int((prob >= .5) == bool(row["label"]))
            errors.append((prob-row["label"])**2)
            scored += 1
    if not scored:
        raise ValueError("No usable records.")
    if args.command == "client":
        save_trainable(model, args.out)
        args.out.with_suffix(".json").write_text(json.dumps(dict(site=args.site, count=scored,
            parent=digest(args.base), sha256=digest(args.out))))
    else:
        print(json.dumps(dict(total=len(rows), scored=scored, accuracy=correct/scored, brier=sum(errors)/scored)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision", default=VISION)
    parser.add_argument("--language", default=LANGUAGE)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "client", "aggregate", "evaluate"):
        cmd = sub.add_parser(name)
        if name != "init":
            cmd.add_argument("--base", required=True, type=Path)
        if name != "evaluate":
            cmd.add_argument("--out", required=True, type=Path)
        if name in ("client","evaluate"):
            cmd.add_argument("--manifest", required=True, type=Path)
        if name == "client":
            cmd.add_argument("--site", required=True)
        if name == "aggregate":
            cmd.add_argument("--updates", required=True, nargs="+", type=Path)
    run(parser.parse_args())
