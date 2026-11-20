"""Run a reproducible federated simulation and export held-out case predictions."""
import argparse
import copy
import itertools
import json
import secrets
from pathlib import Path

import torch
from torch import nn
from model import NeuroFuse

VOCAB = ["the", "window", "opens", "today", "a", "garden", "looks", "bright"]
ROOT = Path(__file__).resolve().parent


def cohort(n, site, seed):
    rng = torch.Generator().manual_seed(seed)
    signal = torch.randn(n, 3, generator=rng)
    image = torch.randn(n, 1, 16, 16, generator=rng) * .45
    image[:, :, 4:12, 4:12] += signal[:, 0, None, None, None]
    tokens = torch.randint(0, 4, (n, 6), generator=rng)
    tokens += (signal[:, 1, None] + torch.randn(n, 6, generator=rng) > 0).long() * 4
    times = torch.rand(n, 4, 1, generator=rng).cumsum(1)
    values = torch.randn(n, 4, 2, generator=rng) * .45 + signal[:, 2, None, None]
    visits = torch.cat((times, values), 2)
    mask = torch.rand(n, 3, generator=rng) > .2
    mask[~mask.any(1), site % 3] = True
    label = (signal.sum(1) + .4 * torch.randn(n, generator=rng) > 0).float()
    return image, tokens, visits, mask, label


def local_update(global_model, batch, args, noise_rng):
    model = copy.deepcopy(global_model)
    opt = torch.optim.SGD(model.parameters(), lr=.12)
    features, labels = batch[:-1], batch[-1]
    for _ in range(args.local_steps):
        opt.zero_grad()
        if args.noise:
            # One synthetic participant per row. Full batch, one clipped vector per row.
            sums = [torch.zeros_like(p) for p in model.parameters()]
            for i, y in enumerate(labels):
                model.zero_grad()
                logit, _ = model(*(v[i:i+1] for v in features))
                nn.functional.binary_cross_entropy_with_logits(logit, y.reshape(1)).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                for total, param in zip(sums, model.parameters()):
                    total.add_(param.grad)
            for total, param in zip(sums, model.parameters()):
                noise = torch.randn(total.shape, generator=noise_rng) * args.noise
                param.grad = (total + noise) / len(labels)
        else:
            logits, _ = model(*features)
            nn.functional.binary_cross_entropy_with_logits(logits, labels).backward()
        opt.step()
    return model.state_dict(), len(labels)


@torch.no_grad()
def evaluate(model, test):
    image, tokens, visits, observed, y = test
    cases = [{"id": f"S4-{i+1:03}", "tokens": " ".join(VOCAB[t] for t in tokens[i]),
              "visits": visits[i].round(decimals=2).tolist(), "label": int(y[i]),
              "image": image[i, 0].round(decimals=2).tolist(), "subsets": {}} for i in range(8)]
    metrics = {}
    for pattern in itertools.product([False, True], repeat=3):
        if not any(pattern):
            continue
        key = "".join(str(int(v)) for v in pattern)
        mask = torch.tensor(pattern).expand_as(observed)
        logits, weights = model(image, tokens, visits, mask)
        score = logits.sigmoid()
        metrics[key] = {"accuracy": (logits.ge(0) == y.bool()).float().mean().item(),
                        "brier": (score-y).square().mean().item()}
        for i, case in enumerate(cases):
            case["subsets"][key] = {"score": score[i].item(), "weights": weights[i].tolist()}
    logits, _ = model(image, tokens, visits, observed)
    metrics["observed"] = {"accuracy": (logits.ge(0) == y.bool()).float().mean().item(),
                           "brier": (logits.sigmoid()-y).square().mean().item()}
    return cases, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--local-steps", type=int, default=3)
    p.add_argument("--noise", type=float, default=0., help="Experimental per-record Gaussian noise multiplier")
    p.add_argument("--output", type=Path, default=ROOT / "docs" / "experiment.json")
    args = p.parse_args()
    if min(args.rounds, args.local_steps) < 1 or not 0 <= args.noise < float("inf"):
        p.error("Specify positive training steps and a finite nonnegative noise multiplier.")
    torch.set_num_threads(1)
    torch.manual_seed(23)
    model = NeuroFuse()
    clients = [cohort(n, k, 70+k) for k, n in enumerate([48, 60, 72])]
    noise_rng = torch.Generator().manual_seed(secrets.randbits(63))
    for _ in range(args.rounds):
        updates = [local_update(model, client, args, noise_rng) for client in clients]
        count = sum(n for _, n in updates)
        model.load_state_dict({key: sum(s[key] * (n/count) for s, n in updates)
                               for key in model.state_dict()})
    model.eval()
    cases, metrics = evaluate(model, cohort(96, 3, 95))
    result = dict(cases=cases, metrics=metrics, train_sites=3, train_records=180,
                  test_records=96, rounds=args.rounds, local_steps=args.local_steps,
                  noise=args.noise, seed=23)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, allow_nan=False))
    torch.save(model.state_dict(), args.output.with_suffix(".pt"))
    print(json.dumps(metrics["observed"]))


if __name__ == "__main__":
    main()
