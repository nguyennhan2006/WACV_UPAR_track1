#!/usr/bin/env python
"""Run a submitted package (X2F-CMASK or C4) on a list of images and write its predictions.

The package's run.py is used unchanged, through the same calls as the Codabench scoring program
(load_model, predict_batch). By default the time guard of run.py is switched off, so every image is
processed with the designed configuration (no resolution drop, tail models kept), whatever the speed
of the GPU (Section 9 of README.md).

  python inference/predict.py --package x2f_cmask --images /data/test/gt.csv --root /data/test \
      --out out/predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
# Submitted packages in the Google Drive folder of README Section 6: file id and md5 of the zip
PACKAGES = {
    "x2f_cmask": ("1YrnN5wIJYCr0r_SD9Fqmz6uxAB5pv39H", "e9bb5cdf6d567767d8d0a47791b4d299"),
    "c4": ("19uzMts7ws6smvgca-BxKuI2zEzebwMF5", "b45d3f00362f15a1b99f060296a0aaef"),
}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EPS = 1e-20


def log(msg: str) -> None:
    print(f"[predict] {msg}", flush=True)


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def find_run(folder: Path) -> Path:
    hits = sorted(folder.rglob("run.py"), key=lambda p: len(p.parts))
    if not hits:
        sys.exit(f"no run.py in {folder}")
    return hits[0].parent


def package_dir(spec: str, cache: Path, check_md5: bool) -> Path:
    """Folder holding run.py: an unpacked package, a package zip, or a released package name."""
    path, md5, file_id = Path(spec), None, None
    if path.is_dir():
        return find_run(path)
    if path.is_file():
        name = path.stem
    elif spec in PACKAGES:
        name, (file_id, md5) = spec, PACKAGES[spec]
        path = cache / f"{spec}.zip"
    else:
        sys.exit(f"--package {spec!r}: no such file or folder, and not one of {sorted(PACKAGES)}")
    out = cache / name
    if (out / ".complete").exists():
        return find_run(out)
    cache.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        import gdown
        log(f"downloading the {name} package from Google Drive (file id {file_id}) to {path}")
        if not gdown.download(id=file_id, output=str(path), quiet=False, resume=True):
            sys.exit(f"download failed: get the zip by hand (README Section 6) and save it as {path}")
    if check_md5:
        log(f"checking the md5 of {path}")
        got, known = md5sum(path), {v[1]: k for k, v in PACKAGES.items()}
        if md5 and got != md5:
            sys.exit(f"md5 of {path} is {got}, expected {md5}: delete the file and download it again")
        log(f"md5 {got}" + (f" = submitted {known[got]} package" if got in known else " (not a submitted package)"))
    log(f"unpacking {path} to {out}")
    with zipfile.ZipFile(path) as z:
        z.extractall(out)
    (out / ".complete").touch()
    return find_run(out)


def read_list(spec: str):
    """(relative paths, ground truth or None, attribute names from the header or None).

    A CSV whose first column is the image path (header optional, e.g. '# image,Age-Young,...'; label
    columns after it are used for scoring), a text file with one path per line, or a folder of images."""
    p = Path(spec)
    if p.is_dir():
        return sorted(q.relative_to(p).as_posix() for q in p.rglob("*") if q.suffix.lower() in IMAGE_EXT), None, None
    with open(p, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if r and r[0].strip()]
    head, names = [c.strip() for c in rows[0]], None
    if head[0].startswith("#") or head[0].lower() in ("image", "image_path", "path", "filename"):
        names, rows = head[1:] or None, rows[1:]
    gt = np.array([[float(x) for x in r[1:]] for r in rows], np.float32) if len(rows[0]) > 1 else None
    return [r[0].strip() for r in rows], gt, names


def domain(rel: str) -> str:
    # Top-level folder of the relative path, as run.py groups images (MEVID / Synth on the test set)
    parts = rel.replace("\\", "/").strip("/").split("/")
    return parts[0] if len(parts) > 1 else "?"


def par_metrics(gt, pr, valid):
    # Same as par_metrics in training/: mA over attributes, instance metrics per image then averaged
    g, p = gt[:, valid].astype(bool), pr[:, valid].astype(bool)
    tp = (g & p).sum(0); fn = (g & ~p).sum(0); tn = (~g & ~p).sum(0); fp = (~g & p).sum(0)
    mA = float(np.mean((tp / (tp + fn + EPS) + tn / (tn + fp + EPS)) / 2))
    inter = (g & p).sum(1)
    ip, ir = inter / (p.sum(1) + EPS), inter / (g.sum(1) + EPS)
    if1 = float(np.mean(2 * ip * ir / (ip + ir + EPS)))
    return dict(score=2 * mA * if1 / (mA + if1 + EPS), mA=mA, instance_f1=if1,
                instance_precision=float(np.mean(ip)), instance_recall=float(np.mean(ir)))


def score(rels, gt, labels):
    doms = np.array([domain(r) for r in rels])
    res = {}
    for d in dict.fromkeys(doms):
        m = doms == d
        g = gt[m] > 0.5
        res[d] = dict(images=int(m.sum()), **par_metrics(g, labels[m], g.sum(0) > 0))
    keys = ["score", "mA", "instance_f1", "instance_precision", "instance_recall"]
    res["mean over domains"] = dict(images=len(rels), **{k: float(np.mean([r[k] for r in res.values()])) for k in keys})
    print(f"{'domain':<20}{'images':>8}" + "".join(f"{k:>20}" for k in keys), flush=True)
    for d, r in res.items():
        print(f"{d:<20}{r['images']:>8}" + "".join(f"{r[k]:>20.4f}" for k in keys), flush=True)
    return res


def write_csv(path: Path, rels, names, values, fmt: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image"] + list(names))
        for rel, row in zip(rels, values):
            w.writerow([rel] + [fmt % x for x in row])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", default="x2f_cmask",
                    help="x2f_cmask or c4 (downloaded once into --cache), a package zip, or an unpacked package folder")
    ap.add_argument("--images", required=True, help="CSV / text list of image paths relative to --root, or a folder of images")
    ap.add_argument("--root", help="folder the listed paths are relative to (default: the folder of the list)")
    ap.add_argument("--out", default="predictions.csv", help="probabilities; labels and scores are written next to it")
    ap.add_argument("--cache", default=os.environ.get("UPAR_PACKAGES", str(REPO / "packages")),
                    help="where released packages are downloaded and unpacked")
    ap.add_argument("--time-guard", choices=["off", "on"], default="off",
                    help="on = behave as on Codabench (on a slow GPU it lowers the resolution or drops the tail models)")
    ap.add_argument("--split-at", type=int,
                    help="X2F-CMASK: send stream positions >= N to the tail models (default: images under a 'Synth*' folder)")
    ap.add_argument("--limit", type=int, help="only the first N listed images (quick check)")
    ap.add_argument("--no-md5", action="store_true", help="skip the md5 check of the package zip")
    args = ap.parse_args()

    rels, gt, hdr_names = read_list(args.images)
    if args.limit:
        rels, gt = rels[:args.limit], None if gt is None else gt[:args.limit]
    n = len(rels)
    if not n:
        sys.exit(f"no images in {args.images}")
    root = Path(args.root) if args.root else Path(args.images) if Path(args.images).is_dir() else Path(args.images).parent
    paths = [r if os.path.isabs(r) else str((root / r).resolve()) for r in rels]
    missing = [r for r, q in zip(rels, paths) if not os.path.isfile(q)]
    if missing:
        sys.exit(f"{len(missing)} of {n} images not found under {root}, e.g. {missing[:3]}")

    run_dir = package_dir(args.package, Path(args.cache), not args.no_md5)
    cfg = json.load(open(run_dir / "assets" / "config.json", encoding="utf-8"))
    names = cfg["attributes"]
    if hdr_names and sorted(hdr_names) == sorted(names):
        names = hdr_names                                   # outputs follow the column order of the list
    elif hdr_names or (gt is not None and gt.shape[1] != len(names)):
        log("the list's columns are not the package's 40 attributes: outputs use the package order, no scoring")
        gt = None
    log(f"package {run_dir} | {n} images under {root} | domains {dict((d, sum(domain(r) == d for r in rels)) for d in dict.fromkeys(map(domain, rels)))}")

    # Head/tail routing of X2F-CMASK. On Codabench the test stream was MEVID #0-28094, then Synth #28095-29247,
    # and run.py split it by position; feeding the Synth images last gives the same routing for any list order.
    order = list(range(n))
    split = (cfg.get("split") or {}).get("at")
    if split is not None:
        if args.split_at is not None:
            split_at = args.split_at
            log(f"routing by position: images #{split_at}.. of the list -> tail models")
        else:
            tail = [i for i in order if domain(rels[i]).lower().startswith("synth")]
            if tail:
                head = [i for i in order if not domain(rels[i]).lower().startswith("synth")]
                order, split_at = head + tail, len(head)
                log(f"routing by domain: {len(head)} images -> head pair, {len(tail)} 'Synth' images -> tail models")
            else:
                split_at = int(split)
                log(f"no 'Synth' folder in the list: routing by position as on Codabench, images #{split_at}.. -> tail models")
        os.environ["UPAR_SPLIT_AT"] = str(split_at)
    os.environ["UPAR_N_EXPECTED"] = str(n)

    sys.path.insert(0, str(run_dir))
    import run
    t0 = time.time()
    run.load_model()
    log(f"model loaded in {time.time() - t0:.0f}s")
    guard = run._S.get("guard")
    if args.time_guard == "off" and guard is not None:
        # Unlimited budget: no tier drop, tail models never dropped; the tail still runs at its fixed tier (tail_tier).
        # UPAR_GUARD_OFF=1 is not used because it also removes that tier.
        guard.update(budget_s=float("inf"), hard_s=float("inf"), shadow_on=False)
        log("time guard off")

    bs = int(getattr(run, "BATCH_SIZE", 64))
    probs = np.zeros((n, len(names)), np.float32)
    t1 = time.time()
    for pos in range(0, n, bs):
        idx = order[pos:pos + bs]
        batch = [{"image_path": paths[i], "image": rels[i], "index": pos + j, "attribute_names": names} for j, i in enumerate(idx)]
        probs[idx] = np.asarray(run.predict_batch(batch), np.float32)
        done = pos + len(idx)
        if done % (50 * bs) == 0 or done == n:
            log(f"{done}/{n} images | {time.time() - t1:.0f}s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = (probs > 0.5).astype(np.int8)                  # calibrated thresholds are folded into the scores
    write_csv(out, rels, names, probs, "%.6f")
    write_csv(out.with_name(out.stem + "_labels.csv"), rels, names, labels, "%d")
    log(f"wrote {out} and {out.with_name(out.stem + '_labels.csv')}")
    if gt is not None:
        res = score(rels, gt, labels)
        sp = out.with_name(out.stem + "_scores.json")
        sp.write_text(json.dumps(res, indent=1), encoding="utf-8")
        log(f"wrote {sp}")


if __name__ == "__main__":
    main()
