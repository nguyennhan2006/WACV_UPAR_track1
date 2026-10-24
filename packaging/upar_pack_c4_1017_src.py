# %% [markdown]
# # UPAR 2027 · đóng gói C4 = F1 (H+ FREEZE 8 + LP-FT) .6 + soup(L2-LPFT, L2-LPFT-s2) .4 @224 mọi ảnh · soup tính trên Kaggle · T4
# **Input:** <kaggle-user>/upar-valc-pkg-1016, <kaggle-user>/upar-pack-tools-1017, <kaggle-user>/upar-f1-lpft-1017, <kaggle-user>/upar-l2var-1017 (s2), <kaggle-user>/upar-l2b-1017 (lpft), <kaggle-user>/wacv-data, <kaggle-user>/upar-resplit-0710

# %%
print("[cell] pack C4", flush=True)
import os, sys, json, subprocess, torch
from pathlib import Path
IN, WORK = Path("/kaggle/input"), Path("/kaggle/working")
pk = [d for d in sorted(IN.rglob("pkg")) if (d / "assets/config.json").exists()]
cfgs = {d: len(json.load(open(d / "assets/config.json"))["models"]) for d in pk}
A = next(d for d, n in cfgs.items() if n == 4); B = next(d for d, n in cfgs.items() if n == 3)
TL = next(IN.rglob("pack_hplus.py")).parent; M3 = next(p for p in IN.rglob("best_ema.pt") if p.parent.name == "m3")
CK = {p.parent.name: p for p in IN.rglob("best_ema.pt")}; print({k: str(v) for k, v in CK.items()}, flush=True)
F1 = next(p for k, p in CK.items() if "f1" in k.lower())
a, b = torch.load(CK["lpft"], map_location="cpu", weights_only=True), torch.load(CK["s2"], map_location="cpu", weights_only=True)
assert set(a) == set(b); torch.save({k: (0.5 * a[k].float() + 0.5 * b[k].float()) if a[k].is_floating_point() else a[k] for k in a}, "/tmp/L2SOUP.pt"); del a, b
print("soup L2 -> /tmp/L2SOUP.pt | F1", F1, flush=True)
ANN = next(d for d in IN.rglob("task1") if (d / "val/gt.csv").exists() and "resplit" in str(d))
DATA = next(d for d in IN.rglob("*") if d.is_dir() and {"PA100k", "PETA", "Market1501"} <= {x.name for x in d.iterdir() if x.is_dir()})
p = subprocess.Popen([sys.executable, str(TL / "pack_hplus.py"), "--a", str(A), "--b", str(B), "--m3", str(M3), "--tools", str(TL), "--ann", str(ANN), "--data", str(DATA),
                      "--out", str(WORK / "out"), "--spec", f"ext:{F1}", "--mode", "pair", "--runpy", "run_v17x.py", "--budget", "266", "--chunk", "0",
                      "--l2", "ext:/tmp/L2SOUP.pt", "--name", "SUBMIT_c4_clean_f1lpft_L2lpftsoup_224_gpu_w8.zip"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
for line in p.stdout: print(line, end="", flush=True)
print("rc", p.wait(), flush=True)
