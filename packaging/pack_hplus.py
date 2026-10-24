"""17/10: ĐÓNG GÓI bản nộp sạch với H+ là biến thể bất kỳ (f8 / soup / WiSE / ext) — chạy trên Kaggle GPU (nặng RAM, không chạy trên máy).
1) dựng weights H+ theo đặc tả (như eval_hplus.py) -> fp16 + int8 theo kênh (định dạng w8 của run.py);
2) cấu hình: cặp (H .6 + L2 .4) hoặc ba model (H .48 + L2 .32 + RADIO .20), tất cả 224x112, không model phụ, không đuôi (tail_n 0);
3) logit val qua ĐÚNG run.py của gói (weights đã lượng tử) -> ngưỡng calibrate in-sample trên toàn val (calibrate của upar_v2_src.py);
4) tự kiểm: predict_batch trên 256 ảnh val == sigmoid(Σ w·z − logit(ngưỡng)); in ms/ảnh (T4) để đối chiếu;
5) ghi zip (run.py, model.py, metadata.yaml, assets/config.json, assets/weights_*.pt) + md5 + provenance.
python pack_hplus.py --a A/pkg --b B/pkg --m3 m3/best_ema.pt --tools T --ann task1 --data D --out O --spec soup:0.5 --mode pair|triple --runpy run_v17s.py
    [--budget 266] [--chunk 0] [--name SUBMIT_...zip]"""
import os, sys, io, json, time, shutil, zipfile, hashlib, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, torch
ap = argparse.ArgumentParser()
for k in ("a", "b", "tools", "ann", "data", "out", "spec", "mode", "runpy"): ap.add_argument("--" + k, required=True)
ap.add_argument("--m3", default=""); ap.add_argument("--budget", type=float, default=266.0); ap.add_argument("--chunk", type=int, default=0); ap.add_argument("--name", default="")
ap.add_argument("--l2", default="base")
ap.add_argument("--tier", type=int, default=0)   # 17/10: 1 = mọi ảnh chạy 192x96 (tail_n = toàn bộ test, tail_tier 1); ngưỡng tính trên logit val ở đúng tier   # 17/10: base = L2 của gói A; ext:<best_ema.pt> hoặc ext:@<thư mục> (tìm <thư mục>/best_ema.pt trong /kaggle/input)
a = ap.parse_args(); T0 = time.time(); log = lambda s: print(f"[pack] {time.time() - T0:6.0f}s | {s}", flush=True)
A, B, TL, OUT = Path(a.a), Path(a.b), Path(a.tools), Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
assert a.mode in ("pair", "triple")
def deq(sd):
    if any(k.endswith("::w8scale") for k in sd):
        sd = {k: v.float() * sd[k + "::w8scale"].float() if k + "::w8scale" in sd else v for k, v in sd.items() if not k.endswith("::w8scale")}
    return {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
def resolve(arg):                                                     # "@f1" -> /kaggle/input/**/f1/best_ema.pt
    if arg.startswith("@"):
        nm = arg[1:].lower(); allp = sorted(Path(os.environ.get("PACK_FIND", "/kaggle/input")).rglob("best_ema.pt"))
        hit = [q for q in allp if q.parent.name.lower() == nm] or [q for q in allp if any(nm in part.lower() for part in q.parts[-4:-1])]   # Kaggle có thể làm phẳng thư mục con
        assert len(hit) >= 1, f"không thấy {arg} trong {[str(q) for q in allp]}"; log(f"{arg} -> {hit[0]}"); return str(hit[0])
    return arg
def w8(sd):
    q = {}
    for k, v in sd.items():
        v = v.half() if v.dtype == torch.float32 else v
        if v.is_floating_point() and v.ndim >= 2 and v.numel() >= 4096:
            f = v.float(); sc = f.abs().amax(dim=tuple(range(1, f.ndim)), keepdim=True).clamp_min(1e-8) / 127.0
            q[k] = torch.round(f / sc).clamp(-127, 127).to(torch.int8); q[k + "::w8scale"] = sc.half()
        else: q[k] = v
    return q
# 1) weights H+
kind, _, arg = a.spec.partition(":"); f8 = deq(torch.load(A / "assets/weights_0.pt", map_location="cpu", weights_only=True))
if kind == "f8": H = f8
elif kind == "g11": H = deq(torch.load(B / "assets/weights_0.pt", map_location="cpu", weights_only=True))
elif kind == "m3": H = deq(torch.load(a.m3, map_location="cpu", weights_only=True))
elif kind == "ext": H = deq(torch.load(resolve(arg), map_location="cpu", weights_only=True))
elif kind in ("soup", "fm3"):
    t = float(arg); o = deq(torch.load(B / "assets/weights_0.pt", map_location="cpu", weights_only=True) if kind == "soup" else torch.load(a.m3, map_location="cpu", weights_only=True))
    H = {k: t * f8[k] + (1 - t) * o[k] if f8[k].is_floating_point() else f8[k] for k in f8}; del o
elif kind == "wise":
    sys.path.insert(0, str(A)); from model import PARNet
    pre = PARNet("dinov3_vith16p", pretrained=True, img_h=224, img_w=112).state_dict(); t = float(arg)
    H = {k: t * f8[k] + (1 - t) * pre[k].float() if (k.startswith("backbone.") and k in pre and f8[k].is_floating_point()) else f8[k] for k in f8}; del pre
else: raise ValueError(a.spec)
qH = w8(H); del H, f8; log(f"H+ '{a.spec}' -> w8 ({len(qH)} khoá)")
# 2) gói tạm
P = OUT / "pkg"; shutil.rmtree(P, ignore_errors=True); (P / "assets").mkdir(parents=True)
shutil.copy(TL / a.runpy, P / "run.py"); shutil.copy(A / "model.py", P / "model.py"); shutil.copy(TL / "metadata.yaml", P / "metadata.yaml")
torch.save(qH, P / "assets/weights_0.pt"); del qH
if a.l2 == "base": shutil.copy(A / "assets/weights_1.pt", P / "assets/weights_1.pt")
else:                                                                 # L2 thay thế: train 252x126 -> chạy 224x112, nội suy pos_embed như make_gpu_ensemble_zip
    import torch.nn.functional as F_
    sdL = deq(torch.load(resolve(a.l2.split(":", 1)[1]), map_location="cpu", weights_only=True)); pe = sdL["backbone.pos_embed"]
    ref = deq(torch.load(A / "assets/weights_1.pt", map_location="cpu", weights_only=True))["backbone.pos_embed"]
    if pe.shape != ref.shape:
        C = pe.shape[-1]; n0 = pe.shape[1] - 1; gh, gw = int(round((n0 * 2) ** 0.5)), int(round((n0 / 2) ** 0.5)); n1 = ref.shape[1] - 1
        th, tw = int(round((n1 * 2) ** 0.5)), int(round((n1 / 2) ** 0.5)); assert gh * gw == n0 and th * tw == n1
        g = F_.interpolate(pe[0, 1:].reshape(1, gh, gw, C).permute(0, 3, 1, 2), size=(th, tw), mode="bicubic", align_corners=False, antialias=True)
        sdL["backbone.pos_embed"] = torch.cat([pe[:, :1], g.permute(0, 2, 3, 1).reshape(1, -1, C)], 1); log(f"L2 pos_embed {gh}x{gw} -> {th}x{tw}")
    torch.save(w8(sdL), P / "assets/weights_1.pt"); del sdL
c = json.load(open(A / "assets/config.json")); ms = c["models"]
W = [0.6, 0.4] if a.mode == "pair" else [0.48, 0.32, 0.20]
models = [dict(ms[0], w=W[0], file="weights_0.pt", source=f"H+ {a.spec}"), dict(ms[1], w=W[1], file="weights_1.pt", source=f"L2 {a.l2}")]
if a.mode == "triple": shutil.copy(B / "assets/weights_2.pt", P / "assets/weights_2.pt"); models.append(dict(ms[3], w=W[2], file="weights_2.pt"))
for m in models: m.pop("optional", None); m.pop("head_only", None)
c.update(models=models, chunk=a.chunk, half_linear=True, time_guard=dict(budget_s=a.budget, n_expected=29248, tail_n=29248 if a.tier else 0, tail_tier=a.tier))
for k in ("thresholds_solo", "route"): c.pop(k, None)
json.dump(c, open(P / "assets/config.json", "w"), indent=1)
# 3) logit val qua run.py của gói -> ngưỡng
if not a.tier: os.environ["UPAR_GUARD_OFF"] = "1"          # tier > 0 cần thang độ phân giải của chốt chặn
sys.path.insert(0, str(P)); import run
run.load_model(); dev = run._S["dev"]
if a.tier: run._set_tier(a.tier); log(f"tier {a.tier}: kích thước {[run._size(sp, a.tier) for _, sp in run._S['models']]}")
rows = [l.split(",", 1)[0] for l in open(Path(a.ann) / "val/gt.csv", encoding="utf-8").read().splitlines()[1:] if l.strip()]
paths = [str(Path(a.data) / r) for r in rows]; tp = ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1))
Z = []
for m, sp in run._S["models"]:
    key = (*run._size(sp, a.tier), sp["_crop"]); zz = []
    for b0 in range(0, len(paths), 64):
        arr = np.stack([d[key] if d is not None else np.zeros((key[0], key[1], 3), np.uint8) for d in tp.map(lambda p: run._load_resize(p, [key]), paths[b0:b0 + 64])])
        x = (torch.from_numpy(arr).to(dev).permute(0, 3, 1, 2).float().div_(255) - sp["_mean"]) / sp["_std"]
        with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"): zz.append(m(x).float().cpu().numpy())
    Z.append(np.concatenate(zz)); np.save(OUT / f"val_logit_m{len(Z) - 1}.npy", Z[-1].astype(np.float16)); log(f"logit val model {len(Z) - 1} ({sp['arch']}) xong")
src = (TL / "upar_v2_src.py").read_text(encoding="utf-8")
va = pd.read_csv(Path(a.ann) / "val/gt.csv"); va = va.rename(columns={va.columns[0]: "image"}); va["ds"] = va.image.str.split("/").str[0]
ns = {"np": np, "pd": pd, "ATTR": [c_ for c_ in va.columns if c_ not in ("image", "ds")]}
exec(src[src.index("EPS = 1e-20"):src.index("# %% [markdown]\n# ## 6.")], ns)
z = sum(w * zz for w, zz in zip(W, Z)); thr, s_in = ns["calibrate"](va, ns["sigm"](z)); thr = [float(t) for t in thr]
c.update(thresholds=thr, thresholds_calibrated=thr, calibrated_on="val resplit 0710 (in-sample, logit qua run.py của gói)",
         version=f"clean {a.mode} H+={a.spec} + L2={a.l2}" + ("" if a.mode == "pair" else " + RADIO") + f" @{['224', '192', '176'][a.tier]} (mọi ảnh) | {a.runpy} | budget {a.budget}")
json.dump(c, open(P / "assets/config.json", "w"), indent=1); log(f"ngưỡng in-sample: Challenge {s_in:.4f}")
# 4) tự kiểm bằng gói cuối (nạp lại run.py)
os.environ.pop("UPAR_GUARD_OFF", None)
run._S.clear(); del run; import gc; gc.collect(); torch.cuda.empty_cache()          # 17/10: giải phóng model lượt tính logit trước khi nạp lại gói (T4 16 GB, 3 model)
for k_ in [k_ for k_ in sys.modules if k_ in ("run", "model")]: del sys.modules[k_]
import importlib; run = importlib.import_module("run"); run.load_model()
idx = np.random.RandomState(1).choice(len(rows), 256, replace=False)
t1 = time.time(); p = np.array(run.predict_batch([dict(attribute_names=c["attributes"], image=rows[i], image_path=paths[i], index=int(i)) for i in idx[:128]])
                              + run.predict_batch([dict(attribute_names=c["attributes"], image=rows[i], image_path=paths[i], index=int(i)) for i in idx[128:]]))
tt = time.time() - t1; thr_ = np.array(thr); q = 1 / (1 + np.exp(-(z[idx] - np.log(thr_ / (1 - thr_)))))
dp = np.abs(p - q); log(f"tự kiểm 256 ảnh: |Δp| TB {dp.mean():.5f} max {dp.max():.4f}, trùng dấu {((p > .5) == (q > .5)).mean():.4f} | {1000 * tt / 256:.1f} ms/ảnh (T4, gồm khởi động)")
assert dp.max() < 0.05 and ((p > .5) == (q > .5)).mean() > 0.995, "gói lệch logit val"
# 5) zip
tag_ = lambda x: x.replace("ext:@", "").replace("ext:", "").replace(":", "").replace("/", "_")[-20:]
name = a.name or f"SUBMIT_clean_{a.mode}_{tag_(a.spec)}_{tag_(a.l2)}_{['224', '192', '176'][a.tier]}_gpu_w8.zip"; zp = OUT / name
with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as zo:
    for f in ("run.py", "model.py", "metadata.yaml", "assets/config.json") + tuple(f"assets/weights_{i}.pt" for i in range(len(models))): zo.write(P / f, f)
h = hashlib.md5(open(zp, "rb").read()).hexdigest()[:8]
json.dump(dict(zip=name, md5=h, spec=a.spec, mode=a.mode, runpy=a.runpy, in_sample=s_in, selfcheck_max_dp=float(dp.max())), open(OUT / (name + ".json"), "w"), indent=1)
shutil.rmtree(P, ignore_errors=True); log(f"ĐÃ GHI {name} ({zp.stat().st_size / 1e9:.2f} GB, md5 {h})")
