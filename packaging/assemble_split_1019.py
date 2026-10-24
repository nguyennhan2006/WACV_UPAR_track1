"""19/10: ghép bản nộp kiểu split_x2f (chia cứng MEVID/Synth, đuôi 4 model 192x96 + lật ngang) với cặp mới từ gói pack_hplus pair (chạy trên Kaggle).
ĐẦU (MEVID, vị trí < 28095): weights_0/1 của gói cặp, thresholds_solo = ngưỡng gói cặp (cal 0.6 m0 + 0.4 m1, logit run.py).
ĐUÔI (Synth): + g11 (weights_2) + RADIO (weights_3) chép nguyên byte từ gói Kaggle có CRC trùng split_x2f, tier 1 + tail_flip,
ngưỡng = cal(0.24 m0 + 0.16 m1 + 0.3 G + 0.3 R) trên val (m0/m1 = logit run.py của gói cặp, G/R = logit 224x112 lúc train, như x2 / split_c4).
Kiểm: (a) dựng lại ngưỡng đuôi x2f từ logit cũ; (b) ngưỡng đầu = ngưỡng gói cặp; (c) spec model 0/1 trùng x2f; tự kiểm đầu (chặt) + đuôi (chỉ in).
python assemble_split_1019.py --pack-out OUT --st SPLIT_TOOLS --tl PACK_TOOLS --ann ANN --data DATA --pkgs A,B --out OUT2 --name NAME"""
import argparse, json, zipfile, zlib, hashlib, shutil, os, sys, time, numpy as np, pandas as pd
from pathlib import Path
ap = argparse.ArgumentParser()
for k in ("--pack-out", "--st", "--tl", "--ann", "--data", "--pkgs", "--out", "--name"): ap.add_argument(k, required=True)
ap.add_argument("--precheck", action="store_true")   # chỉ kiểm (a), spec A, weights_2/3 theo CRC — chạy TRƯỚC pack_hplus
a = ap.parse_args(); OUT, ST, TL, ANN, DATA, O2 = map(Path, (a.pack_out, a.st, a.tl, a.ann, a.data, a.out)); O2.mkdir(parents=True, exist_ok=True)
log = lambda s: print(f"[split] {s}", flush=True)
cx = json.load(open(ST / "x2f_config.json"))
src = (TL / "upar_v2_src.py").read_text(encoding="utf-8")
va = pd.read_csv(ANN / "val/gt.csv"); va = va.rename(columns={va.columns[0]: "image"}); va["ds"] = va.image.str.split("/").str[0]
ns = {"np": np, "pd": pd, "ATTR": [c for c in va.columns if c not in ("image", "ds")]}
exec(src[src.index("EPS = 1e-20"):src.index("# %% [markdown]\n# ## 6.")], ns); cal, sigm = ns["calibrate"], ns["sigm"]
ld = lambda p: np.load(p).astype(np.float64)
F, L, G, R = (ld(ST / f"val_{k}_224.npy") for k in "FLGR"); assert all(x.shape == (len(va), 40) for x in (F, L, G, R))
dmax = lambda u, v: float(np.abs(np.array(u, np.float64) - np.array(v, np.float64)).max())
nd = lambda u, v: int((np.abs(np.array(u, np.float64) - np.array(v, np.float64)) > 1e-6).sum())
t_old, s_old = cal(va, sigm(0.24 * F + 0.16 * L + 0.3 * G + 0.3 * R))
chk_a = dmax(t_old, cx["thresholds"]) < 1e-6; log(f"(a) dựng lại ngưỡng đuôi x2f: {'TRÙNG' if chk_a else 'KHÁC'} ({nd(t_old, cx['thresholds'])}/40, max |d| {dmax(t_old, cx['thresholds']):.2e}) | in-sample {s_old:.4f}")
strip = lambda m: {k: v for k, v in m.items() if k not in ("source", "w", "file", "optional", "head_only")}
want = {2: (842553189, 0xB1B1953E), 3: (653249831, 0x23102E85)}
def crc(p):
    c = 0
    with open(p, "rb") as f:
        while (b := f.read(1 << 24)): c = zlib.crc32(b, c)
    return c
cand = sorted({p for d in a.pkgs.split(",") for p in Path(d).rglob("weights_*.pt")}); W23 = {}
for i, (sz, cr) in want.items():
    W23[i] = next((p for p in cand if p.stat().st_size == sz and crc(p) == cr), None); log(f"weights_{i}: {W23[i]}")
    assert W23[i] is not None, f"không thấy weights_{i} (CRC {cr:#x}) trong {a.pkgs}"
if a.precheck:
    assert chk_a, "không dựng lại được ngưỡng đuôi x2f"
    ms = json.load(open(Path(a.pkgs.split(",")[0]) / "assets/config.json"))["models"]
    for i in (0, 1): assert strip(ms[i]) == strip(cx["models"][i]), (i, strip(ms[i]), strip(cx["models"][i]))
    log("PRECHECK ĐẠT: (a) TRÙNG, spec model 0/1 gói A trùng x2f, weights_2/3 có sẵn"); sys.exit(0)
meta = json.load(open(next(OUT.glob("SUBMIT_*.zip.json")))); PZ = OUT / meta["zip"]; log(f"gói cặp {PZ.name} md5 {meta['md5']} | tự kiểm max dp {meta['selfcheck_max_dp']:.4f}")
zp = zipfile.ZipFile(PZ); pc = json.loads(zp.read("assets/config.json"))
m0, m1 = ld(OUT / "val_logit_m0.npy"), ld(OUT / "val_logit_m1.npy"); assert m0.shape == m1.shape == (len(va), 40)
th, s_h = cal(va, sigm(0.6 * m0 + 0.4 * m1))
log(f"(b) ngưỡng đầu dựng lại từ logit fp16 so với gói cặp: {nd(th, pc['thresholds'])}/40 khác, max |d| {dmax(th, pc['thresholds']):.2e} | in-sample {s_h:.4f} (dùng ngưỡng gói cặp)")
tt, s_t = cal(va, sigm(0.24 * m0 + 0.16 * m1 + 0.3 * G + 0.3 * R)); tt = [float(x) for x in tt]
log(f"ngưỡng đuôi mới: in-sample {s_t:.4f} (x2f {s_old:.4f}) | {nd(tt, cx['thresholds'])}/40 khác x2f, max |d| {dmax(tt, cx['thresholds']):.3f}")
for i in (0, 1): assert strip(pc["models"][i]) == strip(cx["models"][i]), (i, strip(pc["models"][i]), strip(cx["models"][i]))
log("(c) spec model 0/1 của gói cặp trùng x2f")
mp_x, mp_p = (ST / "x2f_model.py.txt").read_bytes(), zp.read("model.py"); log(f"model.py gói cặp == x2f: {mp_x == mp_p} (dùng bản x2f)")
models = [dict(m) for m in cx["models"]]
models[0]["source"] = f"{pc['models'][0].get('source', '')} | w8 qua pack_hplus ({meta['zip']})"; models[1]["source"] = f"{pc['models'][1].get('source', '')} | w8, pos_embed 224x112"
cfg = dict(cx, models=models, thresholds=tt, thresholds_calibrated=tt, thresholds_solo=[float(x) for x in pc["thresholds"]],
           calibrated_on="val resplit 0710 in-sample; đầu = logit run.py gói cặp, đuôi = 0.24 m0 + 0.16 m1 (run.py) + 0.3 G + 0.3 R (logit 224x112 lúc train)",
           version=f"{a.name}: head(<28095) F8c + L2c thr=gói cặp {meta['md5']} | tail(>=28095) F8c 0.24 + L2c 0.16 + G 0.3 + R 0.3 tier 1 + flip | run_split")
ZO = O2 / a.name
with zipfile.ZipFile(ZO, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zo:
    zo.writestr("run.py", (ST / "run_split.py.txt").read_bytes()); zo.writestr("model.py", mp_x); zo.writestr("metadata.yaml", (ST / "x2f_metadata.yaml").read_bytes())
    zo.writestr("assets/config.json", json.dumps(cfg, indent=1, ensure_ascii=False))
    for i in range(4):
        zi = zipfile.ZipInfo(f"assets/weights_{i}.pt", date_time=(2026, 10, 19, 0, 0, 0)); zi.compress_type = zipfile.ZIP_STORED
        fi = zp.open(f"assets/weights_{i}.pt") if i < 2 else open(W23[i], "rb")
        with fi, zo.open(zi, "w", force_zip64=True) as fo: shutil.copyfileobj(fi, fo, 1 << 24)
with zipfile.ZipFile(ZO) as z:
    assert z.testzip() is None
    for i in (0, 1): assert z.getinfo(f"assets/weights_{i}.pt").CRC == zp.getinfo(f"assets/weights_{i}.pt").CRC
    for i in (2, 3): assert z.getinfo(f"assets/weights_{i}.pt").CRC == want[i][1]
    assert zlib.crc32(z.read("run.py")) == 0x9C6C1C26, "run.py khác run_split của x2f"
    log("zip: " + ", ".join(f"{x.filename} {x.file_size} {x.CRC:#010x}" for x in z.infolist()))
zp.close()
# tự kiểm: giải nén, nạp run.py; đầu (chặt) rồi đuôi (UPAR_SPLIT_AT=0, tier 1 + flip, chỉ in)
CK = O2 / "chk"; shutil.rmtree(CK, ignore_errors=True)
with zipfile.ZipFile(ZO) as z: z.extractall(CK)
rows = va.image.tolist(); paths = [str(DATA / r) for r in rows]; idx = np.random.RandomState(1).choice(len(rows), 256, replace=False)
sys.path.insert(0, str(CK)); import torch, importlib
def check(tag, z_ref, thr):
    for k_ in [k_ for k_ in sys.modules if k_ in ("run", "model")]: del sys.modules[k_]
    run = importlib.import_module("run"); run.load_model(); t1 = time.time()
    p = np.array(run.predict_batch([dict(attribute_names=cfg["attributes"], image=rows[i], image_path=paths[i], index=int(i)) for i in idx[:128]])
                 + run.predict_batch([dict(attribute_names=cfg["attributes"], image=rows[i], image_path=paths[i], index=int(i)) for i in idx[128:]]))
    tt_ = time.time() - t1; thr_ = np.clip(np.array(thr), 1e-4, 1 - 1e-4); q = 1 / (1 + np.exp(-(z_ref[idx] - np.log(thr_ / (1 - thr_)))))
    dp = np.abs(p - q); agree = float(((p > .5) == (q > .5)).mean())
    log(f"tự kiểm {tag}: |Δp| TB {dp.mean():.5f} max {dp.max():.4f}, trùng dấu {agree:.4f} | {1000 * tt_ / 256:.1f} ms/ảnh (gồm khởi động)")
    run._S.clear(); del run; import gc; gc.collect(); torch.cuda.empty_cache(); return dp, agree
dp, ag = check("ĐẦU (F8c+L2c @224, ngưỡng solo)", 0.6 * m0 + 0.4 * m1, pc["thresholds"])
assert dp.max() < 0.05 and ag > 0.995, "đầu lệch logit val"
os.environ["UPAR_SPLIT_AT"] = "0"
dpt, agt = check("ĐUÔI (4 model tier 1 + flip, so với logit 224 không lật)", 0.24 * m0 + 0.16 * m1 + 0.3 * G + 0.3 * R, tt)
os.environ.pop("UPAR_SPLIT_AT", None); shutil.rmtree(CK, ignore_errors=True)
assert agt > 0.95, "đuôi lệch quá xa"
h = hashlib.md5(open(ZO, "rb").read()).hexdigest()[:8]
json.dump(dict(zip=a.name, md5=h, pair=meta, check_a_x2f_tail=chk_a, in_sample_head=s_h, in_sample_tail=s_t, in_sample_tail_x2f=s_old,
               selfcheck_head_max_dp=float(dp.max()), selfcheck_tail_agree=agt), open(O2 / (a.name + ".json"), "w"), indent=1, ensure_ascii=False)
log(f"ĐÃ GHI {a.name} ({ZO.stat().st_size / 1e9:.2f} GB, md5 {h})")
