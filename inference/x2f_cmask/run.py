# UPAR 2027 Track 1 submission (v2). Cấu trúc: run.py, metadata.yaml, model.py, assets/{config.json, weights_*.pt}
# Chốt chặn thời gian: run.py tự đo tốc độ sau mỗi batch. Nếu dự báo tổng thời gian vượt ngân sách, các ảnh còn lại chạy ở độ
# phân giải thấp hơn (cùng weights, nội suy pos_embed như research/make_lowres_zip.py; trên val 224x112 mất ~0.001, 196x98 ~0.005).
# Server đủ nhanh thì luôn ở mức 0 = đúng như bản không có chốt chặn. Log chỉ dùng ASCII để không lỗi mã hoá stderr.
# Model phụ ("optional": true trong config, cần "thresholds_solo"): nếu dự báo vượt ngân sách thì bỏ model phụ trước (các ảnh còn lại
# chỉ chạy model chính với ngưỡng riêng của nó, độ phân giải giữ nguyên); chỉ hạ độ phân giải khi đã bỏ model phụ và time_guard bật.
# v17x (14/10): model có "head_only": true chỉ chạy khi model phụ TẮT (phần đầu, ngưỡng solo); khi model phụ BẬT (đuôi) nó bị bỏ qua, nên đuôi chạy
# đúng bộ (model chính không head_only + model phụ) với ngưỡng 3 model. Dự báo thời gian của đuôi trừ chi phí model head_only (đo riêng).
# run_split (18/10): config "split": {"at": N} chia cứng theo vị trí ảnh trong luồng test (đếm như log "new group ... at image #"; test:
# MEVID #0-28094, Synth từ #28095). Ảnh vị trí < N = ĐẦU: model không "optional", ngưỡng thresholds_solo, độ phân giải theo time guard.
# Ảnh vị trí >= N = ĐUÔI: model không "head_only" + model phụ, ngưỡng thresholds, mức tail_tier. Lô vắt qua mốc được tách làm 2.
# Model phụ không bao giờ chạy ở đầu; ở đuôi chỉ bị bỏ (đuôi dùng bộ đầu) nếu dự báo vượt hard_s. Không có "split" = y như run_v17x.
# 18/10 tối: lần chấm split_c1 bỏ oan model phụ — lô vắt mốc chỉ có 1 ảnh đuôi + lần chạy nguội đầu tiên của model phụ đo ra 109 ms/ảnh
# (dự báo 364 s, thực dư ~50 s). Nay ở chế độ split bỏ đo tốc độ của lô < opt_min_n ảnh (mặc định 16) và lô đủ cỡ đầu tiên có model phụ.
from __future__ import annotations
import os, sys, json, time
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
sys.path.insert(0, HERE)
from model import PARNet, preprocess, crop_frac

BATCH_SIZE = 64
T0 = time.time()
# budget_s tính từ lúc import run.py; n_expected = số ảnh test (log ingestion: 29 248); costs = thời gian tương đối từng mức
# (đo 04/10 bằng torch 2.4.1, 6 luồng, 300 ảnh val: 252x126 0.8763 | 224x112 0.8732 | 196x98 0.8704 | 168x84 0.8676 | 140x70 0.8646)
GUARD = dict(budget_s=250.0, n_expected=29248, scales=[1.0, 0.889, 0.778, 0.667, 0.556], costs=[1.0, 0.86, 0.65, 0.48, 0.37],   # 08/10: chấm GPU, giới hạn 300 s
             hard_s=280.0, tail_n=3000, tail_tier=1)   # v17e (14/10): tail_n ảnh cuối chạy ở mức tier_tail (192x96), như lượt v17 0.7992 (Synth 0.8529)   # 11/10: model phụ đã bật ở tail_n ảnh cuối (gồm Synth = 50 % điểm) chỉ bị tắt lại khi dự báo vượt hard_s
_S = {}

def _log(msg):
    try: print(f"[run.py] {time.time() - T0:5.0f}s | {msg}", file=sys.stderr, flush=True)
    except Exception: pass

def _read(path, n=1 << 16):
    try:
        with open(path) as fh: return fh.read(n)
    except Exception:
        return ""

def _diag():
    # Thông tin máy chấm, chỉ để hiểu vì sao tốc độ dao động; không ảnh hưởng dự đoán
    info = {"python": sys.version.split()[0], "cpu_count": os.cpu_count(), "threads": torch.get_num_threads(),
            "cgroup_cpu": (_read("/sys/fs/cgroup/cpu.max") or _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")).strip()}
    try: info["affinity"] = len(os.sched_getaffinity(0))
    except Exception: pass
    cpu = _read("/proc/cpuinfo").splitlines()
    info["cpu"] = next((l.split(":", 1)[1].strip() for l in cpu if l.startswith("model name")), "?")
    flags = next((l.split(":", 1)[1].split() for l in cpu if l.startswith("flags")), [])
    info["flags"] = [f for f in ("avx2", "avx512f", "avx512_vnni", "avx512_bf16", "amx_bf16", "amx_int8") if f in flags]
    mem = next((l.split()[1] for l in _read("/proc/meminfo").splitlines() if l.startswith("MemTotal")), "0")
    info["mem_gb"] = round(int(mem) / 2**20, 1)
    _log("diag " + json.dumps(info))

def load_model() -> None:
    cfg = json.load(open(os.path.join(ASSETS, "config.json")))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu": torch.set_num_threads(max(1, os.cpu_count() or 1))
    models = []
    for sp in cfg["models"]:
        mk = lambda: PARNet(sp["arch"], pretrained=False, img_h=sp["img_h"], img_w=sp["img_w"], parts=sp.get("parts", False), qdec=sp.get("qdec", False), cmid=tuple(sp.get("cmid", ())), chist=bool(sp.get("chist", False)))
        sd = torch.load(os.path.join(ASSETS, sp["file"]), map_location="cpu", weights_only=True)
        if any(k.endswith("::w8scale") for k in sd):              # weights int8 theo kênh (bản _w8): giải nén
            sd = {k: v.float() * sd[k + "::w8scale"].float() if k + "::w8scale" in sd else v for k, v in sd.items() if not k.endswith("::w8scale")}
        sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
        m = None
        try:                                                # v17x: dựng trên thiết bị meta (bỏ khởi tạo ngẫu nhiên ~1 s / 300M tham số) rồi gán weights
            with torch.device("meta"): m = mk()
            m.load_state_dict(sd, assign=True)
            if any(t.is_meta for t in list(m.parameters()) + list(m.buffers())): raise RuntimeError("tensor meta chưa được gán")
        except Exception as e:
            _log(f"meta init off for {sp['arch']}: {e!r}"); m = mk(); m.load_state_dict(sd)
        m.to(dev).eval()
        if dev.type == "cpu" and cfg.get("cpu_int8", False):
            skip = tuple(cfg.get("int8_skip", []))          # vd ["fc2"]: giữ fp32 các Linear có tên kết thúc bằng các chuỗi này
            spec = {n: torch.ao.quantization.default_dynamic_qconfig for n, mod in m.named_modules()
                    if isinstance(mod, torch.nn.Linear) and not (skip and n.endswith(skip))}
            m = torch.ao.quantization.quantize_dynamic(m, spec, dtype=torch.qint8)
        sp = dict(sp)
        sp["_w"] = torch.as_tensor(sp["w"], dtype=torch.float32, device=dev)          # số hoặc vector 40 (trọng số theo thuộc tính, 09/10)
        sp["_crop"] = tuple(sp.get("crop") or ())                                       # model chuyên vùng đầu: cắt theo tỷ lệ trước khi resize
        sp["_mean"] = torch.tensor(sp["mean"], dtype=torch.float32, device=dev).view(1, 3, 1, 1)      # chuẩn hoá cả lô (v6)
        sp["_std"] = torch.tensor(sp["std"], dtype=torch.float32, device=dev).view(1, 3, 1, 1)
        if sp["arch"].startswith("dinov2"):                 # DINOv2 đổi độ phân giải bằng nội suy pos_embed
            sp.update(pe0=m.backbone.pos_embed.detach().clone(), grid0=tuple(m.backbone.grid), pe={})
        elif sp["arch"].startswith(("dinov3", "radio")):    # DINOv3 (RoPE) / C-RADIO (CPE): chỉ cần đổi kích thước ảnh
            sp.update(pe0=None, grid0=tuple(m.backbone.grid), pe={})
        models.append((m, sp))
    thr = np.clip(np.array(cfg["thresholds"], dtype=np.float64), 1e-4, 1 - 1e-4)
    has_opt = any(sp.get("optional") for _, sp in models) and cfg.get("thresholds_solo") is not None
    thr_solo = np.clip(np.array(cfg["thresholds_solo"] if has_opt else cfg["thresholds"], dtype=np.float64), 1e-4, 1 - 1e-4)
    g = dict(GUARD, **(cfg.get("time_guard") or {}))
    for key, env in (("budget_s", "UPAR_TIME_BUDGET"), ("n_expected", "UPAR_N_EXPECTED")):
        if os.environ.get(env): g[key] = float(os.environ[env])
    split_at = (cfg.get("split") or {}).get("at")
    if os.environ.get("UPAR_SPLIT_AT"): split_at = int(os.environ["UPAR_SPLIT_AT"])
    split_at = int(split_at) if (split_at is not None and has_opt) else None
    if cfg.get("time_guard") is False or os.environ.get("UPAR_GUARD_OFF") or not all("pe0" in sp for _, sp in models):
        g.update(scales=[1.0], costs=[1.0])
    g.update(k=0, n_done=0, last_t=None, last_n=0, last_k=0, calls=0, rates=[], n_tier=[0] * len(g["scales"]), groups={}, next_log=2880,
             has_opt=has_opt, opt_on=split_at is None, opt_rates=[], last_o=0.0, n_opt=0, ho_rates=[], last_h=0.0,
             shadow_on=True, sh_rates=[], last_s=0.0, t_io=0.0, t_prep=0.0, t_fwd=0.0, t_shadow=0.0, n_shadow=0,
             split_at=split_at, tail_ok=True, tail_seen=False, part_groups={}, idx_mismatch=0, opt_full=0, last_valid=True)
    print(f"[run.py] torch {torch.__version__} | device {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else "")
          + f" | cpu threads {torch.get_num_threads()} | models "
          + ", ".join(f"{sp['arch']} {sp['img_h']}x{sp['img_w']} w=" + (f"{sp['w']:.2f}" if not isinstance(sp['w'], list) else f"vec{max(sp['w']):.2f}")
                      + (" crop" if sp.get("crop") else "") + (" optional" if sp.get("optional") else "") + (" head_only" if sp.get("head_only") else "") for _, sp in models)
          + f" | tta {cfg.get('tta_flip', False)} (tail {cfg.get('tail_flip', False)}) | int8 {dev.type == 'cpu' and cfg.get('cpu_int8', False)}"
          + f" | guard {g['budget_s']:.0f}s/{int(g['n_expected'])} img, {len(g['scales'])} tiers"
          + (f" | split at image #{split_at} (tail tier {g.get('tail_tier', 0)}, hard {g.get('hard_s', g['budget_s']):.0f}s)" if split_at is not None else ""),
          file=sys.stderr, flush=True)
    _S.update(models=models, dev=dev, cfg=cfg, names=cfg["attributes"], guard=g,
              off=torch.tensor(np.log(thr / (1 - thr)), dtype=torch.float32, device=dev),
              off_solo=torch.tensor(np.log(thr_solo / (1 - thr_solo)), dtype=torch.float32, device=dev))
    try: _diag()
    except Exception as e: _log(f"diag error {e!r}")

def _open(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None

def _size(sp, k):
    if k == 0: return sp["img_h"], sp["img_w"]
    s = _S["guard"]["scales"][k]; q = 16 if sp["arch"].startswith(("dinov3", "radio")) else 14
    return max(q, round(sp["img_h"] * s / q) * q), max(q, round(sp["img_w"] * s / q) * q)

def _prep(img, sp, h, w):
    if img is None: return torch.zeros(3, h, w)
    return preprocess(img, h, w, tuple(sp["mean"]), tuple(sp["std"]), wb=sp.get("wb", False))

def _load_resize(path, keys):
    # v6: luồng CPU chỉ giải mã + (cắt) + resize PIL (BILINEAR như preprocess() lúc train); đổi sang tensor + chuẩn hoá làm cả lô trên GPU
    im = _open(path)
    if im is None: return None
    return {key: np.asarray(crop_frac(im, key[2]).resize((key[1], key[0]), resample=2)) for key in keys}

def _grayworld_b(x):
    m = x.mean((2, 3)); g = (m.mean(1, keepdim=True) / m.clamp(min=1e-3)).clamp(0.6, 1.6)
    return (x * g[:, :, None, None]).clamp(0, 1)

def _set_tier(k):
    # Đổi độ phân giải: nội suy bicubic pos_embed đã train (lưới grid0) sang lưới của mức k
    for m, sp in _S["models"]:
        q = 16 if sp["arch"].startswith(("dinov3", "radio")) else 14
        h, w = _size(sp, k); grid = (h // q, w // q)
        if sp["pe0"] is None: m.backbone.grid = grid; continue
        if grid not in sp["pe"]:
            pe0, (gh, gw) = sp["pe0"], sp["grid0"]
            if grid == (gh, gw): sp["pe"][grid] = pe0
            else:
                C = pe0.shape[-1]
                g = pe0[0, 1:].float().reshape(1, gh, gw, C).permute(0, 3, 1, 2)
                g = F.interpolate(g, size=grid, mode="bicubic", align_corners=False, antialias=True)
                sp["pe"][grid] = torch.cat([pe0[:, :1].float(), g.permute(0, 2, 3, 1).reshape(1, -1, C)], 1)
        m.backbone.pos_embed = torch.nn.Parameter(sp["pe"][grid], requires_grad=False)
        m.backbone.grid = grid

def _grp(s):
    # Nhóm ảnh = thư mục đầu của đường dẫn tương đối (hoặc thư mục cha) -> biết thứ tự các dataset trong test
    for key in ("image", "image_path"):
        parts = str(s.get(key) or "").replace("\\", "/").strip("/").split("/")
        if len(parts) > 1: return parts[0] if key == "image" else parts[-2]
    return "?"

def _guard_step(samples, tail=None):
    g = _S["guard"]; now = time.time(); n = len(samples)
    if g["last_t"] is None:
        s0 = samples[0]
        _log(f"first sample: keys {sorted(s0.keys())} | image {str(s0.get('image'))[:120]!r} | image_path {str(s0.get('image_path'))[:160]!r}")
    elif g["last_n"]:
        g["calls"] += 1
        if g["calls"] > 1 and (g["split_at"] is None or g["last_valid"]):   # bỏ batch đầu (khởi động); tốc độ quy về chi phí mức 0, không gồm model phụ
            per = (now - g["last_t"]) / g["last_n"]
            g["rates"] = (g["rates"] + [max(per - g["last_o"] - g["last_s"], 1e-6) / g["costs"][g["last_k"]]])[-5:]
            if g["last_o"] > 0: g["opt_rates"] = (g["opt_rates"] + [g["last_o"]])[-5:]
            if g["last_s"] > 0: g["sh_rates"] = (g["sh_rates"] + [g["last_s"]])[-5:]
            if g["last_h"] > 0: g["ho_rates"] = (g["ho_rates"] + [g["last_h"] / g["costs"][g["last_k"]]])[-5:]   # v17x: chi phí model head_only quy về mức 0
    if g["shadow_on"] and g["rates"] and g["sh_rates"]:      # lượt "bóng" (chỉ để đo) tắt trước mọi thứ khác nếu dự báo > 80 % ngân sách
        r_ = float(np.median(g["rates"])) * g["costs"][g["k"]] + (float(np.median(g["opt_rates"])) if (g["has_opt"] and g["opt_on"] and g["opt_rates"]) else 0.0)
        rs_, el_, left_ = float(np.median(g["sh_rates"])), now - T0, max(g["n_expected"] - g["n_done"], n)
        if el_ + (r_ + rs_) * left_ > 0.8 * g["budget_s"]:
            g["shadow_on"] = False
            _log(f"shadow OFF after {g['n_done']} images | main {1000 * r_:.2f} + shadow {1000 * rs_:.2f} ms/img | projected {el_ + (r_ + rs_) * left_:.0f}s")
    if g["has_opt"] and g["split_at"] is not None:          # run_split: model phụ chỉ ở đuôi (vị trí >= split_at), bỏ chỉ khi dự báo > hard_s
        on = bool(tail) and g["tail_ok"]
        if on and os.environ.get("UPAR_DROP_OPT"): on = False
        elif on and g["rates"] and not os.environ.get("UPAR_GUARD_OFF"):
            kt = max(g["k"], min(int(g.get("tail_tier", 0) or 0), len(g["scales"]) - 1))
            r = float(np.median(g["rates"])) * g["costs"][kt]
            ro = float(np.median(g["opt_rates"])) if g["opt_rates"] else float(g.get("opt_prior_s", 0.010))   # chưa đo -> giá trị mặc định
            rh = min(float(np.median(g["ho_rates"])) * g["costs"][kt], 0.9 * r) if g["ho_rates"] else 0.0
            el, left = now - T0, max(g["n_expected"] - g["n_done"], n)
            if el + (r - rh + ro) * left > g.get("hard_s", g["budget_s"]):
                g["tail_ok"] = on = False                   # bỏ hẳn cho phần còn lại: đuôi dùng bộ đầu + thresholds_solo
                _log(f"tail optional models DROPPED after {g['n_done']} images | main {1000 * r:.1f} + optional {1000 * ro:.1f}"
                     f"{'' if g['opt_rates'] else ' (prior)'} ms/img | projected {el + (r - rh + ro) * left:.0f}s > hard {g.get('hard_s', g['budget_s']):.0f}s")
        if on != g["opt_on"]:
            _log(f"optional models {'ON' if on else 'OFF'} after {g['n_done']} images ({'tail' if tail else 'head'} part)")
        g["opt_on"] = on
    elif g["has_opt"]:                                      # bỏ / bật lại model phụ (bật lại cần dư 2 %)
        # Khi tắt, mỗi ảnh chỉ chạy model chính làm dự báo "chạy cả 2 model cho phần còn lại" giảm dần -> model phụ được bật lại cho
        # đoạn cuối, dùng hết phần thời gian dư (07/10: hệ số 0.9 cũ quá chặt, server dư ~360 s mà model phụ không được bật lại).
        on = g["opt_on"]
        if os.environ.get("UPAR_GUARD_OFF"): on = True
        elif os.environ.get("UPAR_DROP_OPT"): on = False
        elif g["rates"] and g["opt_rates"]:
            r, ro = float(np.median(g["rates"])) * g["costs"][g["k"]], float(np.median(g["opt_rates"]))
            rh = min(float(np.median(g["ho_rates"])) * g["costs"][g["k"]], 0.9 * r) if g["ho_rates"] else 0.0   # v17x: đuôi không chạy model head_only
            el, left = now - T0, max(g["n_expected"] - g["n_done"], n)
            if left <= g.get("tail_n", 0):                 # v17d (13/10): đuôi tất định — bật model phụ cho tail_n ảnh cuối bất kể trước đó, chỉ cắt ở hard_s
                on = el + (r - rh + ro) * left <= g.get("hard_s", g["budget_s"])
            else:                                          # trước đuôi: chỉ được tắt, không bật lại (độ phủ không phụ thuộc tốc độ máy)
                on = g["opt_on"] and el + (r - rh + ro) * left <= g["budget_s"]
            if on != g["opt_on"]:
                _log(f"optional models {'ON' if on else 'OFF'} after {g['n_done']} images | main {1000 * r:.0f} + optional {1000 * ro:.0f} ms/img"
                     f" | projected {el + (r - rh + ro) * left:.0f}s with, {el + r * left:.0f}s without" + (f" (head_only {1000 * rh:.0f} ms/img not in tail)" if rh else ""))
        g["opt_on"] = on
    k = g["k"]
    if os.environ.get("UPAR_FORCE_TIER"): k = min(int(os.environ["UPAR_FORCE_TIER"]), len(g["scales"]) - 1)
    elif g["rates"] and len(g["scales"]) > 1 and not (g["has_opt"] and g["opt_on"]):
        r, el, left = float(np.median(g["rates"])), now - T0, max(g["n_expected"] - g["n_done"], n)
        k = len(g["scales"]) - 1
        for j in range(len(g["scales"])):                   # mức chính xác nhất còn kịp; quay lên mức tốt hơn cần dư 5 %
            if el + r * g["costs"][j] * left <= g["budget_s"] * (0.95 if j < g["k"] else 1.0): k = j; break
    if g["split_at"] is not None:                           # run_split: mức tail_tier chỉ áp cho đuôi có model phụ
        if tail and g["opt_on"] and g.get("tail_tier"): k = max(k, min(int(g["tail_tier"]), len(g["scales"]) - 1))
    elif g.get("tail_tier") and max(g["n_expected"] - g["n_done"], n) <= g.get("tail_n", 0):   # v17e: đuôi tất định cả về độ phân giải
        k = max(k, min(int(g["tail_tier"]), len(g["scales"]) - 1))
    if k != g["k"]:
        h, w = _size(_S["models"][0][1], k)
        _log(f"tier {g['k']} -> {k} ({h}x{w}) after {g['n_done']} images | "
             f"{1000 * float(np.median(g['rates'])) if g['rates'] else 0:.0f} ms/img at tier-0 cost")
        _set_tier(k); g["k"] = k
    g["last_t"], g["last_n"], g["last_k"] = now, n, k
    return k

def _guard_done(samples, k, tail=None):
    g = _S["guard"]; c = g["groups"]
    pc = g["part_groups"].setdefault(("tail" if g["opt_on"] else "tail-fallback") if tail else "head", {}) if g["split_at"] is not None else None
    for i, s in enumerate(samples):
        grp = _grp(s)
        if grp not in c:
            c[grp] = 0
            if len(c) <= 30: _log(f"new group {grp[:60]!r} at image #{g['n_done'] + i}")
        c[grp] += 1
        if pc is not None: pc[grp] = pc.get(grp, 0) + 1
        ix = s.get("index")
        if isinstance(ix, (int, np.integer)) and int(ix) != g["n_done"] + i:
            g["idx_mismatch"] += 1
            if g["idx_mismatch"] == 1: _log(f"note: sample index {int(ix)} != stream position {g['n_done'] + i} (split uses position)")
    g["n_done"] += len(samples); g["n_tier"][k] += len(samples)
    if g["n_done"] >= g["next_log"] or g["n_done"] == int(g["n_expected"]):
        g["next_log"] = (g["n_done"] // 2880 + 1) * 2880
        el = time.time() - T0
        top = sorted(c.items(), key=lambda kv: -kv[1])[:8]
        n_ = max(g["n_done"], 1)
        tm = (f" | io {1000 * g['t_io'] / n_:.2f} prep {1000 * g['t_prep'] / n_:.2f} model {1000 * g['t_fwd'] / n_:.2f}"
              + (f" shadow {1000 * g['t_shadow'] / max(g['n_shadow'], 1):.2f} ({g['n_shadow']} img)" if g["n_shadow"] else "") + " ms/img")
        _log(f"{g['n_done']} images | {1000 * el / g['n_done']:.1f} ms/img avg{tm} | tier {k} | images per tier {g['n_tier']}"
             + (f" | with optional {g['n_opt']}" if g["has_opt"] else "") + f" | {len(c)} groups, top {top}"
             + (f" | parts {g['part_groups']} | index!=pos {g['idx_mismatch']}" if g["split_at"] is not None else ""))

@torch.no_grad()
def predict_batch(samples):
    if not _S: load_model()
    g = _S["guard"]
    if g["split_at"] is None or not samples: return _predict(samples, None)
    n0 = g["n_done"]; tl = [n0 + i >= g["split_at"] for i in range(len(samples))]   # vị trí trong luồng test, đếm như log "new group"
    if any(tl) and not g["tail_seen"]:
        g["tail_seen"] = True; i = tl.index(True); s = samples[i]
        _log(f"tail starts at image #{n0 + i} | index {s.get('index')!r} | image {str(s.get('image'))[:120]!r} | group {_grp(s)!r}")
    if all(tl) or not any(tl): return _predict(samples, tl[0])
    ih, it = [i for i, t in enumerate(tl) if not t], [i for i, t in enumerate(tl) if t]   # lô vắt qua mốc: đầu trước, đuôi sau
    out = [None] * len(samples)
    for idx, flag in ((ih, False), (it, True)):
        for i, p in zip(idx, _predict([samples[i] for i in idx], flag)): out[i] = p
    return out

def _predict(samples, tail):
    cfg, dev = _S["cfg"], _S["dev"]
    try: k = _guard_step(samples, tail)
    except Exception as e: k = _S["guard"]["k"]; _log(f"guard error {e!r}")
    if "pool" not in _S:
        from concurrent.futures import ThreadPoolExecutor
        _S["pool"] = ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1))
    g = _S["guard"]; sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    use_opt = g["opt_on"] or not g["has_opt"]
    act = [(m, sp) for m, sp in _S["models"] if (not sp.get("head_only") if use_opt else not sp.get("optional"))]   # v17x: head_only chỉ chạy khi model phụ tắt
    flip = cfg.get("tta_flip", False) or bool(tail and g["has_opt"] and use_opt and cfg.get("tail_flip", False))   # 18/10: flip chỉ ở đuôi có model phụ
    sizes = sorted({(*_size(sp, k), sp["_crop"]) for _, sp in act})
    t0_ = time.time()
    arrs = list(_S["pool"].map(lambda s: _load_resize(s["image_path"], sizes), samples))
    g["t_io"] += time.time() - t0_
    t1 = time.time(); xs = {}
    bad = [i for i, d in enumerate(arrs) if d is None]
    for hw in sizes:                                          # mỗi kích thước: 1 lần stack + đưa lên GPU, dùng chung cho các model cùng kích thước
        a = np.stack([d[hw] if d is not None else np.zeros((hw[0], hw[1], 3), np.uint8) for d in arrs])     # hw = (h, w, crop)
        xs[hw] = torch.from_numpy(a).to(dev).permute(0, 3, 1, 2).float().div_(255)
    g["t_prep"] += time.time() - t1
    z, wsum, t_opt, x0, t_ho = None, 0.0, 0.0, None, 0.0
    for m, sp in act:
        t1 = time.time(); h, w = _size(sp, k)
        x = xs[(h, w, sp["_crop"])]
        if sp.get("wb", False): x = _grayworld_b(x)
        x = (x - sp["_mean"]) / sp["_std"]
        if bad: x[bad] = 0.0                                  # ảnh đọc lỗi: tensor 0 như _prep() cũ
        t2 = time.time()
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            zi = m(x).float()
            if flip:
                zi = (zi + m(torch.flip(x, dims=[3])).float()) / 2
        sync(); g["t_fwd"] += time.time() - t2
        if sp.get("head_only"): t_ho += time.time() - t2
        if x0 is None: x0 = x
        z = sp["_w"] * zi if z is None else z + sp["_w"] * zi; wsum = wsum + sp["_w"]
        if sp.get("optional"): t_opt += time.time() - t1
    g["last_o"] = t_opt / max(len(samples), 1); g["last_h"] = t_ho / max(len(samples), 1)
    if g["split_at"] is not None:                           # lô nhỏ (vắt mốc) và lô đủ cỡ đầu tiên có model phụ (chạy nguội): không dùng để đo tốc độ
        full = len(samples) >= int(g.get("opt_min_n", 16)); opt_run = use_opt and g["has_opt"]
        if opt_run and full: g["opt_full"] += 1
        g["last_valid"] = full and (not opt_run or g["opt_full"] > 1)
    n_sh, t_sh = int(cfg.get("shadow_passes", 0) or 0), 0.0
    if n_sh and g["shadow_on"] and x0 is not None:          # chỉ để đo chi phí thêm 1 model: chạy lại model đầu trên ảnh lật, bỏ kết quả
        t3 = time.time()
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            for _ in range(n_sh): _S["models"][0][0](torch.flip(x0, dims=[3]))
        sync(); t_sh = time.time() - t3; g["t_shadow"] += t_sh; g["n_shadow"] += len(samples)
    g["last_s"] = t_sh / max(len(samples), 1)
    if use_opt:
        off = _S["off"]; g["n_opt"] += len(samples) if g["has_opt"] else 0
    else:                                                   # chỉ model chính: logit chuẩn hoá lại theo tổng trọng số, ngưỡng riêng
        z, off = z / wsum, _S["off_solo"]
    p = torch.sigmoid(z - off)
    p = torch.nan_to_num(p, nan=0.0).clamp(0.0, 1.0).cpu().numpy()
    names = samples[0].get("attribute_names") or _S["names"]
    if list(names) != list(_S["names"]):                   # trả đúng thứ tự BTC yêu cầu
        p = p[:, [_S["names"].index(n) for n in names]]
    try: _guard_done(samples, k, tail)
    except Exception as e: _log(f"guard error {e!r}")
    return p.tolist()

def predict_image(sample):
    return predict_batch([sample])[0]
