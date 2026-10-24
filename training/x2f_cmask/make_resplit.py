# -*- coding: utf-8 -*-
"""Gộp train + val chính thức của 3 dataset rồi chia lại: mỗi dataset 15 % làm val mới, phần còn lại làm train mới.
- Chia theo NHÓM để cùng một người / chuỗi ảnh gần trùng không nằm ở cả hai phía:
  Market1501 theo mã người (tên file "0002_c1s1_..."), PA100k theo khối 20 chỉ số liền nhau, PETA theo khối 5 chỉ số liền nhau
  (PA100k / PETA không có mã người trong tên file; khối liền nhau chỉ giảm, không loại hết, ảnh gần trùng).
- Cân bằng nhãn: iterative stratification (Sechidis et al. 2011) ở mức nhóm, theo số ảnh dương từng thuộc tính: lần lượt xử lý thuộc tính
  hiếm nhất còn lại, đưa từng nhóm chứa nó vào phía đang thiếu thuộc tính đó nhất.
Ghi `research/resplit_0710/annotations/task1/{train,val}/gt.csv` (cùng định dạng file BTC) + báo cáo cân bằng.
python research/make_resplit.py [tỷ lệ val=0.15] [seed=0]"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; ANN = ROOT / "UPAR-Challenge-2027/data/annotations/task1"
F = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
OUT = ROOT / "research/resplit_0710"
raw = {s: pd.read_csv(ANN / s / "gt.csv") for s in ("train", "val")}
COL0 = raw["train"].columns[0]; ATTR = list(raw["train"].columns[1:]); assert list(raw["val"].columns) == list(raw["train"].columns)
d = pd.concat([raw["train"].assign(old="train"), raw["val"].assign(old="val")], ignore_index=True)
d["ds"] = d[COL0].str.split("/").str[0]
num = d[COL0].str.extract(r"(\d+)\.(?:jpg|png)$")[0].astype(float).fillna(-1)   # 19 ảnh Market đuôi ".jpg.jpg" (Market chia theo mã người, không dùng số này)
pid = d[COL0].str.extract(r"/(-?\d+)_c\d")[0]
d["grp"] = np.where(d.ds == "Market1501", "M" + pid.fillna("x"),
           np.where(d.ds == "PA100k", "A" + (num // 20).astype(int).astype(str), "P" + (num // 5).astype(int).astype(str)))

def strat_groups(Y, sizes, f, rng):
    """Y: (nhóm, thuộc tính) số ảnh dương; sizes: số ảnh mỗi nhóm. Trả về mảng bool: nhóm vào val."""
    ratios = np.array([f, 1 - f]); need = ratios[:, None] * Y.sum(0)[None, :]; need_n = ratios * sizes.sum()
    fold = np.full(len(sizes), -1); left = np.ones(len(sizes), bool)
    while left.any():
        c = Y[left].sum(0)
        if not (c > 0).any():                                   # nhóm không còn nhãn dương nào: chia theo số ảnh còn thiếu
            for g in rng.permutation(np.where(left)[0]):
                k = int(np.argmax(need_n)); fold[g] = k; need_n[k] -= sizes[g]; left[g] = False
            break
        j = int(np.argmin(np.where(c > 0, c, np.inf)))          # thuộc tính hiếm nhất còn lại
        for g in rng.permutation(np.where(left & (Y[:, j] > 0))[0]):
            cand = np.flatnonzero(need[:, j] == need[:, j].max())
            if len(cand) > 1: cand = cand[need_n[cand] == need_n[cand].max()]
            k = int(rng.choice(cand)); fold[g] = k; need[k] -= Y[g]; need_n[k] -= sizes[g]; left[g] = False
    return fold == 0

rng = np.random.default_rng(SEED); d["new"] = "train"
for ds, x in d.groupby("ds"):
    g = x.groupby("grp"); Y = g[ATTR].sum().values.astype(float); sizes = g.size().values.astype(float)
    to_val = strat_groups(Y, sizes, F, rng)
    d.loc[x.index[x.grp.isin(g.size().index[to_val])], "new"] = "val"

rep = []
for ds, x in d.groupby("ds"):
    tr, va = x[x.new == "train"], x[x.new == "val"]
    pt, pv = tr[ATTR].mean(), va[ATTR].mean()
    rare = pt[pt < 0.02].index
    rep.append(dict(dataset=ds, ảnh=len(x), val=len(va), tỷ_lệ_val=round(len(va) / len(x), 4), nhóm=x.grp.nunique(),
                    nhóm_cả_hai_phía=int(len(set(tr.grp) & set(va.grp))),
                    lệch_tỷ_lệ_dương_max=round(float((pt - pv).abs().max()), 4),
                    lệch_tương_đối_TB_thuộc_tính_hiếm=round(float(((pt[rare] - pv[rare]).abs() / pt[rare].clip(1e-9)).mean()), 3) if len(rare) else None,
                    val_từ_train_cũ=int((va.old == "train").sum()), val_từ_val_cũ=int((va.old == "val").sum())))
R = pd.DataFrame(rep); pd.set_option("display.width", 250); print(R.to_string(index=False))
AGE = ["Age-Young", "Age-Adult", "Age-Old"]
print("\nTỷ lệ dương tuổi (train mới / val mới):"); print(d.groupby(["ds", "new"])[AGE].mean().round(4).to_string())
for s in ("train", "val"):
    p = OUT / "annotations/task1" / s; p.mkdir(parents=True, exist_ok=True)
    d[d.new == s][[COL0] + ATTR].to_csv(p / "gt.csv", index=False)
(OUT / "README.md").write_text(
    f"# Chia lại train / val (07/10)\n\nGộp train + val chính thức của Market1501, PA100k, PETA; mỗi dataset {F:.0%} làm val mới, chia theo nhóm "
    f"(Market: mã người; PA100k: khối 20 chỉ số; PETA: khối 5 chỉ số), cân bằng nhãn bằng iterative stratification. Seed {SEED}.\n"
    f"Script: `research/make_resplit.py`. Train mới {int((d.new == 'train').sum())} ảnh, val mới {int((d.new == 'val').sum())} ảnh.\n\n"
    + R.to_markdown(index=False) + "\n", encoding="utf-8")
d[[COL0, "ds", "old", "new", "grp"]].to_csv(OUT / "split_map.csv", index=False)
print(f"\nGhi: {OUT / 'annotations/task1'} | train {int((d.new == 'train').sum())}, val {int((d.new == 'val').sum())}")
