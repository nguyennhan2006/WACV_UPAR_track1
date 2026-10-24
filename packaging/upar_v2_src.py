# %% [markdown]
# # UPAR 2027 Track 1 · v4 · 252×126 + SmoothQuant int8 (chạy < 1 giờ) · tự chọn model + tự tìm ngưỡng (Kaggle, GPU T4 ×2)
#
# **v4 (01/10/2026):** server chấm 29 248 ảnh và phải xong trong 1 giờ. fp32 252×126 quá giờ, còn int8 thường mất khoảng 0.04 điểm trên val. v4 **giữ nguyên 252×126** và đóng gói bằng **SmoothQuant + int8, giữ `fc2` ở fp32**. Đo bằng torch 2.4.1 trên weights thật: điểm val ngang fp32, tốc độ khoảng 1.05× bản 0.776.
#
# Notebook này kế thừa bản đạt **0.776** (`upar_train_dinov2-6`) và chạy **một lần từ đầu đến cuối** để ra file nộp.
#
# **Điểm mới so với bản 0.776**
# 1. **Thêm backbone CLIP ViT-B/16** (OpenAI, viết lại bằng torch thuần, key khớp HuggingFace). Head được khởi tạo bằng *text embedding* của câu mô tả từng thuộc tính (ý tưởng AttriVision, WACVW 2025). Input 256×128 cho 16×8 = 128 patch (DINOv2 252×126 có 162 patch + 4 register). Đo bằng torch 2.4.1 CPU: CLIP fp32 ở 256×128 nhanh ngang DINOv2 int8 của bản cũ, nên không lo vượt thời gian kể cả khi phải tắt int8.
# 2. **Tự chọn model**: train lần lượt các ứng viên trong `CANDIDATES` (mặc định CLIP-B/16 và DINOv2-B/14). Nếu *Add Input* bản nộp cũ, notebook đưa bản cũ vào so sánh luôn. Model thắng được chọn theo **điểm cross-fit trung thực** trên val.
# 3. **Tự tìm ngưỡng + THR_SHRINK bằng cross-fitting 2 fold** (Market chia theo identity). Ngưỡng tìm trên nửa này được chấm trên nửa kia, nên không còn lạc quan như calibrate in-sample. Shrink chọn tự động thay vì cố định 1.0.
# 4. **Tự quyết định int8**: chạy zip bằng torch 2.4.1 CPU. Nếu int8 làm giảm điểm quá `INT8_MAX_DROP` thì tự đóng gói lại bản fp32.
# 5. Cấu trúc zip theo mẫu của BTC: `run.py` và `metadata.yaml` ở gốc, `model.py` ở gốc, `assets/weights.pt` và `assets/config.json`.
#
# **Cách chạy**
# - Settings: **Accelerator = GPU T4 ×2**, **Internet = On**. Nếu *Save & Run All* thì chọn GPU T4 ×2 trong hộp thoại Save Version.
# - *Add Input* (không bắt buộc): (a) Dataset chứa ảnh `Market1501/`, `PA100k/`, `PETA/`, nếu không có thì notebook tự tải; (b) **file nộp cũ 0.776** (zip hoặc thư mục có `run.py`) để so sánh. Kaggle tự giải nén zip.
# - Chạy **Run All**, không cần sửa gì. Thời gian ước tính khoảng 2–3 giờ cho 2 ứng viên × 8 epoch.
# - Kết quả: **`/kaggle/working/SUBMIT_v2_<arch>_<thời gian>.zip`**. Tải về và nộp. Bảng so sánh ứng viên nằm ở mục 7.
#
# `DEBUG = True` chạy thử toàn bộ pipeline trong khoảng 10 phút với dữ liệu nhỏ. **Không nộp bản debug.**

# %% [markdown]
# ## 0. Config

# %%
from pathlib import Path
import os, sys, json, time, math, random, shutil, subprocess, io, glob, gc, re, zipfile, importlib, importlib.util, itertools, bisect
import numpy as np, pandas as pd

IS_KAGGLE = Path("/kaggle/working").exists()
NB_T0 = time.time()                 # mốc đầu notebook (giới hạn 12 giờ của Kaggle tính từ lúc phiên bắt đầu)
LOCAL_DEBUG = (not IS_KAGGLE) and os.environ.get("UPAR_LOCAL_DEBUG", "1") == "1"   # chạy thử trên máy không có GPU

CFG = dict(
    CANDIDATES    = ["dinov2_vitb14_reg"],  # thêm biến thể thắng LODO, vd "dinov2_vitb14_reg_best" (xem HUONG_DAN_CHAY.md mục 4)  # train lần lượt, tự chọn model / ensemble tốt nhất theo điểm cross-fit
    ENSEMBLE      = True,        # thử mọi cặp ứng viên (trộn logit), chỉ giữ cặp có chi phí CPU <= MAX_REL_COST
    MAX_REL_COST  = 1.35,        # chi phí CPU tối đa (× mốc DINOv2-B int8 252x126). fp32 252x126 (bản 0.7772, chạy kịp 3/4 lần) ≈ 1.25-1.3×
    ENS_WEIGHTS   = [0.3, 0.4, 0.5, 0.6, 0.7],
    ENS_MIN_GAIN  = 0.002,       # ensemble chỉ được chọn nếu hơn model đơn tốt nhất ít nhất chừng này (vì tốn thêm thời gian)
    USE_BASELINE  = True,        # nếu có bản nộp cũ trong /kaggle/input -> đưa vào so sánh (không train lại)
    DEBUG         = os.environ.get("UPAR_DEBUG", "0") == "1",   # 13/10: env cho smoke Colab CPU       # True: 1000 train / 500 val mỗi dataset, 1 epoch -> chạy thử
    EPOCHS        = 8,
    BATCH         = 128,
    WD            = 0.05,
    WARMUP_EP     = 1.0,
    EMA           = 0.999,
    LABEL_SMOOTH  = 0.05,
    DS_BALANCE    = 0.5,         # trọng số lấy mẫu ∝ n_dataset^(-alpha)
    AUG_LOWRES_P  = 0.4,
    AUG_JPEG_P    = 0.3,
    TEXT_INIT     = True,        # CLIP: khởi tạo head bằng text embedding
    TEXT_SCALE    = 50.0,
    SHRINK_GRID   = [0.0, 0.25, 0.5, 0.75, 1.0],   # ngưỡng xuất = sigmoid(k * logit(ngưỡng calibrate)), k chọn bằng cross-fit
    CPU_INT8      = False,       # False = fp32 như bản 0.7772 (đã chạy kịp 3 lần trên server, ~1.25-1.3x mốc int8) -> ưu tiên điểm
                                 # "smooth" = SmoothQuant + int8, giữ fc2 fp32: nhanh hơn ~19% (52 phút trên server), mất ~0.003 (LODO)
                                 #            -> dùng khi biến thể tốn thêm thời gian (vd 280x126) | True = int8 thường: mất ~0.047, KHÔNG dùng
    SQ_ALPHA      = 0.5,
    INT8_SKIP     = ["fc2"],
    INT8_MAX_DROP = 0.005,
    TTA_FLIP      = False,       # server BTC từng quá thời gian khi bật TTA với ViT-B -> giữ False
    LODO_LAST_EPOCH = True,      # LODO: lấy epoch cuối thay vì epoch có val 2 dataset đã thấy cao nhất (03/10: L bỏ PETA bị chọn epoch 1/8, mất 0.013)
    FINAL_FULL    = False,       # True: train lại model thắng trên train+val (CHỈ khi BTC cho phép dùng val để train)
    WORKERS       = int(os.environ.get("UPAR_WORKERS", min(4, os.cpu_count() or 1))),   # 13/10: env cho Colab A100 (nhiều CPU hơn)
    MULTI_GPU     = True,
    LOG_EVERY     = 50,
    SEED          = 42,
    CHECK_TORCH241= os.environ.get("UPAR_CHECK_TORCH241", "1") == "1",   # 13/10: Colab A100 tắt (đóng gói + kiểm 2.4.1 làm trên Kaggle CPU)
    TIME_GUARD    = False,   # 07/10: tắt chốt chặn thời gian (nghi làm MEVID tụt khi server chậm); bản 3 805 s vẫn được chấm
    HF_SECRET     = "WACV-UPAR-NTN",   # tên Kaggle Secret chứa token Hugging Face (cần cho DINOv3)
    CLEAN_OUTPUT  = True,
    ANN_PREFER    = "",          # "" = annotation chính thức; "resplit" = train / val chia lại (research/make_resplit.py, dataset upar-resplit-0710)
    # ---- 08/10/2026: BTC chấm trên GPU (Nvidia A10 24 GB, 20 CPU, tối đa 5 phút / lần chạy) ----
    GPU_SUBMIT    = True,        # chọn model theo ước tính giây trên A10 (không đo CPU), kiểm torch 2.4.1 nhẹ, run.py có chốt chặn GUARD_BUDGET
    MAX_EST_SEC   = 225.0,       # ngân sách ước tính cho model đơn / model chính / TTA (giới hạn thật 300 s)
    MAX_EST_SEC_ENS = 240.0,     # hệ ghép: tổng ước tính (model phụ là "optional": máy chậm thì run.py bỏ nó trước, quay về ngưỡng của model chính)
    A10_GPU_MS    = 2.46,        # log G0 (08/10): DINOv3-L 256x128 fp16 lô 64 trên A10 = 2.46 ms/ảnh; model khác ∝ tham số × token
    A10_PREP_MS   = 0.15,        # resize + đưa lên GPU cho mỗi kích thước ảnh, ∝ số pixel (bài v12 09/10, run.py v6: CPU chỉ còn ~0.75 ms/ảnh; v5 đo 1.26)
    A10_FIXED_MS  = 0.45,        # đọc ảnh + phần Python còn lại (v12: L3 + L2 chạy 184 s = 6.3 ms/ảnh, GPU ước tính 5.55 ms)
    A10_LOAD_S    = 5.0,         # khởi động + nạp model đầu (log G0: 5 s); mỗi model thêm +2 s
    GUARD_BUDGET  = 250.0,       # run.py: bỏ model phụ rồi mới hạ độ phân giải nếu dự báo tổng thời gian vượt mức này
    TTA_EVAL      = True,        # thử TTA lật ngang cho model đơn (chỉ khi ước tính ×2 vẫn vừa ngân sách)
    ENS_MAX       = 3,           # ghép tham lam tối đa chừng này model (mỗi bước phải hơn ≥ ENS_MIN_GAIN)
    ENS_ALPHAS    = [0.2, 0.3, 0.4, 0.5],
    REUSE_ONLY    = False,       # True: chỉ dùng run có sẵn trong Input, không train (notebook ghép)
    REUSE_LOGITS_ONLY = False,   # 13/10 (LODO E35c/d): dùng lại run cũ chỉ có val_logits + meta (không weights) — đủ để chấm LODO, không dùng được cho WiSE / soup / FRDL / zip
    SEL_DS        = None,        # dataset dùng để chọn / so; None = tự động (split mới: Market1501 + PA100k vì PETA rò rỉ người)
    SEL_DROP_OLDTRAIN_MARKET = False,  # split mới: bỏ ảnh Market của val mới lấy từ train chính thức khi chọn và calibrate
                                       # (model học từ teacher split chính thức, vd bản 1008, bị đội điểm ở đó: +0.022 AUC)
    T241_N_PER    = 32,          # số ảnh / dataset để kiểm zip bằng torch 2.4.1 CPU (model L chạy CPU rất chậm)
    W8_ABOVE_MB   = 700.0,       # zip lớn hơn mức này (vd 2 model L ~1.1 GB) -> xuất thêm bản weights int8 theo kênh (~1/2) phòng Codabench chặn
    # ---- 10/10/2026: model lớn (DINOv3 ViT-H+ 840M) trên T4 ×2 ----
    TRAIN_BUDGET_H = float(os.environ.get("UPAR_TRAIN_BUDGET_H", 10.3)),       # AUTO_EP: số epoch = số epoch kịp xong trước mốc này (giờ, tính từ đầu notebook); còn ~1.5 giờ cho đóng gói
    TIME_LIMIT_H  = float(os.environ.get("UPAR_TIME_LIMIT_H", 11.0)),        # chốt cứng: quá mốc này thì dừng train, lưu EMA hiện tại (Kaggle cắt ở 12 giờ là mất hết Output)
    FIT_MAX_GB    = float(os.environ.get("UPAR_FIT_MAX_GB", 13.8)),        # AUTO_FIT: đỉnh bộ nhớ cho phép trên mỗi T4 (15 GB dùng được)
)
if CFG["GPU_SUBMIT"]:
    CFG["SHRINK_GRID"] = [1.0]   # k = 1 tốt nhất ở mọi fold LODO (E0, E23); chọn theo val chỉ là nhiễu (bài 0.7806 lỡ chọn 0.75)
# Siêu tham số riêng từng kiến trúc. IMG luôn là bội số của patch.
_DINO_B = dict(ARCH="dinov2_vitb14_reg", IMG=(252, 126), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0)
_DINO_S = dict(ARCH="dinov2_vits14_reg", IMG=(252, 126), LR=1e-4, LLRD=0.75, DROP_PATH=0.05, HEAD_LR=1e-3, GRAD_CLIP=5.0)
# E24 (07/10): augmentation theo nhóm thuộc tính. Mỗi ảnh bốc 1 chính sách; loss (và KD) chỉ tính trên nhóm mà chính sách đó giữ nhãn.
# H = kính / mũ / tóc (chi tiết nhỏ ở đầu, mất khi ảnh quá nhỏ), C = 24 màu (hỏng khi đổi hue / xám / chỉnh sáng mạnh), P = toàn thân.
_V3B = dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, EPOCHS=12)
_V3B8 = dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0)   # = mốc LODO dinov3_vitb16 (8 epoch)
# E35 (12/10): ảnh train phục hồi offline (upar_restore_gen.ipynb). GATE: che loss nhóm màu áo / quần khi ΔE màu trung bình vùng giữa ảnh vào và ảnh phục hồi
# vượt CẢ phân vị Q LẪN DE_MIN, che nhóm đầu (tóc, kính, mũ) khi PSNR vùng đầu dưới CẢ phân vị 1-Q LẪN PS_MAX (= ảnh phục hồi bịa chi tiết).
# Bản thử 12/10 (48 ảnh): ΔE màu vùng trung vị ~1, tối đa 3.4 (ESRGAN / SinSR gần như không đổi màu); PSNR đầu trung vị ~29, ca mặt bị vẽ lại ~23-25.
_RS_GATE = dict(Q=0.9, DE_MIN=3.0, PS_MAX=25.0)
# Căn cứ (research/aug_sensitivity.py, model 0.7831, 1 200 ảnh val): ảnh nhỏ / kiểu CCTV làm AUC kính −0.14..−0.17, mũ −0.11, màu chỉ −0.01..−0.03;
# ảnh xám / xoay hue gần như không đổi nhóm H, P (≤ 0.013 AUC) -> H/P vốn không dựa vào màu; sáng ×1.5 / nhạt ×0.4 giữ AUC màu nhưng BA màu quần
# −0.05..−0.07 (logit màu trôi theo phơi sáng) trong khi nhãn MEVID theo bộ quần áo, không đổi theo camera.
_AUG_CAM = dict(CJ_P=1.0, BRIGHT=0.4, CONTRAST=0.4, SAT=0.3, WB_P=0.5, NOISE_P=0.3, NOISE_S=(0.01, 0.04))  # camera / phơi sáng khác: giữ nhãn mọi nhóm
_AUG_DEG = dict(LOWRES_P=0.9, LOWRES_W=(14, 40), JPEG_P=0.6, JPEG_Q=(15, 50), NOISE_P=0.5, NOISE_S=(0.02, 0.08), BLUR_P=0.4, WB_P=0.5)  # camera xấu: màu còn, chi tiết đầu mất
_AUG_PHOTO = dict(CJ_P=1.0, BRIGHT=0.5, CONTRAST=0.5, SAT=0.9, HUE=0.5, GRAY_P=0.25, LOWRES_P=0.15)   # phá màu, giữ hình: H / P không được dựa vào màu
_GAUG = [dict(P=0.35, LOSS="HPC"), dict(P=0.25, AUG=_AUG_CAM, LOSS="HPC"), dict(P=0.2, AUG=_AUG_DEG, LOSS="PC"), dict(P=0.2, AUG=_AUG_PHOTO, LOSS="HP")]
_GAUG_NOMASK = [dict(p_, LOSS="HPC") for p_ in _GAUG]          # đối chứng: cùng phân bố augmentation, mọi thuộc tính đều học trên mọi ảnh
_RC = dict(P=0.5, POW=0.5, COVER=0.22, SMIN=0.25, VMIN=0.22)   # đổi màu áo / quần Blue, Green, Purple sang màu khác (ưu tiên màu hiếm), đổi nhãn theo
_KDL3 = dict(TEACHERS=["dinov3_vitl16"], LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True)                 # KD của bản 0.7831
# 13/10 T2 (xem t2_targets): file teacher trong Input + nhóm thuộc tính (tên đầy đủ hoặc tiền tố)
_T2_FILES = dict(FILE="t2audit_teacher_P1_market.npy", ROWS="t2audit_teacher_rows.csv")
_T2_EXCL = ["Accessory-Glasses-", "Hair-Length-"]
_T2_PERSIST = ["Age-", "Gender-Female", "Accessory-Bag", "Accessory-Backpack", "Accessory-Hat", "UpperBody-Color-", "LowerBody-Color-"]
# 13/10 E2(mevid): suy giảm kiểu CCTV, GIỮ mọi nhãn (quy ước MEVID gán theo bộ trang phục: phần bị che / mờ vẫn dương). Khoá không nêu
# (BLUR_P 0.15, ERASE_P 0.3, CJ) giữ mặc định make_train_tf. Khác _AUG_DEG / dgaug (đã khép, cỡ B): cỡ L/H+, có TruncOcc, cổng theo proxy MEVID.
_AUG_CCTV = dict(LOWRES_P=0.6, LOWRES_W=(14, 48), JPEG_P=0.5, JPEG_Q=(15, 60), MOTION_P=0.2, NOISE_P=0.3, NOISE_S=(0.01, 0.05),
                 TRUNC_P=0.25, TRUNC=(0.15, 0.4), CT_P=0.2)
ARCH_CFG = {
    # ---- E2 (kịch bản K1-K6), cùng model gốc DINOv2-B 252x126, chỉ đổi cách train ----
    "dinov2_vitb14_reg_dsbias":  dict(_DINO_B, DS_BIAS=True),                          # K2 bias riêng từng dataset
    "dinov2_vitb14_reg_wb":      dict(_DINO_B, WB=True),                               # K3 gray-world khi train VÀ khi suy luận
    "dinov2_vitb14_reg_wbaug":   dict(_DINO_B, AUG=dict(WB_P=0.5)),                    # K3b chỉ jitter white balance khi train
    "dinov2_vitb14_reg_rfs":     dict(_DINO_B, RFS_T=0.03),                            # K4 lặp mẫu thuộc tính hiếm
    "dinov2_vitb14_reg_groupce": dict(_DINO_B, GROUP_CE=0.5),                          # K5 softmax phụ cho tuổi / loại quần
    "dinov2_vitb14_reg_280":     dict(_DINO_B, IMG=(280, 126)),                        # K6 thêm chi tiết (đóng gói SmoothQuant)
    "dinov2_vitb14_reg_kd":      dict(_DINO_B, KD=dict(TEACHERS=["dinov2_vitb14_reg", "dinov2_vitb14_reg_196", "clip_vitb16_160"],
                                                       LAMBDA=0.5, T=1.0)),              # K1 chưng cất từ ensemble
    # ---- E3 / E4 (K8-K13, nhắm Instance-F1), cùng model gốc ----
    "dinov2_vitb14_reg_colorce":   dict(_DINO_B, COLOR_CE=0.5),                        # K8 softmax phụ cho màu áo / màu quần
    "dinov2_vitb14_reg_softf1":    dict(_DINO_B, SOFTF1=0.5),                          # K9 loss soft Instance-F1 phụ
    "dinov2_vitb14_reg_mixstyle":  dict(_DINO_B, MIXSTYLE=dict(P=0.5, ALPHA=0.1, BLOCKS=[1, 3, 5])),   # K10 trộn thống kê token giữa ảnh
    "dinov2_vitb14_reg_qdec":      dict(_DINO_B, QDEC=True),                           # K11 decoder truy vấn + self-attention giữa thuộc tính
    "dinov2_vitb14_reg_colorsoft": dict(_DINO_B, COLOR_SOFT=0.15),                     # K12 nhãn mềm theo độ gần màu
    "dinov2_vitb14_reg_wbce05":    dict(_DINO_B, WBCE_POW=0.5),                        # K13 trọng số wBCE mũ 0.5
    # ---- E5 (K16): cùng cấu hình bản gốc, chỉ đổi seed -> đo nhiễu giữa 2 lần train; cũng là nguyên liệu cho soup (K15) ----
    "dinov2_vitb14_reg_seed2":     dict(_DINO_B, SEED=2),
    "dinov2_vitb14_reg_seed3":     dict(_DINO_B, SEED=3),
    # ---- E8: quy ước màu như test (mỗi vùng 1 màu, áo hoạ tiết = Other) + teacher điền nhãn thiếu ----
    "dinov2_vitb14_reg_cfix":          dict(_DINO_B, CFIX=dict(PETA_OTHER_NEG=True)),           # H1 + H2
    "dinov2_vitb14_reg_cfix_fill":     dict(_DINO_B, CFIX=dict(PETA_OTHER_NEG=True),            # + K19
                                            KD=dict(TEACHERS=["dinov2_vitb14_reg"], MODE="fill", LAMBDA=1.0)),
    "dinov2_vitb14_reg_cfix_fill_cce": dict(_DINO_B, CFIX=dict(PETA_OTHER_NEG=True),            # + K8
                                            KD=dict(TEACHERS=["dinov2_vitb14_reg"], MODE="fill", LAMBDA=1.0), COLOR_CE=0.5),
    # ---- E6b (K17): chưng cất từ DINOv2-L (teacher chỉ dùng lúc train, suy luận vẫn là B) ----
    "dinov2_vitb14_reg_kdL":       dict(_DINO_B, KD=dict(TEACHERS=["dinov2_vitl14_reg"], LAMBDA=0.5, T=1.0)),
    "dinov2_vitb14_reg_kdLB":      dict(_DINO_B, KD=dict(TEACHERS=["dinov2_vitl14_reg", "dinov2_vitb14_reg"], W=[0.6, 0.4], LAMBDA=0.5, T=1.0)),
    # ---- E10 (03/10): gộp 2 biến thể thắng E4 (K13 wbce05 +0.0017, K11 qdec +0.0014) và chưng cất từ L ----
    "dinov2_vitb14_reg_wq":        dict(_DINO_B, WBCE_POW=0.5, QDEC=True),
    "dinov2_vitb14_reg_kdL_wq":    dict(_DINO_B, WBCE_POW=0.5, QDEC=True, KD=dict(TEACHERS=["dinov2_vitl14_reg"], LAMBDA=0.5, T=1.0)),
    # ---- E11 (04/10): chưng cất mạnh hơn từ L cùng fold. kdL mới lấy ~1/4 lợi thế của L (+0.0025 / +0.0098), test xác nhận cùng chiều (+0.0013).
    # T: nhiệt độ cho cả teacher và student, loss ×T² (Hinton); T = 1 giống hệt kdL. ONLINE: teacher chạy trên chính ảnh đã augment mà
    # student thấy (consistent teaching, Beyer et al. CVPR'22) thay vì nhãn mềm tính trước trên ảnh gốc. FEAT: thêm 1 − cos giữa đặc trưng
    # student (qua 1 lớp chiếu, bỏ sau khi train) và đặc trưng teacher [CLS, trung bình patch].
    "dinov2_vitb14_reg_kdL_l08T2":  dict(_DINO_B, KD=dict(TEACHERS=["dinov2_vitl14_reg"], LAMBDA=0.8, T=2.0)),
    "dinov2_vitb14_reg_kdLon":      dict(_DINO_B, KD=dict(TEACHERS=["dinov2_vitl14_reg"], LAMBDA=0.5, T=1.0, ONLINE=True)),
    "dinov2_vitb14_reg_kdLon_feat": dict(_DINO_B, KD=dict(TEACHERS=["dinov2_vitl14_reg"], LAMBDA=0.5, T=1.0, ONLINE=True, FEAT=1.0)),
    "dinov2_vitb14_reg_kdLon_ep12": dict(_DINO_B, EPOCHS=12, KD=dict(TEACHERS=["dinov2_vitl14_reg"], LAMBDA=0.5, T=1.0, ONLINE=True)),
    # FEAT_REG: thêm 1 − cos giữa đặc trưng 3 vùng (đầu / thân trên / thân dưới) của student và teacher -> nhắm lợi thế của L ở kính, tuổi, túi
    "dinov2_vitb14_reg_kdLon_featR": dict(_DINO_B, KD=dict(TEACHERS=["dinov2_vitl14_reg"], LAMBDA=0.5, T=1.0, ONLINE=True, FEAT=1.0, FEAT_REG=1.0)),
    # ---- E12 (04/10): màu chiếm 57 % mA mất trên test, backbone lớn không giúp (đặc trưng DINO bất biến với màu) ----
    "dinov2_vitb14_reg_cmid":       dict(_DINO_B, CMID=(3, 7)),                 # head màu từ patch token sau khối 4 và 8
    "dinov2_vitb14_reg_chist":      dict(_DINO_B, CHIST=True),                  # histogram màu tường minh vùng áo / quần
    "dinov2_vitb14_reg_cmid_chist": dict(_DINO_B, CMID=(3, 7), CHIST=True),
    # ---- E15 (05/10): ghép kdL (+0.0025 TB 3 fold; mũ, túi) và cmid (+0.0027; màu quần, kính) - hai cơ chế khác nhau ----
    "dinov2_vitb14_reg_kdL_cmid":   dict(_DINO_B, CMID=(3, 7), KD=dict(TEACHERS=["dinov2_vitl14_reg"], LAMBDA=0.5, T=1.0)),
    # ---- E13 (04/10): bớt overfit vào 3 domain nguồn, giữ đặc trưng pretrain (in-domain 0.886 vs domain lạ 0.822; 12 epoch không giúp) ----
    "dinov2_vitb14_reg_freeze6":    dict(_DINO_B, FREEZE=6),                    # đóng băng patch embed, pos/cls/register và 6 khối đầu
    "dinov2_vitb14_reg_llrd06":     dict(_DINO_B, LLRD=0.6),                    # lớp dưới gần như đứng yên (khối 1: 0.6^12 ≈ 0.002× LR)
    "dinov2_vitb14_reg_lpft":       dict(_DINO_B, LPFT=1, EPOCHS=9),            # LP-FT: 1 epoch chỉ train head, rồi 8 epoch fine-tune toàn bộ
    # ---- DINOv3 (05/10): bài nộp DINOv3-B 256x128 + query head đạt 0.7806 trên test (kdL 0.7785); kiểm bằng LODO ----
    "dinov3_vitb16":            dict(IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0),
    "dinov3_vitb16_cmid_chist": dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                                     CMID=(3, 7), CHIST=True),
    # LR 3e-5: layer scale của DINOv3 lớn ~2× DINOv2 từ khối 5 trở lên -> cùng LR làm đặc trưng đổi mạnh hơn (research/probe_backbones.py)
    "dinov3_vitb16_lr3":        dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=3e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0),
    "dinov3_vitb16_cmid_chist_lr3": dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=3e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                                         CMID=(3, 7), CHIST=True),
    # E15 (05/10): nền DINOv3; KD từ DINOv2-L cùng fold (nhãn mềm tính trước, không phụ thuộc kiến trúc student)
    "dinov3_vitb16_kdL":            dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                                         KD=dict(TEACHERS=["dinov2_vitl14_reg"], LAMBDA=0.5, T=1.0)),
    "dinov3_vitb16_kdL_cmid_chist": dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                                         CMID=(3, 7), CHIST=True, KD=dict(TEACHERS=["dinov2_vitl14_reg"], LAMBDA=0.5, T=1.0)),
    # ---- E16 (05/10): teacher DINOv3-L/16 (cùng 256x128, patch 16 với student -> dùng được KD online + đặc trưng theo vùng) ----
    "dinov3_vitl16":                dict(IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64),
    "dinov3_vitb16_kdL3":           dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                                         KD=dict(TEACHERS=["dinov3_vitl16"], LAMBDA=0.5, T=1.0)),
    "dinov3_vitb16_kdL3_cmid_chist": dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                                         CMID=(3, 7), CHIST=True, KD=dict(TEACHERS=["dinov3_vitl16"], LAMBDA=0.5, T=1.0)),
    "dinov3_vitb16_kdL3on_featR":   dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                                         KD=dict(TEACHERS=["dinov3_vitl16"], LAMBDA=0.5, T=1.0, ONLINE=True, FEAT=1.0, FEAT_REG=1.0)),
    # E16b: ghép KD online + đặc trưng vùng (E16 tốt nhất, +0.0078) với nhánh màu (bù màu quần mà featR làm hỏng ở E11b)
    "dinov3_vitb16_kdL3on_featR_cmid_chist": dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                                         CMID=(3, 7), CHIST=True, KD=dict(TEACHERS=["dinov3_vitl16"], LAMBDA=0.5, T=1.0, ONLINE=True, FEAT=1.0, FEAT_REG=1.0)),
    # E17 (06/10): bản dẫn điểm được nói là train 20 epoch -> kiểm lại bằng LODO trên DINOv3 (log mỗi epoch có điểm domain chưa thấy)
    "dinov3_vitb16_ep20":           dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, EPOCHS=20),
    "dinov3_vitb16_kdL3on_featR_ep20": dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, EPOCHS=20,
                                         KD=dict(TEACHERS=["dinov3_vitl16"], LAMBDA=0.5, T=1.0, ONLINE=True, FEAT=1.0, FEAT_REG=1.0)),
    # ---- E18 (06/10): công thức bản test 0.7831 (DINOv3-B, KD online từ DINOv3-L, KD trên cả 40 thuộc tính, 12 epoch) và ghép với featR / màu ----
    "dinov3_vitb16_kdL3on_all_ep12":   dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, EPOCHS=12, KD=dict(TEACHERS=["dinov3_vitl16"], LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True)),
    # E23 (07/10): công thức 0.7831 với lịch dài hơn (KD online làm chính quy hoá: 8 -> 12 epoch đã tăng ở fold PA100k)
    "dinov3_vitb16_kdL3on_all_ep16":   dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, EPOCHS=16, KD=dict(TEACHERS=["dinov3_vitl16"], LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True)),
    # E26 (H7, 07/10): teacher theo nhóm thuộc tính (docs mục 38 phần 1). H = tóc/kính/mũ, P = tuổi/giới/dáng/túi, C = 24 màu.
    "dinov3_vitb16_kdL3L2_ep12":       dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, EPOCHS=12,
                                            KD=dict(TEACHERS=["dinov3_vitl16", "dinov2_vitl14_reg"], GROUPS=["HPC", "HP"], LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True)),
    "dinov3_vitb16_kdL3L2_ccC_ep12":   dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, EPOCHS=12,
                                            KD=dict(TEACHERS=["dinov3_vitl16", "dinov2_vitl14_reg", "dinov3_vitb16_cmid_chist", "clip_vitb16_160"],
                                                    GROUPS=["HP", "HP", "C", "C"], W=[1.0, 1.0, 0.7, 0.3], LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True)),
    "dinov3_vitb16_kdL3L2_ep20":       dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, EPOCHS=20,
                                            KD=dict(TEACHERS=["dinov3_vitl16", "dinov2_vitl14_reg"], GROUPS=["HPC", "HP"], LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True)),
    "dinov3_vitb16_kdL3L2_ep16":       dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, EPOCHS=16,
                                            KD=dict(TEACHERS=["dinov3_vitl16", "dinov2_vitl14_reg"], GROUPS=["HPC", "HP"], LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True)),
    "dinov3_vitb16_kdL3noC_ep12":      dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, EPOCHS=12,
                                            KD=dict(TEACHERS=["dinov3_vitl16"], ONLY="HP", LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True)),   # đối chứng: màu chỉ học nhãn thật
    # E22 (07/10): công thức 0.7831 + nhánh màu đọc khối 6 và 10 (tap 5, 9; probe docs mục 33) so với khối 4 và 8 cũ (tap 3, 7)
    "dinov3_vitb16_kdL3on_all_cm59_ep12": dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, EPOCHS=12, CMID=(5, 9),
                                            KD=dict(TEACHERS=["dinov3_vitl16"], LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True)),
    "dinov3_vitb16_kdL3on_all_cm37_ep12": dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, EPOCHS=12, CMID=(3, 7),
                                            KD=dict(TEACHERS=["dinov3_vitl16"], LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True)),
    # chỉ để chạy thử đường KD nhiều teacher khác kích thước (E26) trên CPU
    "dbg_t_small":   dict(ARCH="dinov2_vits14_reg", IMG=(56, 28), LR=1e-4, LLRD=0.75, DROP_PATH=0.0, HEAD_LR=1e-3, GRAD_CLIP=5.0),
    "dbg_t_clip":    dict(ARCH="clip_vitb16", IMG=(64, 32), LR=3e-5, LLRD=0.70, DROP_PATH=0.0, HEAD_LR=1e-3, GRAD_CLIP=1.0),
    "dbg_h7":        dict(ARCH="dinov2_vits14_reg", IMG=(56, 28), LR=1e-4, LLRD=0.75, DROP_PATH=0.0, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                          KD=dict(TEACHERS=["dbg_t_small", "dbg_t_clip"], GROUPS=["HPC", "C"], W=[1.0, 0.5], LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True)),
    "dbg_h7only":    dict(ARCH="dinov2_vits14_reg", IMG=(56, 28), LR=1e-4, LLRD=0.75, DROP_PATH=0.0, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                          KD=dict(TEACHERS=["dbg_t_clip"], ONLY="HP", LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True)),
    # E28 (07/10): teacher DINOv3-L train lâu hơn (12 / 20 epoch thay vì 8) và student công thức 0.7831 học từ teacher đó
    "dinov3_vitl16_ep12":              dict(ARCH="dinov3_vitl16", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=12),
    "dinov3_vitl16_ep20":              dict(ARCH="dinov3_vitl16", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=20),
    "dinov3_vitb16_kdL3e12on_all_ep12": dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, EPOCHS=12,
                                            KD=dict(TEACHERS=["dinov3_vitl16_ep12"], LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True)),
    "dinov3_vitb16_kdL3e20on_all_ep12": dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, EPOCHS=12,
                                            KD=dict(TEACHERS=["dinov3_vitl16_ep20"], LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True)),
    # E27 (08/10): model ảnh-chữ đọc màu theo vùng do DINO khoanh (H8). SRC = DINOv3-B đã fine-tune cùng fold / cùng split (đóng băng);
    # _pmfix = cùng head nhưng mask đều trong khung áo / quần (đối chứng: tách tác dụng của khoanh vùng khỏi tác dụng của head mới)
    "clip_vitb16_160_pm":     dict(ARCH="clip_vitb16", IMG=(160, 80), LR=3e-5, LLRD=0.70, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=1.0, PMASK=dict(SRC="dinov3_vitb16", TAU=1.5)),
    "clip_vitb16_160_pmfix":  dict(ARCH="clip_vitb16", IMG=(160, 80), LR=3e-5, LLRD=0.70, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=1.0, PMASK=dict(FIXED=True)),
    "pe_core_b16_160_pm":     dict(ARCH="pe_core_b16", IMG=(160, 80), LR=3e-5, LLRD=0.70, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=1.0, PMASK=dict(SRC="dinov3_vitb16", TAU=1.5, TAPS=(8,))),
    "pe_core_b16_160_pmfix":  dict(ARCH="pe_core_b16", IMG=(160, 80), LR=3e-5, LLRD=0.70, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=1.0, PMASK=dict(FIXED=True, TAPS=(8,))),
    # E27b: như E26b (L3 + L2 cho 16 thuộc tính không phải màu; màu từ DINOv3-B nhánh màu ×0.7 + model ảnh-chữ ×0.3) nhưng model ảnh-chữ có màu theo vùng
    "dinov3_vitb16_kdL3L2_clipPmC_ep12": dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, EPOCHS=12,
                                            KD=dict(TEACHERS=["dinov3_vitl16", "dinov2_vitl14_reg", "dinov3_vitb16_cmid_chist", "clip_vitb16_160_pm"],
                                                    GROUPS=["HP", "HP", "C", "C"], W=[1.0, 1.0, 0.7, 0.3], LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True)),
    "dinov3_vitb16_kdL3L2_pePmC_ep12":   dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, EPOCHS=12,
                                            KD=dict(TEACHERS=["dinov3_vitl16", "dinov2_vitl14_reg", "dinov3_vitb16_cmid_chist", "pe_core_b16_160_pm"],
                                                    GROUPS=["HP", "HP", "C", "C"], W=[1.0, 1.0, 0.7, 0.3], LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True)),
    # chỉ để chạy thử E27 trên CPU
    "dbg_clip_pm":   dict(ARCH="clip_vitb16", IMG=(64, 32), LR=3e-5, LLRD=0.70, DROP_PATH=0.0, HEAD_LR=1e-3, GRAD_CLIP=1.0, PMASK=dict(SRC="dbg_t_small", TAU=1.5)),
    "dbg_pe_pm":     dict(ARCH="pe_core_b16", IMG=(64, 32), LR=3e-5, LLRD=0.70, DROP_PATH=0.0, HEAD_LR=1e-3, GRAD_CLIP=1.0, PMASK=dict(SRC="dbg_t_small", TAU=1.5, TAPS=(8,))),
    "dbg_pe_pmfix":  dict(ARCH="pe_core_b16", IMG=(64, 32), LR=3e-5, LLRD=0.70, DROP_PATH=0.0, HEAD_LR=1e-3, GRAD_CLIP=1.0, PMASK=dict(FIXED=True, TAPS=(8,))),
    "dbg_kd_pm":     dict(ARCH="dinov2_vits14_reg", IMG=(56, 28), LR=1e-4, LLRD=0.75, DROP_PATH=0.0, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                          KD=dict(TEACHERS=["dbg_t_small", "dbg_clip_pm"], GROUPS=["HPC", "C"], W=[1.0, 0.5], LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True)),
    "dinov3_vitb16_kdL3on_all_ep20":   dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, EPOCHS=20, KD=dict(TEACHERS=["dinov3_vitl16"], LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True)),
    "dinov3_vitb16_kdL3on_featR_all_ep12": dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, EPOCHS=12,
                                         KD=dict(TEACHERS=["dinov3_vitl16"], LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True, FEAT=1.0, FEAT_REG=1.0)),
    "dinov3_vitb16_kdL3on_featR_all_cc_ep12": dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, EPOCHS=12, CMID=(3, 7), CHIST=True,
                                         KD=dict(TEACHERS=["dinov3_vitl16"], LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True, FEAT=1.0, FEAT_REG=1.0)),
    # ---- E19 (06/10): giữ đặc trưng pretrain cho biến thể tốt nhất E18 (OOD bão hoà từ epoch 4-5, in-domain vẫn tăng) ----
    "dinov3_vitb16_best_lpft":  dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, EPOCHS=13, LPFT=1, CMID=(3, 7), CHIST=True, KD=dict(TEACHERS=["dinov3_vitl16"], LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True, FEAT=1.0, FEAT_REG=1.0)),   # 1 epoch chỉ head rồi 12 epoch fine-tune
    "dinov3_vitb16_best_frz4":  dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, EPOCHS=12, FREEZE=4, CMID=(3, 7), CHIST=True, KD=dict(TEACHERS=["dinov3_vitl16"], LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True, FEAT=1.0, FEAT_REG=1.0)), # đóng băng embedding + 4 khối đầu (nhánh màu đọc khối 4, 8)
    "dinov3_vitb16_dbgfrz":     dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                                     EPOCHS=2, LPFT=1, FREEZE=4, CMID=(3, 7), CHIST=True),
    "dinov3_vitb16_dbg_kd3on":      dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                                         KD=dict(TEACHERS=["dinov3_vitb16"], LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True, FEAT=1.0, FEAT_REG=1.0)),
    "dinov3_vitb16_dbg_kd":         dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                                         CMID=(3, 7), CHIST=True, KD=dict(TEACHERS=["dinov2_vits14_reg"], LAMBDA=0.5, T=1.0)),
    "dinov3_vitb16_qdec":       dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, QDEC=True),
    # ---- E24 (07/10): augmentation theo nhóm + đổi màu có đổi nhãn. E24a: không KD (sàng lọc), spC / spHP = "teacher" chuyên nhóm
    # (cùng các chính sách của gaug nhưng chỉ học nhóm của mình) để so "chuyên gia riêng" với "một model nhiều chính sách". E24b: công thức 0.7831 ----
    "dinov3_vitb16_ep12":          dict(_V3B),
    "dinov3_vitb16_gaug_ep12":     dict(_V3B, GAUG=_GAUG),
    "dinov3_vitb16_gaugnm_ep12":   dict(_V3B, GAUG=_GAUG_NOMASK),
    "dinov3_vitb16_rc_ep12":       dict(_V3B, RECOLOR=_RC),
    "dinov3_vitb16_gaug_rc_ep12":  dict(_V3B, GAUG=_GAUG, RECOLOR=_RC),
    "dinov3_vitb16_spC_ep12":      dict(_V3B, GAUG=[dict(P=0.45, LOSS="C"), dict(P=0.3, AUG=_AUG_CAM, LOSS="C"), dict(P=0.25, AUG=_AUG_DEG, LOSS="C")], RECOLOR=_RC),
    "dinov3_vitb16_spHP_ep12":     dict(_V3B, GAUG=[dict(P=0.4, LOSS="HP"), dict(P=0.25, AUG=_AUG_CAM, LOSS="HP"), dict(P=0.15, AUG=_AUG_DEG, LOSS="P"), dict(P=0.2, AUG=_AUG_PHOTO, LOSS="HP")]),
    "dinov3_vitb16_kdL3on_all_gaug_ep12":    dict(_V3B, GAUG=_GAUG, KD=_KDL3),
    "dinov3_vitb16_kdL3on_all_rc_ep12":      dict(_V3B, RECOLOR=_RC, KD=_KDL3),
    "dinov3_vitb16_kdL3on_all_gaug_rc_ep12": dict(_V3B, GAUG=_GAUG, RECOLOR=_RC, KD=_KDL3),
    # E24c (sau E24a): teacher L chuyên nhóm, student nhận màu từ teacher màu, 16 thuộc tính còn lại từ teacher kia
    "dinov3_vitl16_spC":           dict(ARCH="dinov3_vitl16", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64,
                                        GAUG=[dict(P=0.45, LOSS="C"), dict(P=0.3, AUG=_AUG_CAM, LOSS="C"), dict(P=0.25, AUG=_AUG_DEG, LOSS="C")], RECOLOR=_RC),
    "dinov3_vitl16_spHP":          dict(ARCH="dinov3_vitl16", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64,
                                        GAUG=[dict(P=0.4, LOSS="HP"), dict(P=0.25, AUG=_AUG_CAM, LOSS="HP"), dict(P=0.15, AUG=_AUG_DEG, LOSS="P"), dict(P=0.2, AUG=_AUG_PHOTO, LOSS="HP")]),
    "dinov3_vitb16_kd2sp_ep12":    dict(_V3B, GAUG=_GAUG, RECOLOR=_RC, KD=dict(_KDL3, TEACHERS=["dinov3_vitl16_spC", "dinov3_vitl16_spHP"], GROUPS=["C", "HP"])),
    "dinov3_vitb16_kdC3_ep12":     dict(_V3B, GAUG=_GAUG, RECOLOR=_RC, KD=dict(_KDL3, TEACHERS=["dinov3_vitl16_spC", "dinov3_vitl16"], GROUPS=["C", "HP"])),
    # ---- biến thể nhỏ chỉ để chạy thử code trên CPU ----
    "dinov2_vits14_reg_dbg_all": dict(_DINO_S, DS_BIAS=True, WB=True, RFS_T=0.03, GROUP_CE=0.5, AUG=dict(WB_P=0.5, MOTION_P=0.3, CROP_P=0.5)),
    "dinov2_vits14_reg_dbg_kd":  dict(_DINO_S, KD=dict(TEACHERS=["dinov2_vits14_reg"], LAMBDA=0.5)),
    "dinov2_vits14_reg_dbg_kdwq": dict(_DINO_S, WBCE_POW=0.5, QDEC=True, KD=dict(TEACHERS=["dinov2_vits14_reg"], LAMBDA=0.5, T=1.0)),
    "dinov2_vits14_reg_dbg_kdon": dict(_DINO_S, KD=dict(TEACHERS=["dinov2_vits14_reg"], LAMBDA=0.5, T=2.0, ONLINE=True, FEAT=1.0)),
    "dinov2_vits14_reg_dbg_kdonR": dict(_DINO_S, KD=dict(TEACHERS=["dinov2_vits14_reg"], LAMBDA=0.5, T=2.0, ONLINE=True, FEAT=1.0, FEAT_REG=1.0)),
    "dinov2_vits14_reg_dbg_color": dict(_DINO_S, CMID=(3, 7), CHIST=True),
    "dinov2_vits14_reg_dbg_kdcmid": dict(_DINO_S, CMID=(3, 7), KD=dict(TEACHERS=["dinov2_vits14_reg"], LAMBDA=0.5, T=1.0)),
    "dinov2_vits14_reg_dbg_frz":   dict(_DINO_S, FREEZE=6, LPFT=1, EPOCHS=2),
    "dinov2_vits14_reg_dbg_gaug":  dict(_DINO_S, GAUG=_GAUG, RECOLOR=dict(_RC, P=1.0)),
    "dinov2_vits14_reg_dbg_gkd":   dict(_DINO_S, GAUG=_GAUG, RECOLOR=_RC, KD=dict(_KDL3, TEACHERS=["dinov2_vits14_reg_dbg_gaug", "dinov2_vits14_reg"], GROUPS=["C", "HP"])),
    "dinov2_vits14_reg_dbg_seed2": dict(_DINO_S, SEED=2, BATCH=64),
    "dinov2_vits14_reg_dbg_e8":  dict(_DINO_S, CFIX=dict(PETA_OTHER_NEG=True), KD=dict(TEACHERS=["dinov2_vits14_reg"], MODE="fill", LAMBDA=1.0), COLOR_CE=0.5),
    "dinov2_vits14_reg_dbg_e3":  dict(_DINO_S, COLOR_CE=0.5, SOFTF1=0.5, MIXSTYLE=dict(P=1.0, ALPHA=0.1, BLOCKS=[1, 3, 5]), QDEC=True,
                                      COLOR_SOFT=0.15, WBCE_POW=0.5, DS_BIAS=True),
    "siglip2_vitb16":    dict(IMG=(256, 128), LR=3e-5, LLRD=0.70, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=1.0),
    "clip_vitb16":       dict(IMG=(256, 128), LR=3e-5, LLRD=0.70, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=1.0),
    "dinov2_vitb14_reg": dict(IMG=(252, 126), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0),
    # Nặng hơn ~3.5 lần khi suy luận: chỉ dùng nếu chắc chắn server có GPU
    "clip_vitl14":       dict(IMG=(252, 126), LR=2e-5, LLRD=0.80, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=1.0, BATCH=64),
    # 09/10 (GPU): họ ảnh-chữ cỡ L làm thành viên ghép khác họ DINO (mạnh ở màu). PE-Core-L Apache-2.0, khoá khớp timm vit_pe_core_large_patch14_336
    "pe_core_l14":       dict(IMG=(252, 126), LR=2e-5, LLRD=0.80, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=1.0, BATCH=64),
    # E6 (K17) teacher: batch 64 cho vừa bộ nhớ T4 (DataParallel 32 ảnh/GPU); LLRD 0.85^25 ≈ 0.017 tương đương B 0.75^13 ≈ 0.024
    "dinov2_vitl14_reg": dict(IMG=(252, 126), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64),
    # 17/10: = dinov2_vitl14_reg (thành viên L2 của v17c) + LP-FT (epoch đầu chỉ train đầu phân loại) — LODO giữ PETA: LP-FT sửa méo đặc trưng của H+
    "dinov2_vitl14_reg_lpft": dict(ARCH="dinov2_vitl14_reg", IMG=(252, 126), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, LPFT=1, SAVE_EP_LOGITS=True),
    # 17/10 đợt Kaggle (khai báo trước, docs/KE_HOACH_1017.md mục 10): biến thể LP-FT cho thành viên L2
    "dinov2_vitl14_reg_lpft_s2":  dict(ARCH="dinov2_vitl14_reg", IMG=(252, 126), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, LPFT=1, SAVE_EP_LOGITS=True, SEED=2),   # seed 2 -> soup
    "dinov2_vitl14_reg_lpft2":    dict(ARCH="dinov2_vitl14_reg", IMG=(252, 126), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, LPFT=2, SAVE_EP_LOGITS=True),           # 2 epoch LP
    "dinov2_vitl14_reg_lpft_bg":  dict(ARCH="dinov2_vitl14_reg", IMG=(252, 126), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, LPFT=1, SAVE_EP_LOGITS=True,
                                       BGAUG=dict(P=0.4, W=(1, 1, 1))),   # + tăng cường nền (chẩn đoán 17/10: thay nền xám tụt 0.011–0.033)
    "dinov2_vitl14_reg_lpft_224": dict(ARCH="dinov2_vitl14_reg", IMG=(224, 112), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, LPFT=1, SAVE_EP_LOGITS=True),           # train đúng độ phân giải chạy
    # 17/10 đóng băng ngẫu nhiên (kiểu LISA): pha fine-tune, mỗi khối bị đóng băng với xác suất P, bốc lại sau EVERY bước (0 = đầu mỗi epoch)
    "dinov2_vitl14_reg_lpft_rf":   dict(ARCH="dinov2_vitl14_reg", IMG=(252, 126), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, LPFT=1, SAVE_EP_LOGITS=True,
                                         RFREEZE=dict(P=0.5, EVERY=50)),
    "dinov2_vitl14_reg_lpft_rfep": dict(ARCH="dinov2_vitl14_reg", IMG=(252, 126), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, LPFT=1, SAVE_EP_LOGITS=True,
                                         RFREEZE=dict(P=0.5, EVERY=0)),
    # 18/10 chưng cất RADIO-v4-H (model PAR đã fine-tune; thêm RADIO vào đuôi là nguồn phần lợi Synth trên test) vào thành viên L2: KD online nhãn mềm
    # mọi thuộc tính (ALL, như bản 0.7831) + khớp đặc trưng (FEAT: kdLon_feat là biến thể E11 duy nhất hơn mốc có ý nghĩa trên LODO). Cặp nộp vẫn F1 + L2
    # -> thời gian chấm không đổi. T_FP16: ma trận teacher fp16 (RADIO-H 2.6 -> 1.3 GB mỗi GPU). Khai báo cổng: docs/KE_HOACH_1017.md mục 13
    "dinov2_vitl14_reg_lpft_kdr":  dict(ARCH="dinov2_vitl14_reg", IMG=(252, 126), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, LPFT=1, SAVE_EP_LOGITS=True,
                                         KD=dict(TEACHERS=["radio_v4h"], LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True, FEAT=1.0, T_FP16=True)),
    # 08/10 (GPU): model L nộp thẳng. Ước tính A10: 288x144 ~159 s, 320x160 ~194 s (mốc 256x128 = 126 s đo thật)
    "dinov3_vitl16_288":  dict(ARCH="dinov3_vitl16", IMG=(288, 144), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64),
    "dinov3_vitl16_320":  dict(ARCH="dinov3_vitl16", IMG=(320, 160), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64),
    "dinov3_vitl16_s2":   dict(ARCH="dinov3_vitl16", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, SEED=2),   # ghép cùng họ (dùng chung ảnh 256x128)
    # 09/10: model chuyên vùng đầu (probe: ghép cả người + đầu phóng to tăng kính +0.033, mũ +0.016, hói +0.023 AUC; tuổi / giới không lợi).
    # Cắt 32 % trên, 10-90 % ngang của ảnh gốc rồi phóng lên ảnh vuông; chỉ học nhóm H (kính / mũ / tóc); lấy epoch cuối.
    # Chỉ dùng làm thành viên thứ 3 của hệ ghép với trọng số riêng nhóm H (research/make_gpu_ensemble_zip.py "dir:H@α"), không nộp riêng.
    "dinov3_vitb16_head": dict(ARCH="dinov3_vitb16", IMG=(192, 192), CROP=(0.10, 0.0, 0.90, 0.32), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                               GAUG=[dict(P=1.0, LOSS="H")], LAST=True),
    "dinov2_vitb14_head": dict(ARCH="dinov2_vitb14_reg", IMG=(196, 196), CROP=(0.10, 0.0, 0.90, 0.32), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                               GAUG=[dict(P=1.0, LOSS="H")], LAST=True),
    "dbg_head_s":         dict(ARCH="dinov2_vits14_reg", IMG=(98, 98), CROP=(0.10, 0.0, 0.90, 0.32), LR=1e-4, LLRD=0.75, DROP_PATH=0.05, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                               GAUG=[dict(P=1.0, LOSS="H")], LAST=True),
    # 09/10: DINOv3-L học nhãn mềm = trung bình logit 4 họ (L3, DINOv2-L, PE-L, CLIP-L; trên val mới cross-fit M+PA 0.9078 > L3 0.9027),
    # tính trước trên ảnh train (offline, mỗi teacher đúng kích thước / chuẩn hoá của nó) -> khi nộp chỉ tốn thời gian của 1 model L
    "dinov3_vitl16_kd4":  dict(ARCH="dinov3_vitl16", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64,
                               KD=dict(TEACHERS=["dinov3_vitl16", "dinov2_vitl14_reg", "pe_core_l14", "clip_vitl14"], LAMBDA=0.5, T=1.0, ALL=True)),
    # DINOv3-L học thêm từ trung bình logit DINOv3-L + DINOv2-L (trên val mới cặp này AUC 0.9723 > L3 0.9688), teacher chạy online trên ảnh đã augment
    # 10/10: DINOv3 ViT-H+/16 (840M, 32 khối, SwiGLU; weights timm không gated). Giữ công thức L3 (LR 3e-5, LLRD 0.85, DP 0.15, lô 64, CLS + TB patch);
    # T4 16 GB không đủ cho fine-tune toàn bộ -> đóng băng khối dưới (LLRD 0.85: khối 16 vốn chỉ học 0.85^17 ≈ 6 % LR), khối đóng băng giữ fp16,
    # EMA dùng chung phần đóng băng. AUTO_FIT thử (FREEZE, gradient checkpointing) theo thứ tự, lấy cấu hình đầu tiên vừa bộ nhớ;
    # AUTO_EP: số epoch ≤ 8 sao cho xong trước TRAIN_BUDGET_H. Ước tính A10: ~6.8 ms/ảnh ở 256x128 (2.8× L3) -> nộp đơn ~222 s.
    "dinov3_vith16p":     dict(IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8,
                               FREEZE=16, FROZEN_FP16=True, AUTO_FIT=[(16, False), (16, True), (20, False), (20, True), (24, True)], AUTO_EP=True,
                               EXTRA_VAL=[(224, 112)], FP16_CHECK=True),
    # 17/10: H+ công thức g11 (T4: AUTO_FIT -> FREEZE 20) + LP-FT -> bạn soup cho F1 (cùng pretrain, cùng đầu LP)
    "dinov3_vith16p_lpft": dict(ARCH="dinov3_vith16p", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8,
                               FREEZE=16, FROZEN_FP16=True, AUTO_FIT=[(16, False), (16, True), (20, False), (20, True), (24, True)], AUTO_EP=True,
                               EXTRA_VAL=[(224, 112)], FP16_CHECK=True, SAVE_EP_LOGITS=True, LPFT=1),
    # 11/10 G13: C-RADIOv4-H (631M, ViT-H/16 + CPE, chưng cất SigLIP2-g + DINOv3-7B + SAM3; NVIDIA Open Model License). Cùng công thức H+;
    # nhẹ hơn H+ 25 % (MLP GELU thay SwiGLU) nên AUTO_FIT thử đóng băng ít hơn. Ước tính A10 ~4.7 ms/ảnh ở 224x112, ~6 ms ở 256x128 (≈ H+ 224).
    "radio_v4h":          dict(IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8,
                               FREEZE=12, FROZEN_FP16=True, AUTO_FIT=[(8, False), (8, True), (12, False), (12, True), (16, True), (20, True), (24, True)],
                               AUTO_EP=True, EXTRA_VAL=[(224, 112)], FP16_CHECK=True),
    "dbg_radio":          dict(ARCH="radio_dbg", IMG=(64, 32), LR=1e-4, LLRD=0.85, DROP_PATH=0.1, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                               FREEZE=1, FROZEN_FP16=True, CKPT=True, EXTRA_VAL=[(48, 32)], FP16_CHECK=True),
    # 11/10 E30: cùng công thức, train trên nhãn train đã làm sạch bằng dự đoán ngoài-fold (dataset upar-resplitclean-*, ANN_PREFER "resplitclean").
    # Tên riêng (_clean) để không bao giờ dùng lại nhầm run nhãn gốc cùng tên trong Input. Val giữ nhãn gốc.
    # L3: LAST=True để A/B đúng công thức với v11 (teacher L3 resplit giữ epoch cuối); H+ / L2 của v15 chọn epoch tốt nhất theo val -> giữ như cũ
    "dinov3_vitl16_clean":   dict(ARCH="dinov3_vitl16", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, LAST=True),
    "dinov3_vitl16_cleanconv": dict(ARCH="dinov3_vitl16", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, LAST=True),  # E30 + E30b
    # 11/10 E32: L3 như v11 (giữ epoch cuối) + bỏ loss tuổi ở ảnh đồng phục PA100k / PETA-CUHK -> A/B với v11 trên test (Young / Adult / Old)
    "dinov3_vitl16_agemask":   dict(ARCH="dinov3_vitl16", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, LAST=True, AGEMASK=True),
    "dbg_agemask":             dict(ARCH="radio_dbg", IMG=(64, 32), LR=1e-4, LLRD=0.85, DROP_PATH=0.1, HEAD_LR=1e-3, GRAD_CLIP=5.0, AGEMASK=True),
    # 18/10 CMASK: 4 thành viên bộ ghép (f8 / L2 / g11 / RADIO) đúng công thức cũ + bỏ khỏi loss các ô OOF L3 nghi nhãn sai (cờ E30 bản B), GIỮ nhãn gốc.
    # Cần Input dataset upar-cmask-1018 (cmask_flags.npz, research/make_cmask.py). Tên riêng (_cmask) để không dùng lại nhầm run nhãn gốc trong Input.
    "dinov2_vitl14_reg_cmask": dict(ARCH="dinov2_vitl14_reg", IMG=(252, 126), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64,
                                    EXTRA_VAL=[(224, 112)], CMASK=True),
    "dinov3_vith16p_cmask":    dict(ARCH="dinov3_vith16p", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8,
                                    FREEZE=16, FROZEN_FP16=True, AUTO_FIT=[(16, False), (16, True), (20, False), (20, True), (24, True)], AUTO_EP=True,
                                    EXTRA_VAL=[(224, 112)], FP16_CHECK=True, CMASK=True),
    "radio_v4h_cmask":         dict(ARCH="radio_v4h", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8,
                                    FREEZE=12, FROZEN_FP16=True, AUTO_FIT=[(8, False), (8, True), (12, False), (12, True), (16, True), (20, True), (24, True)],
                                    AUTO_EP=True, EXTRA_VAL=[(224, 112)], FP16_CHECK=True, CMASK=True),
    "dinov3_vith16p_f8_cmask": dict(ARCH="dinov3_vith16p", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8,
                                    FREEZE=8, FROZEN_FP16=True, AUTO_FIT=[(8, False), (8, True), (12, False), (12, True), (16, True), (20, False)], AUTO_EP=True,
                                    EXTRA_VAL=[(224, 112)], FP16_CHECK=True, SAVE_EP_LOGITS=True, SAVE_EP_EXTRA=(4, 5, 6, 7), CMASK=True),
    "dbg_cmask":               dict(ARCH="radio_dbg", IMG=(64, 32), LR=1e-4, LLRD=0.85, DROP_PATH=0.1, HEAD_LR=1e-3, GRAD_CLIP=5.0, CMASK=True),
    # 11/10 E33: lấy mẫu lại khi train (góp ý: "mỗi batch chia đều mẫu"). Mặc định hiện tại: WeightedRandomSampler ∝ n_dataset^-0.5 + wBCE.
    # ds0 / ds1: lấy mẫu ∝ n_dataset^0 (tự nhiên) / ^-1 (đều theo dataset, kỳ vọng); dseq: MỖI batch đúng 1/3 mỗi dataset;
    # cas: mỗi batch 40 thuộc tính bằng nhau (class-aware); cassq: tỷ lệ ∝ căn số ảnh dương. cas* tắt wBCE (WBCE_POW 0) để không bù hai lần;
    # caswb giữ wBCE (đối chứng bù hai lần). rfs: lặp ảnh thuộc tính hiếm (K4, E2a trên DINOv2-B +0.0008) trên DINOv3-B để so cùng nền.
    # LODO: nền dinov3_vitb16 (8 epoch, mốc E14 fold PA100k / E15 fold Market, PETA). Nộp: L3 như v11 (epoch cuối) + sampler.
    "dinov3_vitb16_ds0":   dict(_V3B, EPOCHS=8, DS_BAL=0.0),
    "dinov3_vitb16_ds1":   dict(_V3B, EPOCHS=8, DS_BAL=1.0),
    "dinov3_vitb16_dseq":  dict(_V3B, EPOCHS=8, SAMPLER=dict(MODE="ds")),
    "dinov3_vitb16_cas":   dict(_V3B, EPOCHS=8, SAMPLER=dict(MODE="attr", POW=0.0, MAX_REP=20), WBCE_POW=0.0),
    "dinov3_vitb16_cassq": dict(_V3B, EPOCHS=8, SAMPLER=dict(MODE="attr", POW=0.5, MAX_REP=20), WBCE_POW=0.0),
    "dinov3_vitb16_caswb": dict(_V3B, EPOCHS=8, SAMPLER=dict(MODE="attr", POW=0.0, MAX_REP=20)),
    "dinov3_vitb16_rfs":   dict(_V3B, EPOCHS=8, RFS_T=0.03),
    "dinov3_vitl16_dseq":  dict(ARCH="dinov3_vitl16", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, LAST=True,
                                SAMPLER=dict(MODE="ds")),
    "dinov3_vitl16_cas":   dict(ARCH="dinov3_vitl16", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, LAST=True,
                                SAMPLER=dict(MODE="attr", POW=0.0, MAX_REP=20), WBCE_POW=0.0),
    "dinov3_vitl16_cassq": dict(ARCH="dinov3_vitl16", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, LAST=True,
                                SAMPLER=dict(MODE="attr", POW=0.5, MAX_REP=20), WBCE_POW=0.0),
    "dbg_dseq":            dict(_DINO_S, SAMPLER=dict(MODE="ds")),
    "dbg_cas":             dict(_DINO_S, SAMPLER=dict(MODE="attr", POW=0.0, MAX_REP=20), WBCE_POW=0.0),
    "dinov2_vitl14_reg_clean": dict(ARCH="dinov2_vitl14_reg", IMG=(252, 126), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64),
    # 12/10: bản B + trộn màu E30b cho bộ ghép kiểu v17 (test L3: trộn màu MEVID +0.0018, iF1 màu +0.009) — cấu hình giống bản _clean, tên riêng
    "dinov2_vitl14_reg_cleanconv": dict(ARCH="dinov2_vitl14_reg", IMG=(252, 126), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64),
    "radio_v4h_cleanconv":  dict(ARCH="radio_v4h", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8,
                                 FREEZE=12, FROZEN_FP16=True, AUTO_FIT=[(8, False), (8, True), (12, False), (12, True), (16, True), (20, True), (24, True)],
                                 AUTO_EP=True, EXTRA_VAL=[(224, 112)], FP16_CHECK=True),
    "dinov3_vith16p_cleanconv": dict(ARCH="dinov3_vith16p", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8,
                                 FREEZE=16, FROZEN_FP16=True, AUTO_FIT=[(16, False), (16, True), (20, False), (20, True), (24, True)], AUTO_EP=True,
                                 EXTRA_VAL=[(224, 112)], FP16_CHECK=True),
    # 12/10: RADIO trên nhãn sạch (cặp fold không rò PETA) -> đủ bộ H+ / L2 / RADIO như v17 nếu nhãn sạch thắng trên test
    "radio_v4h_clean":    dict(ARCH="radio_v4h", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8,
                               FREEZE=12, FROZEN_FP16=True, AUTO_FIT=[(8, False), (8, True), (12, False), (12, True), (16, True), (20, True), (24, True)],
                               AUTO_EP=True, EXTRA_VAL=[(224, 112)], FP16_CHECK=True),
    "dinov3_vith16p_clean":  dict(ARCH="dinov3_vith16p", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8,
                                  FREEZE=16, FROZEN_FP16=True, AUTO_FIT=[(16, False), (16, True), (20, False), (20, True), (24, True)], AUTO_EP=True,
                                  EXTRA_VAL=[(224, 112)], FP16_CHECK=True),
    "dbg_clean":             dict(ARCH="radio_dbg", IMG=(64, 32), LR=1e-4, LLRD=0.85, DROP_PATH=0.1, HEAD_LR=1e-3, GRAD_CLIP=5.0),
    # 12/10 E34 (dữ liệu hiếm, LODO DINOv3-B 8 epoch như mốc dinov3_vitb16): rc = E24 đổi màu có đổi nhãn (chưa từng chạy), hswap = ghép nửa
    # trên / nửa dưới ưu tiên màu quần hiếm (mới); asl / aslw (E29, chưa từng chạy) có sẵn ở trên.
    "dinov3_vitb16_rc":      dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, RECOLOR=_RC),
    "dinov3_vitb16_hswap":   dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                                  HSWAP=dict(P=0.3, POW=0.5, CUT=(0.47, 0.55))),
    # 12/10 E36: L3 như v11 (epoch cuối) + loss nhất quán giữa 2 ảnh cùng người khác camera (Market), 8 cặp / batch 64, trọng số 1
    "dinov3_vitl16_idpair":  dict(ARCH="dinov3_vitl16", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, LAST=True,
                                  IDPAIR=dict(K=8, W=1.0)),
    "dbg_idpair":            dict(ARCH="radio_dbg", IMG=(64, 32), LR=1e-4, LLRD=0.85, DROP_PATH=0.1, HEAD_LR=1e-3, GRAD_CLIP=5.0, IDPAIR=dict(K=2, W=1.0)),
    # 13/10 T2: nhãn mềm từ teacher nhiều góc nhìn (kaggle_results/run_analysis2_1013, dataset upar-t2-teacher-1013): ảnh Market train có người ở ≥ 2 camera
    # học y = (1 − LAM)·GT + LAM·P1, P1 = TB (cân theo camera) sigmoid logit OOF E30 trên các camera KHÁC của cùng người. Chỉ đổi nhãn train, val giữ nguyên.
    # ATTRS / EXCLUDE: tên hoặc tiền tố thuộc tính (ATTR chưa có ở đây). Kính / tóc ở Market gán theo từng ảnh -> không áp teacher.
    # t2a: LAM 0.5 mọi thuộc tính trừ kính / tóc; t2b: LAM 1.0 chỉ nhóm "bền" theo người. Công thức L3 như v11 / E36 (epoch cuối), bỏ IDPAIR.
    "dinov3_vitl16_t2a":     dict(ARCH="dinov3_vitl16", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, LAST=True,
                                  T2=dict(_T2_FILES, LAM=0.5, ATTRS=None, EXCLUDE=_T2_EXCL)),
    "dinov3_vitl16_t2b":     dict(ARCH="dinov3_vitl16", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, LAST=True,
                                  T2=dict(_T2_FILES, LAM=1.0, ATTRS=_T2_PERSIST, EXCLUDE=_T2_EXCL)),
    # H+ đúng công thức thành viên v17 (dinov3_vith16p: 8 epoch, chọn epoch tốt nhất theo val — run g11 tốt nhất = epoch cuối), AUTO_FIT bắt đầu từ
    # cấu hình g11 đã chọn (FREEZE 20, không checkpointing; FREEZE 16 tràn bộ nhớ) để khỏi tốn thời gian thử
    "dinov3_vith16p_t2a":    dict(ARCH="dinov3_vith16p", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8,
                                  FREEZE=20, FROZEN_FP16=True, AUTO_FIT=[(20, False), (20, True), (24, True)], AUTO_EP=True,
                                  EXTRA_VAL=[(224, 112)], FP16_CHECK=True, T2=dict(_T2_FILES, LAM=0.5, ATTRS=None, EXCLUDE=_T2_EXCL)),
    # 13/10 M2: bản sinh đôi null của dinov3_vith16p_t2a (N3) — y hệt (cùng seed mặc định, cùng AUTO_FIT, chọn epoch theo MEAN, không LAST), chỉ bỏ T2.
    # N3 − twin = hiệu ứng T2 gần cặp; twin − g11 = mẫu nhiễu đường ngẫu nhiên H+ đầu tiên. Lưu logit val từng epoch (256) + 224x112 ở ep5/6/7.
    "dinov3_vith16p_t2null": dict(ARCH="dinov3_vith16p", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8,
                                  FREEZE=20, FROZEN_FP16=True, AUTO_FIT=[(20, False), (20, True), (24, True)], AUTO_EP=True,
                                  EXTRA_VAL=[(224, 112)], FP16_CHECK=True, SAVE_EP_LOGITS=True, SAVE_EP_EXTRA=(5, 6, 7)),
    # 13/10 A100 (Colab, docs/KE_HOACH_A100_COLAB_1013.md): = dinov3_vith16p_t2a, chỉ đổi FREEZE 20 -> 8 (T4 16 GB bắt đóng băng 20/32 khối;
    # với LLRD 0.85 khối 9..20 có LR 0.02..0.12 x LR đỉnh nên mở chúng mới có nghĩa, khối 1..8 < 0.03 -> vẫn đóng). _s2: sinh đôi seed 2 đo nhiễu.
    # SNAP_EP 4..6 để bản FULL lấy đúng epoch tốt nhất của run train-only (t2a: epoch 5).
    "dinov3_vith16p_t2a_f8":    dict(ARCH="dinov3_vith16p", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8,
                                     FREEZE=8, FROZEN_FP16=True, AUTO_FIT=[(8, False), (8, True), (12, False), (12, True), (16, True), (20, False)], AUTO_EP=True,
                                     EXTRA_VAL=[(224, 112)], FP16_CHECK=True, SAVE_EP_LOGITS=True, SAVE_EP_EXTRA=(4, 5, 6, 7),
                                     T2=dict(_T2_FILES, LAM=0.5, ATTRS=None, EXCLUDE=_T2_EXCL)),
    # smoke Colab CPU (SMOKE trong upar_a100_colab.ipynb): cùng lớp DinoV3Eva với H+, nhỏ; đi đủ nhánh T2 / FROZEN_FP16 / EXTRA_VAL / SAVE_EP / SNAP_EP
    "dbg_t2a_f8":               dict(ARCH="dinov3_dbgeva", IMG=(64, 32), LR=1e-4, LLRD=0.85, DROP_PATH=0.1, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=32,
                                     FREEZE=1, FROZEN_FP16=True, AUTO_FIT=[(1, False), (1, True)], AUTO_EP=True, EXTRA_VAL=[(48, 32)], FP16_CHECK=True,
                                     SAVE_EP_LOGITS=True, SAVE_EP_EXTRA=(0,), T2=dict(_T2_FILES, LAM=0.5, ATTRS=None, EXCLUDE=_T2_EXCL)),
    # 14/10 EVA-02-L (MIM m38m) thay DINOv2-L trong hệ ghép (docs/KE_HOACH_A100_COLAB_1013.md): công thức L2 (LR 3e-5, LLRD 0.85, DP 0.15, lô 64, 8 ep),
    # train thẳng 224x112 (= độ phân giải lúc chấm, không nội suy), fine-tune toàn bộ (A100). Lưu logit val từng epoch.
    # 14/10 H+ ba tầng độ phân giải (người dùng đề xuất): như thành viên v17 (g11: 256x128, FREEZE 20, 8 ep, chọn epoch theo val), mỗi lô chọn 1 trong
    # 256 / 224 / 192 (lúc chấm: 224x112 cho phần lớn ảnh, 192x96 cho 3000 ảnh đuôi v17e). Logit val thêm 224 + 192 (từng epoch 5-7).
    "dinov3_vith16p_mres":      dict(ARCH="dinov3_vith16p", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8,
                                     FREEZE=20, FROZEN_FP16=True, AUTO_FIT=[(20, False), (20, True), (24, True)], AUTO_EP=True,
                                     EXTRA_VAL=[(224, 112), (192, 96)], FP16_CHECK=True, SAVE_EP_LOGITS=True, SAVE_EP_EXTRA=(5, 6, 7),
                                     MRES=dict(SIZES=[(256, 128), (224, 112), (192, 96)])),
    # 14/10: gộp 2 cải tiến cùng họ đều dương nhỏ trên val: FREEZE 8 (A100, +0.0012 ghép) + MRES (+0.0009 chung tier) -> chạy trên A100
    "dinov3_vith16p_f8_mres":   dict(ARCH="dinov3_vith16p", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8,
                                     FREEZE=8, FROZEN_FP16=True, AUTO_FIT=[(8, False), (8, True), (12, False), (12, True), (16, True), (20, False)], AUTO_EP=True,
                                     EXTRA_VAL=[(224, 112), (192, 96)], FP16_CHECK=True, SAVE_EP_LOGITS=True, SAVE_EP_EXTRA=(5, 6, 7),
                                     MRES=dict(SIZES=[(256, 128), (224, 112), (192, 96)])),
    # 15/10 bản sạch: chạy đủ model cho mọi ảnh trong 5 phút buộc hạ độ phân giải (176 / 192) -> H+ FREEZE 8 train đúng các mức đó
    "dinov3_vith16p_f8_mlo":    dict(ARCH="dinov3_vith16p", IMG=(224, 112), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8,
                                     FREEZE=8, FROZEN_FP16=True, AUTO_FIT=[(8, False), (8, True), (12, False), (12, True), (16, True), (20, False)], AUTO_EP=True,
                                     EXTRA_VAL=[(192, 96), (176, 80)], FP16_CHECK=True, SAVE_EP_LOGITS=True, SAVE_EP_EXTRA=(5, 6, 7),
                                     MRES=dict(SIZES=[(224, 112), (192, 96), (176, 80)])),
    # 15/10 bước 2: như _mlo + tăng cường CÂN HAI CHIỀU chất lượng ảnh. Mặc định chỉ làm XẤU ảnh (thu nhỏ p .4, JPEG p .3 q 25–90, mờ p .15);
    # train vốn trải từ ảnh nhỏ nén nặng (Market, PA100k ~2 bit/điểm) tới ảnh ít nén (PETA ~8 bit/điểm) -> bớt làm xấu, thêm làm NÉT, nới
    # tương phản / bão hoà vừa phải (24/40 thuộc tính là màu nên không đổi hue). Chỉ giữ nếu qua cổng val (M3 vs M4, cùng đường run.py).
    "dinov3_vith16p_f8_mlo_q":  dict(ARCH="dinov3_vith16p", IMG=(224, 112), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8,
                                     FREEZE=8, FROZEN_FP16=True, AUTO_FIT=[(8, False), (8, True), (12, False), (12, True), (16, True), (20, False)], AUTO_EP=True,
                                     EXTRA_VAL=[(192, 96), (176, 80)], FP16_CHECK=True, SAVE_EP_LOGITS=True, SAVE_EP_EXTRA=(5, 6, 7),
                                     MRES=dict(SIZES=[(224, 112), (192, 96), (176, 80)]),
                                     AUG=dict(LOWRES_P=0.25, JPEG_P=0.2, JPEG_Q=(40, 95), SHARP_P=0.3, SHARP_F=2.0, CONTRAST=0.35, SAT=0.25)),
    # 15/10 M5: như _mlo (M3) nhưng FINE-TUNE TOÀN BỘ (FREEZE 0) — A100 80 GB còn dư (M3 dùng ~28 GB); xu hướng val: FREEZE 20 (g11) < FREEZE 8 (f8)
    "dinov3_vith16p_f0_mlo":    dict(ARCH="dinov3_vith16p", IMG=(224, 112), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8,
                                     FREEZE=0, FROZEN_FP16=True, AUTO_FIT=[(0, False), (0, True), (4, False), (4, True), (8, False), (8, True)], AUTO_EP=True,
                                     EXTRA_VAL=[(192, 96), (176, 80)], FP16_CHECK=True, SAVE_EP_LOGITS=True, SAVE_EP_EXTRA=(5, 6, 7),
                                     MRES=dict(SIZES=[(224, 112), (192, 96), (176, 80)])),
    "dbg_mres_f0":              dict(ARCH="dinov3_dbgeva", IMG=(64, 32), LR=1e-4, LLRD=0.85, DROP_PATH=0.1, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=32,
                                     FREEZE=0, FROZEN_FP16=True, AUTO_FIT=[(0, False), (0, True)], EXTRA_VAL=[(48, 32), (32, 16)], MRES=dict(SIZES=[(64, 32), (48, 32), (32, 16)])),
    "dbg_bgaug":                dict(ARCH="dinov3_dbgeva", IMG=(64, 32), LR=1e-4, LLRD=0.85, DROP_PATH=0.1, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=32, LPFT=1,
                                     BGAUG=dict(P=1.0, W=(1, 1, 1))),
    "dbg_mres_q":               dict(ARCH="dinov3_dbgeva", IMG=(64, 32), LR=1e-4, LLRD=0.85, DROP_PATH=0.1, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=32,
                                     FREEZE=1, FROZEN_FP16=True, EXTRA_VAL=[(48, 32), (32, 16)], MRES=dict(SIZES=[(64, 32), (48, 32), (32, 16)]),
                                     AUG=dict(LOWRES_P=0.25, JPEG_P=0.2, JPEG_Q=(40, 95), SHARP_P=0.3, SHARP_F=2.0, CONTRAST=0.35, SAT=0.25)),
    "dbg_mres":                 dict(ARCH="dinov3_dbgeva", IMG=(64, 32), LR=1e-4, LLRD=0.85, DROP_PATH=0.1, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=32,
                                     FREEZE=1, FROZEN_FP16=True, EXTRA_VAL=[(48, 32), (32, 16)], MRES=dict(SIZES=[(64, 32), (48, 32), (32, 16)])),
    "eva02_large_m38m":         dict(IMG=(224, 112), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8, SAVE_EP_LOGITS=True),
    "dbg_eva02":                dict(ARCH="eva02_dbg", IMG=(56, 28), LR=1e-4, LLRD=0.85, DROP_PATH=0.1, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=32, SAVE_EP_LOGITS=True),
    "dinov3_vith16p_f8":        dict(ARCH="dinov3_vith16p", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8,
                                     FREEZE=8, FROZEN_FP16=True, AUTO_FIT=[(8, False), (8, True), (12, False), (12, True), (16, True), (20, False)], AUTO_EP=True,
                                     EXTRA_VAL=[(224, 112)], FP16_CHECK=True, SAVE_EP_LOGITS=True, SAVE_EP_EXTRA=(4, 5, 6, 7)),   # = g11 (thành viên v17), FREEZE 8, không T2
    # 16/10 LP-FT (Kumar et al. ICLR'22: fine-tune làm méo đặc trưng pretrain -> kém ngoài miền): epoch đầu chỉ train đầu phân loại, rồi f8 như cũ
    "dinov3_vith16p_f8_lpft":   dict(ARCH="dinov3_vith16p", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8,
                                     FREEZE=8, FROZEN_FP16=True, AUTO_FIT=[(8, False), (8, True), (12, False), (12, True), (16, True), (20, False)], AUTO_EP=True,
                                     EXTRA_VAL=[(224, 112)], FP16_CHECK=True, SAVE_EP_LOGITS=True, SAVE_EP_EXTRA=(4, 5, 6, 7), LPFT=1),
    "dinov3_vith16p_f8_s2":     dict(ARCH="dinov3_vith16p", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8,
                                     FREEZE=8, FROZEN_FP16=True, AUTO_FIT=[(8, False), (8, True), (12, False), (12, True), (16, True), (20, False)], AUTO_EP=True,
                                     EXTRA_VAL=[(224, 112)], FP16_CHECK=True, SAVE_EP_LOGITS=True, SAVE_EP_EXTRA=(4, 5, 6, 7), SEED=2),   # sinh đôi seed 2 của dinov3_vith16p_f8
    "dinov3_vith16p_t2a_f8_s2": dict(ARCH="dinov3_vith16p", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8,
                                     FREEZE=8, FROZEN_FP16=True, AUTO_FIT=[(8, False), (8, True), (12, False), (12, True), (16, True), (20, False)], AUTO_EP=True,
                                     EXTRA_VAL=[(224, 112)], FP16_CHECK=True, SAVE_EP_LOGITS=True, SAVE_EP_EXTRA=(4, 5, 6, 7), SEED=2,
                                     T2=dict(_T2_FILES, LAM=0.5, ATTRS=None, EXCLUDE=_T2_EXCL)),
    # 13/10 E2(mevid): L3 công thức v11 (8 epoch, seed mặc định, không T2 / IDPAIR, epoch cuối) + _AUG_CCTV; lưu logit val từng epoch
    "dinov3_vitl16_cctv":    dict(ARCH="dinov3_vitl16", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, LAST=True,
                                  SAVE_EP_LOGITS=True, AUG=_AUG_CCTV),
    # 13/10 M5: bản sinh đôi seed của L3 v11 (= dinov3_vitl16_s2 + LAST + lưu logit val từng epoch; 8 epoch, không T2 / IDPAIR / AUG riêng)
    # s2l − v11 = mẫu nhiễu seed cỡ L cho cổng G-L3 (N1 / N2 / L3cctv chỉ tính là tín hiệu khi vượt ngưỡng này)
    "dinov3_vitl16_s2l":     dict(ARCH="dinov3_vitl16", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, SEED=2,
                                  LAST=True, SAVE_EP_LOGITS=True),
    # E2 bước 2 (CHỈ đẩy khi L3cctv qua cổng): = dinov3_vith16p_t2null (M2 twin: cùng seed / AUTO_FIT) + _AUG_CCTV + LAST
    "dinov3_vith16p_cctv":   dict(ARCH="dinov3_vith16p", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64, EPOCHS=8,
                                  FREEZE=20, FROZEN_FP16=True, AUTO_FIT=[(20, False), (20, True), (24, True)], AUTO_EP=True,
                                  EXTRA_VAL=[(224, 112)], FP16_CHECK=True, SAVE_EP_LOGITS=True, SAVE_EP_EXTRA=(5, 6, 7), LAST=True, AUG=_AUG_CCTV),
    # 14/10 B (kaggle/KE_HOACH_1014_L3.md): L3 công thức v11 (8 epoch, seed mặc định, epoch cuối, không T2) + logit val từng epoch.
    # b0 đối chứng. b2 + QDEC2: đầu truy vấn riêng (token khối cuối + khối MID, mặc định 2/3 độ sâu; prior hàng đầu / thân trên / thân dưới),
    #   loss = BCE(z) + LAM·BCE(z_q) cùng mask; val_logits.npy / zip vẫn là z, z_q ở val_logits_q*.npy để ghép offline.
    #   (b1 / b3 = DS_RES residual theo dataset đã bỏ: probe A 13/10 không qua, kaggle_results/run_probe_a_1013/)
    "dinov3_vitl16_b0":      dict(ARCH="dinov3_vitl16", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64,
                                  LAST=True, SAVE_EP_LOGITS=True),
    "dinov3_vitl16_b2":      dict(ARCH="dinov3_vitl16", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64,
                                  LAST=True, SAVE_EP_LOGITS=True, QDEC2=dict(DIM=256, HEADS=8, MID=None, LAM=1.0)),
    "dbg_t2":                dict(ARCH="radio_dbg", IMG=(64, 32), LR=1e-4, LLRD=0.85, DROP_PATH=0.1, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                                  T2=dict(_T2_FILES, LAM=0.5, ATTRS=None, EXCLUDE=_T2_EXCL)),
    "dbg_hswap":             dict(ARCH="radio_dbg", IMG=(64, 32), LR=1e-4, LLRD=0.85, DROP_PATH=0.1, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                                  HSWAP=dict(P=0.9, POW=0.5), RECOLOR=dict(_RC, P=1.0)),
    # 12/10 E35 (hướng diffusion phục hồi, LODO DINOv3-B 8 epoch, mốc dinov3_vitb16). Suy luận luôn trên ảnh gốc, không chạy SR lúc nộp.
    # R1 rsesr / R2 rsdiff: với xác suất P thay ảnh gốc bằng ảnh phục hồi (Real-ESRGAN / ResShift v3), giữ mọi nhãn. R3 rsdiffg: như R2 + fidelity gate.
    # R4 rscons: mỗi ảnh 2 view (gốc, phục hồi): BCE gốc + ALPHA · BCE phục hồi (nhãn qua cổng) + CONS · consistency (thuộc tính qua cổng, kể cả
    # thuộc tính dataset không gán nhãn). R5 augcons (đối chứng R4): view thứ 2 là ảnh gốc augment lần nữa -> tách lợi ích diffusion khỏi lợi ích consistency.
    "dinov3_vitb16_rsesr":    dict(_V3B8, RESTORE=dict(METHOD="esrgan", P=0.5)),
    "dinov3_vitb16_rsdiff":   dict(_V3B8, RESTORE=dict(METHOD="resshift", P=0.5)),
    "dinov3_vitb16_rsdiffg":  dict(_V3B8, RESTORE=dict(METHOD="resshift", P=0.5, GATE=_RS_GATE)),
    "dinov3_vitb16_rscons":   dict(_V3B8, RESTORE=dict(METHOD="resshift", PAIR=True, ALPHA=0.5, CONS=1.0, GATE=_RS_GATE)),
    "dinov3_vitb16_augcons":  dict(_V3B8, RESTORE=dict(METHOD=None, PAIR=True, ALPHA=0.5, CONS=1.0)),
    "dinov3_vitb16_rssinsrg": dict(_V3B8, RESTORE=dict(METHOD="sinsr", P=0.5, GATE=_RS_GATE)),       # chỉ khi SinSR chạy được ở notebook sinh ảnh
    "dinov3_vitb16_rsesrcons": dict(_V3B8, RESTORE=dict(METHOD="esrgan", PAIR=True, ALPHA=0.5, CONS=1.0, GATE=_RS_GATE)),
    "dbg_rs":                 dict(ARCH="radio_dbg", IMG=(64, 32), LR=1e-4, LLRD=0.85, DROP_PATH=0.1, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                                   RESTORE=dict(METHOD="esrgan", P=0.9, GATE=dict(Q=0.5, DE_MIN=3.0, PS_MAX=30.0))),
    "dbg_rspair":             dict(ARCH="radio_dbg", IMG=(64, 32), LR=1e-4, LLRD=0.85, DROP_PATH=0.1, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                                   RESTORE=dict(METHOD="esrgan", PAIR=True, ALPHA=0.5, CONS=1.0, GATE=dict(Q=0.5, DE_MIN=3.0, PS_MAX=30.0))),
    # 11/10 E31 (LODO bỏ PA100k, DINOv3-B, mốc dinov3_vitb16 của E14): các hướng chưa thử, mỗi biến thể đổi đúng 1 thứ
    "dinov3_vitb16_ctemp":  dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, AUG=dict(CT_P=0.5)),
    "dinov3_vitb16_trunc":  dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, AUG=dict(TRUNC_P=0.3)),
    # E31b (11/10 tối): ctemp + trunc cùng lúc (E31 bỏ PA100k: ctemp +0.0009, trunc +0.0016, đều dưới ngưỡng +0.002) -> xác nhận trên Market / PETA chưa thấy
    "dinov3_vitb16_cttr":   dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, AUG=dict(CT_P=0.5, TRUNC_P=0.3)),
    "dinov3_vitb16_drop8":  dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, HEAD_DROP=0.8),
    "dinov3_vitb16_augmix": dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, AUG=dict(AUGMIX_P=0.5)),
    "dinov3_vitb16_lbox":   dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, AUG=dict(LBOX=True)),
    "dbg_e31":              dict(ARCH="dinov2_vits14_reg", IMG=(56, 28), LR=1e-4, LLRD=0.75, DROP_PATH=0.0, HEAD_LR=1e-3, GRAD_CLIP=5.0, HEAD_DROP=0.8,
                                 AUG=dict(CT_P=1.0, TRUNC_P=1.0, AUGMIX_P=1.0, LBOX=True)),
    # 11/10 G14: SigLIP2-So400m/16 ở 224x112 (cùng độ phân giải hệ ghép v15) làm thành viên khác họ thay / thêm cạnh DINOv2-L. Ước tính A10 ~2.6 ms/ảnh.
    "siglip2_so400m16_224": dict(ARCH="siglip2_so400m16", IMG=(224, 112), LR=3e-5, LLRD=0.80, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=1.0, BATCH=64,
                                 FREEZE=0, FROZEN_FP16=True, AUTO_FIT=[(0, False), (4, False), (8, False), (12, False)], AUTO_EP=True),
    "dbg_siglip":           dict(ARCH="siglip2_dbg", IMG=(64, 32), LR=1e-4, LLRD=0.8, DROP_PATH=0.1, HEAD_LR=1e-3, GRAD_CLIP=1.0),
    # 10/10 E29 (luồng phụ, LODO bỏ PA100k, so với dinov3_vitb16 của E14): ASL thay BCE; asl = không wBCE, aslw = wBCE^0.5 + γ- nhẹ hơn
    "dinov3_vitb16_asl":  dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                               ASL=dict(GP=0.0, GN=4.0, CLIP=0.05), WBCE_POW=0.0),
    "dinov3_vitb16_aslw": dict(ARCH="dinov3_vitb16", IMG=(256, 128), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                               ASL=dict(GP=0.0, GN=2.0, CLIP=0.05), WBCE_POW=0.5),
    "dbg_asl":            dict(ARCH="dinov2_vits14_reg", IMG=(56, 28), LR=1e-4, LLRD=0.75, DROP_PATH=0.0, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                               ASL=dict(GP=0.0, GN=4.0, CLIP=0.05), WBCE_POW=0.0),
    "dbg_eva":            dict(ARCH="dinov3_dbgeva", IMG=(64, 32), LR=1e-4, LLRD=0.85, DROP_PATH=0.1, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                               FREEZE=1, FROZEN_FP16=True, CKPT=True, EXTRA_VAL=[(48, 32)], FP16_CHECK=True),
    "dinov3_vitl16_kdLL": dict(ARCH="dinov3_vitl16", IMG=(256, 128), LR=3e-5, LLRD=0.85, DROP_PATH=0.15, HEAD_LR=1e-3, GRAD_CLIP=5.0, BATCH=64,
                               KD=dict(TEACHERS=["dinov3_vitl16", "dinov2_vitl14_reg"], LAMBDA=0.5, T=1.0, ONLINE=True, ALL=True)),
    "dinov2_vits14_reg": dict(IMG=(252, 126), LR=1e-4, LLRD=0.75, DROP_PATH=0.05, HEAD_LR=1e-3, GRAD_CLIP=5.0),
    "dinov2_vitb14_reg_parts": dict(ARCH="dinov2_vitb14_reg", PARTS=True, IMG=(252, 126), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0),
    "clip_vitb16_parts":       dict(ARCH="clip_vitb16", PARTS=True, IMG=(256, 128), LR=3e-5, LLRD=0.70, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=1.0),
    "dinov2_vits14_reg_parts": dict(ARCH="dinov2_vits14_reg", PARTS=True, IMG=(252, 126), LR=1e-4, LLRD=0.75, DROP_PATH=0.05, HEAD_LR=1e-3, GRAD_CLIP=5.0),
    # độ phân giải thấp (đo end-to-end torch 2.4.1: DINOv2-B fp32 196x98 = 0.78× bản cũ, chỉ mất ~0.005 điểm khi chưa train lại)
    # E1: cùng model gốc, đổi cách train
    "dinov2_vitb14_reg_dgaug":     dict(ARCH="dinov2_vitb14_reg", IMG=(252, 126), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0,
                                        AUG=dict(MOTION_P=0.3, CROP_P=0.5, CROP_MIN=0.8)),
    "dinov2_vitb14_reg_ep12":      dict(ARCH="dinov2_vitb14_reg", IMG=(252, 126), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0, EPOCHS=12),
    "clip_vitb16_160":             dict(ARCH="clip_vitb16", IMG=(160, 80), LR=3e-5, LLRD=0.70, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=1.0),
    # E25 (07/10): PE-Core B/16 (Meta Perception Encoder, Apache-2.0) làm model phụ khác họ thay CLIP; siêu tham số theo CLIP (cùng kiểu pretrain ảnh-chữ)
    "pe_core_b16_160":             dict(ARCH="pe_core_b16", IMG=(160, 80), LR=3e-5, LLRD=0.70, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=1.0),
    "pe_core_b16":                 dict(IMG=(256, 128), LR=3e-5, LLRD=0.70, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=1.0),
    "pe_core_b16_dbg":             dict(ARCH="pe_core_b16", IMG=(64, 32), LR=3e-5, LLRD=0.70, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=1.0),
    "dinov2_vitb14_reg_196":       dict(ARCH="dinov2_vitb14_reg", IMG=(196, 98), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0),
    "dinov2_vitb14_reg_196_parts": dict(ARCH="dinov2_vitb14_reg", PARTS=True, IMG=(196, 98), LR=5e-5, LLRD=0.75, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=5.0),
    "clip_vitb16_192":             dict(ARCH="clip_vitb16", IMG=(192, 96), LR=3e-5, LLRD=0.70, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=1.0),
    "clip_vitb16_224":   dict(ARCH="clip_vitb16", IMG=(224, 112), LR=3e-5, LLRD=0.70, DROP_PATH=0.10, HEAD_LR=1e-3, GRAD_CLIP=1.0),
}
if os.environ.get("UPAR_FINAL_FULL"): CFG["FINAL_FULL"] = os.environ["UPAR_FINAL_FULL"] == "1"
if LOCAL_DEBUG:     # máy cá nhân, CPU: chỉ kiểm tra code chạy thông
    CFG.update(DEBUG=True, BATCH=8, WORKERS=0, MULTI_GPU=False, LOG_EVERY=2,
               CANDIDATES=os.environ.get("UPAR_CANDIDATES", "clip_vitb16,dinov2_vitb14_reg").split(","))

WORK = Path("/kaggle/working") if IS_KAGGLE else Path(os.environ.get("UPAR_WORK", "_local_work")).resolve()
RUNS = WORK / "runs_v2"; RUNS.mkdir(parents=True, exist_ok=True)
STAMP = time.strftime("%m%d_%H%M")
CODE = WORK / "code_v2"; CODE.mkdir(parents=True, exist_ok=True)
print("Kaggle:", IS_KAGGLE, "| local debug:", LOCAL_DEBUG, "| WORK:", WORK)
# DINOv3 là repo gated: token HF đọc từ Kaggle Secret (Add-ons → Secrets, gắn vào notebook), không ghi ra file / output
if IS_KAGGLE and not os.environ.get("HF_TOKEN"):
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret(CFG["HF_SECRET"]); print("HF token: đã đọc từ Kaggle Secret", CFG["HF_SECRET"])
    except Exception as e:
        print(f"(không đọc được Kaggle Secret {CFG['HF_SECRET']}: {type(e).__name__} - chỉ cần khi train DINOv3)")
print("Ứng viên:", CFG["CANDIDATES"], "| DEBUG:", CFG["DEBUG"])

def seed_all(s):
    random.seed(s); np.random.seed(s)
    import torch; torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# %% [markdown]
# ## 1. Annotation và ảnh (tự tìm trong Input, không có thì tự tải)

# %%
DS_NAMES = ["Market1501", "PA100k", "PETA"]
PROBE = {"Market1501": "Market1501/bounding_box_train/0002_c1s1_000451_03.jpg",
         "PA100k":     "PA100k/release_data/release_data/000001.jpg"}
REPO = WORK / "upar"

def clone_repo():
    if not (REPO / "data/annotations/task1/train/gt.csv").exists():
        shutil.rmtree(REPO, ignore_errors=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/speckean/UPAR-Challenge-2027", str(REPO)], check=True)
    return REPO

_IDX = {}
def input_dirs(root="/kaggle/input", max_depth=10, max_files=300):
    # Chỉ mục thư mục trong Input, duyệt 1 lần. Không đi vào thư mục có > max_files file (thư mục ảnh): glob đệ quy cũ duyệt cả
    # trăm nghìn ảnh, trên Kaggle (06/10) mất 40-55 phút mỗi lần.
    if root not in _IDX:
        t0, out, stack = time.time(), [], [(Path(root), 0)]
        while stack:
            d, k = stack.pop(); subs, nf = [], 0
            try:
                with os.scandir(d) as it:
                    for e in it:
                        if e.is_dir(follow_symlinks=True): subs.append(e.path)
                        else:
                            nf += 1
                            if nf > max_files: break
            except OSError:
                continue
            if nf > max_files: continue
            out.append(d)
            if k < max_depth: stack += [(Path(x), k + 1) for x in subs]
        _IDX[root] = out
        if out: print(f"Chỉ mục Input: {len(out)} thư mục ({time.time() - t0:.0f}s)", flush=True)
    return _IDX[root]

def find_ann():
    if os.environ.get("UPAR_ANN"): return Path(os.environ["UPAR_ANN"])
    hits = sorted(d.parent for d in input_dirs() if d.name == "train" and d.parent.name == "task1" and (d / "gt.csv").exists())
    pref = os.environ.get("UPAR_ANN_PREFER", CFG.get("ANN_PREFER", ""))
    if pref:                                               # 07/10: train / val chia lại -> bắt buộc có trong Input, không lặng lẽ dùng bản chính thức
        hits = [h for h in hits if pref in str(h) and (pref != "resplit" or "resplitclean" not in str(h)) and (pref != "resplitclean" or "resplitcleanconv" not in str(h))]   # E30: nhãn sạch phải gọi đích danh
        assert hits, f"ANN_PREFER = '{pref}' nhưng không thấy annotation chia lại trong Input (Add Input dataset upar-resplit-0710)"
        return hits[0]
    hits = [h for h in hits if "resplit" not in str(h)]   # mặc định: annotation chính thức kể cả khi lỡ gắn dataset chia lại
    if hits: return hits[0]
    return clone_repo() / "data/annotations/task1"

def _is_data_root(p):
    return all((p / PROBE[k]).exists() for k in PROBE) and (p / "PETA" / "images").exists()

def find_data():
    if os.environ.get("UPAR_DATA"): return Path(os.environ["UPAR_DATA"])
    cands = [WORK / "data"] + [d.parent for d in input_dirs() if d.name == "PA100k"]
    for c in cands:
        if _is_data_root(c): return c
    return None

def download_data(dst=WORK / "data"):
    repo = clone_repo()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown==5.2.0"], check=True)   # gdown>=6 bỏ tham số fuzzy
    dst.mkdir(parents=True, exist_ok=True)
    for fn, name in [("prepare_pa100k", "PA100k"), ("prepare_peta", "PETA"), ("prepare_market", "Market1501")]:
        ok = (dst / PROBE[name]).exists() if name in PROBE else (dst / "PETA/images").exists()
        if ok: print(f"[{name}] đã có"); continue
        if name == "Market1501":
            hits = [d for d in input_dirs() if (d / "bounding_box_train").is_dir() and (d / "query").is_dir()]
            if hits:
                if (dst / name).is_symlink(): (dst / name).unlink()
                os.symlink(hits[0], dst / name); print("[Market1501] symlink ->", hits[0]); continue
        if name == "PA100k":
            pz = dst / "PA100k" / "data.zip"; pz.parent.mkdir(parents=True, exist_ok=True)
            if not (pz.exists() and pz.stat().st_size > 400 * 2**20):
                subprocess.run([sys.executable, "-c",
                                f"import gdown; gdown.download(id='1Bod3MrbdCRiSvpR7NSOYi-rF3-ushwtj', output='{pz}', quiet=False)"])
            if pz.exists() and pz.stat().st_size > 400 * 2**20:
                zipfile.ZipFile(pz).extractall(pz.parent); pz.unlink(); print("[PA100k] OK")
            else:
                print("[PA100k] tải data.zip thất bại (Google Drive quota) -> chạy lại sau hoặc gắn PA100k từ Kaggle Input")
            continue
        for p in [dst / name, dst / "Market-1501-v15.09.15", dst / "market_1501.zip"]:
            if p.is_symlink() or p.is_file(): p.unlink()
            elif p.exists(): shutil.rmtree(p)
        r = subprocess.run([sys.executable, "-c", f"from download_datasets import *; {fn}(Path('{dst}'))"], cwd=repo)
        print(f"[{name}] exit code {r.returncode}")
    for z in dst.rglob("*.zip"): z.unlink()

ANN = find_ann(); print("ANN =", ANN)
DATA = find_data()
if DATA is None:
    download_data(); DATA = find_data()
assert DATA is not None, "Không tìm thấy/tải được đủ ảnh 3 dataset - xem log phía trên"
print("DATA =", DATA)

# %% [markdown]
# ## 2. Annotation, mask, fold cross-fit

# %%
def load(split):
    d = pd.read_csv(ANN / split / "gt.csv"); d = d.rename(columns={d.columns[0]: "image"}); d["split"] = split
    return d
df = pd.concat([load("train"), load("val")], ignore_index=True)
ATTR = [c for c in df.columns if c not in ("image", "split")]
assert len(ATTR) == 40
assert all(a.startswith("UpperBody-Color") for a in ATTR[8:20]) and all(a.startswith("LowerBody-Color") for a in ATTR[21:33])   # khớp UB_COLOR / LB_COLOR trong model.py
df["ds"] = df.image.str.split("/").str[0]
DS_ID = {d: i for i, d in enumerate(DS_NAMES)}

# mask (dataset, thuộc tính): 0 = dataset không annotate thuộc tính này -> bỏ khỏi loss (giữ nguyên như bản 0.776)
tr_all = df[df.split == "train"]
MASK = np.ones((len(DS_NAMES), 40), np.float32)
for d, g in tr_all.groupby("ds"):
    MASK[DS_ID[d], g[ATTR].values.sum(0) == 0] = 0
for a in ["UpperBody-Color-Other", "LowerBody-Color-Other"]:
    MASK[DS_ID["PETA"], ATTR.index(a)] = 0

TR = df[df.split == "train"]; VA = df[df.split == "val"]
if CFG["DEBUG"]:
    n_tr, n_va = (40, 30) if LOCAL_DEBUG else (1000, 500)
    TR = TR.sample(frac=1, random_state=0).groupby("ds").head(n_tr)
    VA = VA.sample(frac=1, random_state=0).groupby("ds").head(n_va)
    CFG["EPOCHS"] = 1
TR, VA = TR.reset_index(drop=True), VA.reset_index(drop=True)
print("train:", TR.ds.value_counts().to_dict(), "| val:", VA.ds.value_counts().to_dict())

def pos_rate(frame):
    m = MASK[frame.ds.map(DS_ID).values]
    return ((frame[ATTR].values * m).sum(0) / np.maximum(m.sum(0), 1)).clip(1e-4, 1 - 1e-4).astype(np.float32)
POS_RATE = pos_rate(TR)

def make_folds(frame, seed=0):
    # 2 fold trên val. Market1501 chia theo identity (4 ký tự đầu tên file) để 2 fold không chung người.
    rng = np.random.RandomState(seed); f = np.zeros(len(frame), int)
    for d, idx in frame.groupby("ds").indices.items():
        if d == "Market1501":
            pid = frame.image.values[idx]; pid = np.array([p.rsplit("/", 1)[-1][:4] for p in pid])
            u = np.unique(pid); rng.shuffle(u); fm = {p: i % 2 for i, p in enumerate(u)}
            f[idx] = [fm[p] for p in pid]
        else:
            perm = rng.permutation(len(idx)); f[idx[perm]] = np.arange(len(idx)) % 2
    return f
FOLDS = make_folds(VA)
print("fold sizes:", np.bincount(FOLDS))

# %% [markdown]
# ## 3. `model.py` (đóng gói luôn vào submission, chỉ cần torch + torchvision)

# %%
MODEL_PY = r'''
import math, torch, torch.nn as nn, torch.nn.functional as F, torchvision
import torch.utils.checkpoint
import torchvision.transforms.functional as TF

IMAGENET_NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP_NORM = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
MEAN, STD = IMAGENET_NORM            # giữ tương thích với bản cũ

SIGLIP_NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def norm_for(arch):
    return CLIP_NORM if arch.startswith(("clip", "radio", "eva02")) else SIGLIP_NORM if arch.startswith(("siglip", "pe_core")) else IMAGENET_NORM

def grayworld(t):
    # Cân bằng trắng gray-world trên tensor [0,1] (3,H,W): đưa trung bình 3 kênh về bằng nhau, giới hạn hệ số 0.6-1.6
    m = t.mean((1, 2)); g = (m.mean() / m.clamp(min=1e-3)).clamp(0.6, 1.6)
    return (t * g[:, None, None]).clamp(0, 1)

def crop_frac(img, crop):
    # 09/10: cắt theo tỷ lệ (x0, y0, x1, y1) của ảnh gốc, trước mọi phép resize (model chuyên vùng đầu). crop rỗng -> giữ nguyên ảnh
    if not crop: return img
    W, H = img.size; x0, y0, x1, y1 = crop
    l, t = int(round(x0 * W)), int(round(y0 * H))
    return img.crop((l, t, max(int(round(x1 * W)), l + 1), max(int(round(y1 * H)), t + 1)))

def letterbox(img, h, w, mean=IMAGENET_NORM[0]):
    # 11/10 (E31 lbox): giữ tỷ lệ khung — co ảnh vừa khít (w, h) rồi đệm 2 bên / trên dưới bằng màu trung bình (= 0 sau chuẩn hoá)
    s = min(w / img.width, h / img.height); nw, nh = max(1, round(img.width * s)), max(1, round(img.height * s))
    from PIL import Image as _I
    c = _I.new("RGB", (w, h), tuple(int(round(v * 255)) for v in mean)); c.paste(img.resize((nw, nh), resample=2), ((w - nw) // 2, (h - nh) // 2))
    return c

def preprocess(img, h, w, mean=IMAGENET_NORM[0], std=IMAGENET_NORM[1], wb=False, lbox=False):
    # PIL RGB -> tensor chuẩn hoá. Dùng chung cho val (notebook) và inference (run.py).
    img = letterbox(img, h, w, mean) if lbox else img.resize((w, h), resample=2)          # 2 = BILINEAR
    t = TF.to_tensor(img)
    if wb: t = grayworld(t)
    return TF.normalize(t, mean, std)

def _drop_path(x, p, training):
    if p == 0.0 or not training: return x
    keep = 1 - p
    return x * x.new_empty(x.shape[0], 1, 1).bernoulli_(keep) / keep

STRIPES = ((0.0, 0.30), (0.20, 0.60), (0.50, 1.00))      # tỷ lệ chiều cao: đầu, thân trên, thân dưới

def _stripe_means(tok, grid):
    # tok: (B, gh*gw, C) patch token theo thứ tự hàng -> trung bình từng dải ngang
    B, N, C = tok.shape; gh, gw = grid
    t = tok.reshape(B, gh, gw, C)
    return [t[:, int(round(a * gh)):max(int(round(a * gh)) + 1, int(round(b * gh)))].mean((1, 2)) for a, b in STRIPES]

def _resize_pos(grid_w, new_grid, antialias=True):
    # grid_w: (N, C) lưới vuông -> nội suy bicubic sang new_grid (h, w)
    s = int(math.sqrt(grid_w.shape[0])); C = grid_w.shape[1]
    g = grid_w.reshape(1, s, s, C).permute(0, 3, 1, 2).float()
    g = F.interpolate(g, size=new_grid, mode="bicubic", align_corners=False, antialias=antialias)
    return g.permute(0, 2, 3, 1).reshape(-1, C), s

# ------------------------------ DINOv2 ViT (key khớp checkpoint chính thức của Meta) ------------------------------
class _Attn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__(); self.h = heads
        self.qkv = nn.Linear(dim, dim * 3); self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))

class _Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__(); self.fc1 = nn.Linear(dim, hidden); self.act = nn.GELU(); self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))

class _LayerScale(nn.Module):
    def __init__(self, dim): super().__init__(); self.gamma = nn.Parameter(torch.ones(dim) * 1e-5)
    def forward(self, x): return x * self.gamma

class _Block(nn.Module):
    def __init__(self, dim, heads, dp):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6); self.attn = _Attn(dim, heads); self.ls1 = _LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6); self.mlp = _Mlp(dim, dim * 4); self.ls2 = _LayerScale(dim)
        self.dp = dp
    def forward(self, x):
        x = x + _drop_path(self.ls1(self.attn(self.norm1(x))), self.dp, self.training)
        return x + _drop_path(self.ls2(self.mlp(self.norm2(x))), self.dp, self.training)

class _PatchEmbed(nn.Module):
    def __init__(self, dim, patch): super().__init__(); self.proj = nn.Conv2d(3, dim, patch, patch)
    def forward(self, x): return self.proj(x).flatten(2).transpose(1, 2)

class DinoViT(nn.Module):
    def __init__(self, dim, depth, heads, grid, n_reg=0, patch=14, drop_path=0.0, parts=False):
        super().__init__()
        self.grid, self.n_reg, self.parts = grid, n_reg, parts
        self.patch_embed = _PatchEmbed(dim, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + grid[0] * grid[1], dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, n_reg, dim)) if n_reg else None
        dps = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList([_Block(dim, heads, d) for d in dps])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.out_dim = (2 + 3 * parts) * dim
    def features(self, x, taps=()):
        # (đặc trưng gộp, patch token cuối, patch token sau các khối trong taps)
        B = x.shape[0]
        x = torch.cat([self.cls_token.expand(B, -1, -1), self.patch_embed(x)], 1) + self.pos_embed
        if self.n_reg:
            x = torch.cat([x[:, :1], self.register_tokens.expand(B, -1, -1), x[:, 1:]], 1)
        mids = []
        for i, b in enumerate(self.blocks):
            x = b(x)
            if i in taps: mids.append(x[:, 1 + self.n_reg:])
        x = self.norm(x); pt = x[:, 1 + self.n_reg:]
        f = torch.cat([x[:, 0], pt.mean(1)] + (_stripe_means(pt, self.grid) if self.parts else []), 1)   # CLS + trung bình patch (+ 3 dải ngang)
        return f, pt, mids
    def forward(self, x, return_tokens=False):
        f, pt, _ = self.features(x)
        return (f, pt) if return_tokens else f

DINO_CFG = {"dinov2_vits14": (384, 12, 6), "dinov2_vitb14": (768, 12, 12), "dinov2_vitl14": (1024, 24, 16)}

def load_dinov2_pretrained(vit, name, n_reg):
    url = f"https://dl.fbaipublicfiles.com/dinov2/{name}/{name}{'_reg4' if n_reg else ''}_pretrain.pth"
    sd = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    pe = sd["pos_embed"]
    grid, s = _resize_pos(pe[0, 1:], vit.grid)
    sd["pos_embed"] = torch.cat([pe[:, :1], grid[None].to(pe.dtype)], 1)
    sd.pop("mask_token", None)
    missing, unexpected = vit.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    print(f"Loaded {url} | pos_embed {s}x{s} -> {vit.grid[0]}x{vit.grid[1]}")

# ------------------------------ CLIP ViT (key khớp HuggingFace CLIPVisionModel + visual_projection) ------------------------------
class _QuickGELU(nn.Module):
    def forward(self, x): return x * torch.sigmoid(1.702 * x)

class _ClipAttn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__(); self.h = heads
        self.q_proj = nn.Linear(dim, dim); self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim); self.out_proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape; h = self.h
        q = self.q_proj(x).view(B, N, h, C // h).transpose(1, 2)
        k = self.k_proj(x).view(B, N, h, C // h).transpose(1, 2)
        v = self.v_proj(x).view(B, N, h, C // h).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(x.transpose(1, 2).reshape(B, N, C))

class _ClipMlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__(); self.fc1 = nn.Linear(dim, hidden); self.act = _QuickGELU(); self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))

class _ClipLayer(nn.Module):
    def __init__(self, dim, heads, dp):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-5); self.self_attn = _ClipAttn(dim, heads)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-5); self.mlp = _ClipMlp(dim, dim * 4)
        self.dp = dp
    def forward(self, x):
        x = x + _drop_path(self.self_attn(self.layer_norm1(x)), self.dp, self.training)
        return x + _drop_path(self.mlp(self.layer_norm2(x)), self.dp, self.training)

class _ClipEmb(nn.Module):
    def __init__(self, dim, patch, grid):
        super().__init__()
        self.class_embedding = nn.Parameter(torch.zeros(dim))
        self.patch_embedding = nn.Conv2d(3, dim, patch, patch, bias=False)
        self.position_embedding = nn.Embedding(1 + grid[0] * grid[1], dim)

class _ClipEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dps):
        super().__init__(); self.layers = nn.ModuleList([_ClipLayer(dim, heads, d) for d in dps])

class ClipViT(nn.Module):
    def __init__(self, dim, depth, heads, grid, patch, proj_dim, drop_path=0.0, parts=False):
        super().__init__()
        self.grid, self.dim, self.proj_dim, self.parts = grid, dim, proj_dim, parts
        self.embeddings = _ClipEmb(dim, patch, grid)
        self.pre_layrnorm = nn.LayerNorm(dim, eps=1e-5)          # tên giống hệt HF (kể cả lỗi chính tả)
        dps = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.encoder = _ClipEncoder(dim, depth, heads, dps)
        self.post_layernorm = nn.LayerNorm(dim, eps=1e-5)
        self.visual_projection = nn.Linear(dim, proj_dim, bias=False)
        self.out_dim = proj_dim + (1 + 3 * parts) * dim
    @property
    def blocks(self): return self.encoder.layers
    def embed(self, x):
        B = x.shape[0]
        p = self.embeddings.patch_embedding(x).flatten(2).transpose(1, 2)
        c = self.embeddings.class_embedding.to(p.dtype).expand(B, 1, -1)
        x = torch.cat([c, p], 1) + self.embeddings.position_embedding.weight[None].to(p.dtype)
        x = self.pre_layrnorm(x)
        for l in self.encoder.layers: x = l(x)
        return x
    def image_embeds(self, x):                                  # giống CLIPModel.get_image_features
        return self.visual_projection(self.post_layernorm(self.embed(x)[:, 0]))
    def forward(self, x, return_tokens=False):
        x = self.embed(x)
        e = F.normalize(self.visual_projection(self.post_layernorm(x[:, 0])).float(), dim=-1)   # embedding ảnh CLIP
        f = [e, self.post_layernorm(x[:, 1:].mean(1)).float()]                                  # trung bình patch
        if self.parts: f += [self.post_layernorm(s).float() for s in _stripe_means(x[:, 1:], self.grid)]
        f = torch.cat(f, 1)
        return (f, self.post_layernorm(x[:, 1:])) if return_tokens else f

CLIP_CFG = {"clip_vitb16": ("openai/clip-vit-base-patch16", 768, 12, 12, 16, 512),
            "clip_vitl14": ("openai/clip-vit-large-patch14", 1024, 24, 16, 14, 768)}

def clip_state_from_hf(hf_sd, grid):
    sd = {k[len("vision_model."):]: v for k, v in hf_sd.items() if k.startswith("vision_model.")}
    sd["visual_projection.weight"] = hf_sd["visual_projection.weight"]
    sd.pop("embeddings.position_ids", None)
    pe = sd["embeddings.position_embedding.weight"]
    g, s = _resize_pos(pe[1:], grid)
    sd["embeddings.position_embedding.weight"] = torch.cat([pe[:1].float(), g], 0).to(pe.dtype)
    return sd, s

def load_clip_pretrained(vit, arch):
    from transformers import CLIPModel            # chỉ cần khi train, run.py không dùng
    hf = CLIPModel.from_pretrained(CLIP_CFG[arch][0]).state_dict()
    sd, s = clip_state_from_hf(hf, vit.grid)
    missing, unexpected = vit.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    print(f"Loaded {CLIP_CFG[arch][0]} | pos_embed {s}x{s} -> {vit.grid[0]}x{vit.grid[1]}")

# ------------------------------ SigLIP / SigLIP 2 ViT (key khớp HuggingFace SiglipVisionModel) ------------------------------
class _GELUTanh(nn.Module):
    def forward(self, x): return F.gelu(x, approximate="tanh")

class _SigMlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__(); self.fc1 = nn.Linear(dim, hidden); self.act = _GELUTanh(); self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))

class _SigLayer(nn.Module):
    def __init__(self, dim, heads, dp, hidden=None):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6); self.self_attn = _ClipAttn(dim, heads)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6); self.mlp = _SigMlp(dim, hidden or dim * 4)
        self.dp = dp
    def forward(self, x):
        x = x + _drop_path(self.self_attn(self.layer_norm1(x)), self.dp, self.training)
        return x + _drop_path(self.mlp(self.layer_norm2(x)), self.dp, self.training)

class _SigEmb(nn.Module):
    def __init__(self, dim, patch, grid):
        super().__init__()
        self.patch_embedding = nn.Conv2d(3, dim, patch, patch)
        self.position_embedding = nn.Embedding(grid[0] * grid[1], dim)

class _SigHead(nn.Module):                      # multihead attention pooling (MAP)
    def __init__(self, dim, heads, hidden=None):
        super().__init__()
        self.probe = nn.Parameter(torch.zeros(1, 1, dim))
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.layernorm = nn.LayerNorm(dim, eps=1e-6); self.mlp = _SigMlp(dim, hidden or dim * 4)
    def forward(self, x):
        h = self.attention(self.probe.expand(x.shape[0], -1, -1).to(x.dtype), x, x, need_weights=False)[0]
        h = h + self.mlp(self.layernorm(h))
        return h[:, 0]

class _SigEncoder(nn.Module):
    def __init__(self, dim, heads, dps, hidden=None):
        super().__init__(); self.layers = nn.ModuleList([_SigLayer(dim, heads, d, hidden) for d in dps])

class SiglipViT(nn.Module):
    def __init__(self, dim, depth, heads, grid, patch, drop_path=0.0, parts=False, hidden=None):
        super().__init__()
        self.grid, self.dim, self.parts = grid, dim, parts
        self.embeddings = _SigEmb(dim, patch, grid)
        dps = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.encoder = _SigEncoder(dim, heads, dps, hidden)
        self.post_layernorm = nn.LayerNorm(dim, eps=1e-6)
        self.head = _SigHead(dim, heads, hidden)
        self.out_dim = (2 + 3 * parts) * dim
    @property
    def blocks(self): return self.encoder.layers
    def embed(self, x):
        p = self.embeddings.patch_embedding(x).flatten(2).transpose(1, 2)
        x = p + self.embeddings.position_embedding.weight[None].to(p.dtype)
        for l in self.encoder.layers: x = l(x)
        return self.post_layernorm(x)
    def image_embeds(self, x):                      # giống SiglipModel.get_image_features
        return self.head(self.embed(x))
    def forward(self, x, return_tokens=False):
        x = self.embed(x)
        e = F.normalize(self.head(x).float(), dim=-1)
        f = torch.cat([e, x.mean(1).float()] + ([s.float() for s in _stripe_means(x, self.grid)] if self.parts else []), 1)
        return (f, x) if return_tokens else f

SIGLIP_CFG = {"siglip2_vitb16": ("google/siglip2-base-patch16-224", 768, 12, 12, 16),
              "siglip_vitb16":  ("google/siglip-base-patch16-224", 768, 12, 12, 16),
              # 11/10 G14: SigLIP2 So400m/16 (Apache-2.0; 27 khối, dim 1152, MLP 4304; pos_embed 16×16 nội suy về lưới ảnh). Thành viên thứ 2 khác họ DINO
              "siglip2_so400m16": ("google/siglip2-so400m-patch16-256", 1152, 27, 16, 16, 4304),
              "siglip2_dbg": (None, 64, 2, 2, 16, 96)}                    # chỉ để chạy thử code trên CPU

def siglip_state_from_hf(hf_sd, grid):
    sd = {k[len("vision_model."):]: v for k, v in hf_sd.items() if k.startswith("vision_model.")}
    sd.pop("embeddings.position_ids", None)
    pe = sd["embeddings.position_embedding.weight"]
    g, s = _resize_pos(pe, grid, antialias=False)        # giống interpolate_pos_encoding của HF
    sd["embeddings.position_embedding.weight"] = g.to(pe.dtype)
    return sd, s

def load_siglip_pretrained(vit, arch):
    from transformers import AutoModel
    hf = AutoModel.from_pretrained(SIGLIP_CFG[arch][0]).state_dict()
    sd, s = siglip_state_from_hf(hf, vit.grid)
    missing, unexpected = vit.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    print(f"Loaded {SIGLIP_CFG[arch][0]} | pos_embed {s}x{s} -> {vit.grid[0]}x{vit.grid[1]}")

# ------------------------------ Decoder truy vấn thuộc tính (K11) ------------------------------
class _XAttn(nn.Module):
    # attention đa đầu viết bằng nn.Linear thường (quantize_dynamic xử lý được, không dùng nn.MultiheadAttention)
    def __init__(self, dim, heads):
        super().__init__(); self.h = heads
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim); self.v = nn.Linear(dim, dim); self.o = nn.Linear(dim, dim)
    def forward(self, q, kv):
        B, Nq, C = q.shape; Nk = kv.shape[1]; h = self.h
        q = self.q(q).view(B, Nq, h, C // h).transpose(1, 2)
        k = self.k(kv).view(B, Nk, h, C // h).transpose(1, 2)
        v = self.v(kv).view(B, Nk, h, C // h).transpose(1, 2)
        return self.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, Nq, C))

class QueryDecoder(nn.Module):
    # Mỗi thuộc tính 1 query: cross-attention lên patch token (tự tìm vùng ảnh liên quan), rồi self-attention giữa 40 query
    # (quan hệ giữa thuộc tính, như một đồ thị học được theo từng ảnh). Đầu ra cộng vào logit của head tuyến tính.
    # w = 0 lúc khởi tạo nên ban đầu model y hệt bản gốc.
    def __init__(self, dim, num_attr, heads):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, num_attr, dim) * 0.02)
        self.ln_q = nn.LayerNorm(dim); self.ln_kv = nn.LayerNorm(dim); self.cross = _XAttn(dim, heads)
        self.ln_s = nn.LayerNorm(dim); self.self_attn = _XAttn(dim, heads)
        self.ln_f = nn.LayerNorm(dim); self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.w = nn.Parameter(torch.zeros(num_attr, dim)); self.b = nn.Parameter(torch.zeros(num_attr))
    def forward(self, tok):
        kv = self.ln_kv(tok)
        h = self.query.expand(tok.shape[0], -1, -1)
        h = h + self.cross(self.ln_q(h), kv)
        s = self.ln_s(h); h = h + self.self_attn(s, s)
        h = h + self.mlp(self.ln_f(h))
        return (h.float() * self.w).sum(-1) + self.b

# ------------------------------ 14/10 QDEC2: đầu truy vấn thuộc tính kép, giám sát riêng (z_q) ------------------------------
# Nhóm vùng của từng thuộc tính theo thứ tự 40 thuộc tính của BTC (notebook kiểm khớp tên như UB_COLOR): 0 đầu (tuổi, tóc, kính, mũ),
# 1 thân trên (UpperBody-*, ba lô, túi), 2 thân dưới (LowerBody-*), -1 không prior (giới tính). QD2_ROWS: vùng (tâm hàng token / số hàng) của nhóm 0, 1, 2.
QD2_GROUP = [0, 0, 0, -1, 0, 0, 0] + [1] * 13 + [2] * 15 + [1, 1, 0, 0, 0]
QD2_ROWS = ((0.0, 0.30), (0.15, 0.65), (0.45, 1.0))

class _XAttnB(_XAttn):
    # như _XAttn + bias cộng vào điểm attention (broadcast theo batch / đầu); tính tay, không qua SDPA, để bias học được chắc chắn có gradient
    def forward(self, q, kv, bias):
        B, Nq, C = q.shape; Nk = kv.shape[1]; h = self.h
        q = self.q(q).view(B, Nq, h, C // h).transpose(1, 2)
        k = self.k(kv).view(B, Nk, h, C // h).transpose(1, 2)
        v = self.v(kv).view(B, Nk, h, C // h).transpose(1, 2)
        a = (q @ k.transpose(-2, -1) * (C // h) ** -0.5 + bias).softmax(-1)
        return self.o((a.to(v.dtype) @ v).transpose(1, 2).reshape(B, Nq, C))

class QueryDecoder2(nn.Module):
    # Đầu RIÊNG (không cộng vào z): patch token khối cuối + khối mid, mỗi nguồn LayerNorm + Linear -> dim, cộng embedding theo khối;
    # 40 query, MỘT cross-attention có prior vị trí mềm trên key theo hàng token (bias = thang đo học được của nhóm, khởi tạo 1,
    # × chỉ báo hàng nằm trong vùng của nhóm; không cắt ảnh), rồi MLP + linear riêng từng query -> z_q. w = 0 lúc đầu nên z_q = b (= prior).
    def __init__(self, in_dim, mid, dim=256, heads=8, num_attr=40):
        super().__init__(); self.mid = mid
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim)) for _ in range(2)])
        self.lemb = nn.Parameter(torch.randn(2, dim) * 0.02)
        self.query = nn.Parameter(torch.randn(1, num_attr, dim) * 0.02)
        self.ln_q = nn.LayerNorm(dim); self.cross = _XAttnB(dim, heads)
        self.ln_f = nn.LayerNorm(dim); self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.ln_o = nn.LayerNorm(dim); self.w = nn.Parameter(torch.zeros(num_attr, dim)); self.b = nn.Parameter(torch.zeros(num_attr))
        self.gscale = nn.Parameter(torch.ones(len(QD2_ROWS)))
        self.register_buffer("gid", torch.tensor(QD2_GROUP), persistent=False)
        self.register_buffer("rows", torch.tensor(QD2_ROWS), persistent=False)
    def prior(self, gh, gw):
        # (num_attr, gh*gw): bias của query k lên token ở hàng r = gscale[nhóm k] nếu (r + 0.5) / gh nằm trong vùng của nhóm, ngược lại 0
        row = (torch.arange(gh * gw, device=self.rows.device) // gw + 0.5) / gh
        inr = ((row >= self.rows[:, :1]) & (row <= self.rows[:, 1:])).float()
        g = self.gid.clamp(min=0)
        return self.gscale[g, None] * inr[g] * (self.gid >= 0)[:, None]
    def forward(self, toks, grid):
        assert toks[0].shape[1] == grid[0] * grid[1], (toks[0].shape, grid)
        kv = torch.cat([p(t) + e for p, t, e in zip(self.proj, toks, self.lemb)], 1)
        h = self.query.expand(kv.shape[0], -1, -1)
        h = h + self.cross(self.ln_q(h), kv, self.prior(*grid).repeat(1, len(toks)))
        h = h + self.mlp(self.ln_f(h))
        return (self.ln_o(h).float() * self.w).sum(-1) + self.b

# ------------------------------ DINOv3 ViT (RoPE 2D, patch 16), khoá khớp transformers DINOv3ViTModel ------------------------------
DINO3_CFG = {"dinov3_vits16": (384, 12, 6, "facebook/dinov3-vits16-pretrain-lvd1689m"),
             "dinov3_vitb16": (768, 12, 12, "facebook/dinov3-vitb16-pretrain-lvd1689m"),
             "dinov3_vitl16": (1024, 24, 16, "facebook/dinov3-vitl16-pretrain-lvd1689m")}

def _rot_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def _v3_rope(gh, gw, head_dim, base, device):
    # cos / sin cho từng patch: toạ độ tâm patch đưa về [-1, 1] (như get_patches_center_coordinates, không augment khi train)
    ch = torch.arange(0.5, gh, dtype=torch.float32, device=device) / gh
    cw = torch.arange(0.5, gw, dtype=torch.float32, device=device) / gw
    coords = torch.stack(torch.meshgrid(ch, cw, indexing="ij"), -1).flatten(0, 1) * 2.0 - 1.0
    inv = 1.0 / base ** torch.arange(0, 1, 4 / head_dim, dtype=torch.float32, device=device)
    ang = (2 * math.pi * coords[:, :, None] * inv[None, None, :]).flatten(1, 2).tile(2)
    return ang.cos(), ang.sin()

class _V3Emb(nn.Module):
    def __init__(self, dim, n_reg, patch):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))         # không dùng, chỉ để khớp checkpoint
        self.register_tokens = nn.Parameter(torch.zeros(1, n_reg, dim))
        self.patch_embeddings = nn.Conv2d(3, dim, patch, patch)

class _V3Attn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__(); self.h = heads
        self.q_proj = nn.Linear(dim, dim); self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim); self.o_proj = nn.Linear(dim, dim)
    def forward(self, x, cos, sin, n_pre):
        B, N, C = x.shape; hd = C // self.h
        q, k, v = (f(x).view(B, N, self.h, hd).transpose(1, 2) for f in (self.q_proj, self.k_proj, self.v_proj))
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        rope = lambda t: torch.cat([t[:, :, :n_pre], t[:, :, n_pre:] * cos + _rot_half(t[:, :, n_pre:]) * sin], 2)   # RoPE chỉ cho patch
        o = F.scaled_dot_product_attention(rope(q), rope(k), v)
        return self.o_proj(o.transpose(1, 2).reshape(B, N, C))

class _V3LS(nn.Module):
    def __init__(self, dim): super().__init__(); self.lambda1 = nn.Parameter(torch.ones(dim))
    def forward(self, x): return x * self.lambda1

class _V3Mlp(nn.Module):
    def __init__(self, dim, hidden): super().__init__(); self.up_proj = nn.Linear(dim, hidden); self.down_proj = nn.Linear(hidden, dim)
    def forward(self, x): return self.down_proj(F.gelu(self.up_proj(x)))

class _V3Layer(nn.Module):
    def __init__(self, dim, heads, dp, eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps); self.attention = _V3Attn(dim, heads); self.layer_scale1 = _V3LS(dim)
        self.norm2 = nn.LayerNorm(dim, eps=eps); self.mlp = _V3Mlp(dim, dim * 4); self.layer_scale2 = _V3LS(dim)
        self.dp = dp
    def forward(self, x, cos, sin, n_pre):
        x = x + _drop_path(self.layer_scale1(self.attention(self.norm1(x), cos, sin, n_pre)), self.dp, self.training)
        return x + _drop_path(self.layer_scale2(self.mlp(self.norm2(x))), self.dp, self.training)

class DinoV3ViT(nn.Module):
    # Cùng giao diện với DinoViT (features / forward / grid / out_dim / blocks); RoPE nên đổi kích thước ảnh không cần nội suy gì
    def __init__(self, dim, depth, heads, grid, n_reg=4, drop_path=0.0, parts=False, patch=16, eps=1e-5, theta=100.0):
        super().__init__()
        self.grid, self.n_reg, self.parts, self.patch, self.heads, self.theta = grid, n_reg, parts, patch, heads, theta
        self.embeddings = _V3Emb(dim, n_reg, patch)
        dps = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.layer = nn.ModuleList([_V3Layer(dim, heads, d, eps) for d in dps])
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.out_dim = (2 + 3 * parts) * dim
        self._rope = {}
    @property
    def blocks(self): return self.layer
    def features(self, x, taps=()):
        B, _, H, W = x.shape; self.grid = (H // self.patch, W // self.patch)
        key = (self.grid, str(x.device))
        if key not in self._rope: self._rope[key] = _v3_rope(*self.grid, self.norm.normalized_shape[0] // self.heads, self.theta, x.device)
        cos, sin = self._rope[key]; e = self.embeddings; n_pre = 1 + self.n_reg
        x = torch.cat([e.cls_token.expand(B, -1, -1), e.register_tokens.expand(B, -1, -1),
                       e.patch_embeddings(x).flatten(2).transpose(1, 2)], 1)
        mids = []
        for i, L_ in enumerate(self.layer):
            x = L_(x, cos, sin, n_pre)
            if i in taps: mids.append(x[:, n_pre:])
        x = self.norm(x); pt = x[:, n_pre:]
        f = torch.cat([x[:, 0], pt.mean(1)] + (_stripe_means(pt, self.grid) if self.parts else []), 1)
        return f, pt, mids
    def forward(self, x, return_tokens=False):
        f, pt, _ = self.features(x)
        return (f, pt) if return_tokens else f

DINO3_TIMM = {"facebook/dinov3-vits16-pretrain-lvd1689m": "timm/vit_small_patch16_dinov3.lvd1689m",
              "facebook/dinov3-vitb16-pretrain-lvd1689m": "timm/vit_base_patch16_dinov3.lvd1689m",
              "facebook/dinov3-vitl16-pretrain-lvd1689m": "timm/vit_large_patch16_dinov3.lvd1689m"}

def dinov3_sd_from_timm(t):
    # 11/10: bản timm (KHÔNG gated, chạy được qua Kaggle API không cần Secret) -> khoá kiểu transformers DINOv3ViTModel của DinoV3ViT.
    # timm gộp qkv và không lưu bias q / v (bản HF có khoá bias; k vốn không bias) -> q / v bias = 0; mask_token không dùng khi suy luận.
    sd = {"embeddings.cls_token": t["cls_token"], "embeddings.register_tokens": t["reg_token"],
          "embeddings.patch_embeddings.weight": t["patch_embed.proj.weight"], "embeddings.patch_embeddings.bias": t["patch_embed.proj.bias"],
          "embeddings.mask_token": torch.zeros_like(t["cls_token"]), "norm.weight": t["norm.weight"], "norm.bias": t["norm.bias"]}
    n = 1 + max(int(k.split(".")[1]) for k in t if k.startswith("blocks."))
    for i in range(n):
        b, o = f"blocks.{i}.", f"layer.{i}."
        q, k, v = t[b + "attn.qkv.weight"].chunk(3, 0)
        sd.update({o + "norm1.weight": t[b + "norm1.weight"], o + "norm1.bias": t[b + "norm1.bias"],
                   o + "attention.q_proj.weight": q, o + "attention.k_proj.weight": k, o + "attention.v_proj.weight": v,
                   o + "attention.q_proj.bias": torch.zeros(q.shape[0], dtype=q.dtype), o + "attention.v_proj.bias": torch.zeros(v.shape[0], dtype=v.dtype),
                   o + "attention.o_proj.weight": t[b + "attn.proj.weight"], o + "attention.o_proj.bias": t[b + "attn.proj.bias"],
                   o + "layer_scale1.lambda1": t[b + "gamma_1"], o + "layer_scale2.lambda1": t[b + "gamma_2"],
                   o + "norm2.weight": t[b + "norm2.weight"], o + "norm2.bias": t[b + "norm2.bias"],
                   o + "mlp.up_proj.weight": t[b + "mlp.fc1.weight"], o + "mlp.up_proj.bias": t[b + "mlp.fc1.bias"],
                   o + "mlp.down_proj.weight": t[b + "mlp.fc2.weight"], o + "mlp.down_proj.bias": t[b + "mlp.fc2.bias"]})
    return sd

def load_dinov3_pretrained(vit, repo):
    # Chỉ dùng lúc TRAIN (cần mạng). Ưu tiên repo gated của Meta khi có HF_TOKEN; không có token (vd notebook đẩy qua Kaggle API, không gắn được Secret)
    # hoặc tải lỗi -> bản timm không gated cùng weights (đổi khoá bằng dinov3_sd_from_timm). UPAR_DINOV3_TIMM=1 ép dùng timm. Server không gọi hàm này.
    import os
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    tok = os.environ.get("HF_TOKEN") or None; sd = None
    if tok and not os.environ.get("UPAR_DINOV3_TIMM"):
        try: sd = load_file(hf_hub_download(repo, "model.safetensors", token=tok)); src = repo
        except Exception as e: print(f"[DINOv3] không tải được {repo} ({type(e).__name__}) -> dùng bản timm")
    if sd is None:
        src = DINO3_TIMM[repo]; sd = dinov3_sd_from_timm(load_file(hf_hub_download(src, "model.safetensors")))
    missing, unexpected = vit.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, ("DINOv3 khoá không khớp", missing[:5], unexpected[:5])
    print(f"Loaded {src} (DINOv3, {len(sd)} tensor)")

# ------------------------------ DINOv3 ViT-H+/16 (840M), khoá khớp timm vit_huge_plus_patch16_dinov3 (repo timm không gated) ------------------------------
# Như DinoV3ViT nhưng: qkv gộp không bias, layer scale gamma_1/2, MLP SwiGLU (fc1_g, fc1_x, fc2; ẩn 5120), tên khoá kiểu timm Eva.
# RoPE giống hệt DINOv3-L (theta 100, rotate-half, chỉ cho patch). Thứ tự token: cls, 4 register, patch.
DINO3_EVA_CFG = {"dinov3_vith16p": (1280, 32, 20, 5120, "timm/vit_huge_plus_patch16_dinov3.lvd1689m"),   # dim, depth, heads, ẩn MLP, repo
                 "dinov3_dbgeva": (64, 3, 2, 128, None)}                                                  # chỉ để chạy thử code trên CPU

class _EvaAttn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__(); self.h = heads; self.qkv = nn.Linear(dim, dim * 3, bias=False); self.proj = nn.Linear(dim, dim)
    def forward(self, x, cos, sin, n_pre):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4).unbind(0)
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        rope = lambda t: torch.cat([t[:, :, :n_pre], t[:, :, n_pre:] * cos + _rot_half(t[:, :, n_pre:]) * sin], 2)
        o = F.scaled_dot_product_attention(rope(q), rope(k), v)
        return self.proj(o.transpose(1, 2).reshape(B, N, C))

class _SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__(); self.fc1_g = nn.Linear(dim, hidden); self.fc1_x = nn.Linear(dim, hidden); self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x): return self.fc2(F.silu(self.fc1_g(x)) * self.fc1_x(x))

class _EvaBlock(nn.Module):
    def __init__(self, dim, heads, hidden, dp, eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps); self.attn = _EvaAttn(dim, heads); self.gamma_1 = nn.Parameter(torch.ones(dim))
        self.norm2 = nn.LayerNorm(dim, eps=eps); self.mlp = _SwiGLU(dim, hidden); self.gamma_2 = nn.Parameter(torch.ones(dim))
        self.dp = dp
    def forward(self, x, cos, sin, n_pre):
        x = x + _drop_path(self.gamma_1 * self.attn(self.norm1(x), cos, sin, n_pre), self.dp, self.training)
        return x + _drop_path(self.gamma_2 * self.mlp(self.norm2(x)), self.dp, self.training)

class DinoV3Eva(nn.Module):
    # Cùng giao diện với DinoV3ViT (features / forward / grid / out_dim / blocks / norm). ckpt=True: gradient checkpointing cho khối còn train
    def __init__(self, dim, depth, heads, hidden, grid, n_reg=4, drop_path=0.0, parts=False, patch=16, eps=1e-5, theta=100.0):
        super().__init__()
        self.grid, self.n_reg, self.parts, self.patch, self.heads, self.theta, self.ckpt = grid, n_reg, parts, patch, heads, theta, False
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)); self.reg_token = nn.Parameter(torch.zeros(1, n_reg, dim))
        self.patch_embed = nn.Module(); self.patch_embed.proj = nn.Conv2d(3, dim, patch, patch)
        dps = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList([_EvaBlock(dim, heads, hidden, d, eps) for d in dps])
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.out_dim = (2 + 3 * parts) * dim
        self._rope = {}
    def tokens(self, x, taps=()):
        B, _, H, W = x.shape; self.grid = (H // self.patch, W // self.patch)
        key = (self.grid, str(x.device))
        if key not in self._rope: self._rope[key] = _v3_rope(*self.grid, self.norm.normalized_shape[0] // self.heads, self.theta, x.device)
        cos, sin = self._rope[key]; n_pre = 1 + self.n_reg
        x = torch.cat([self.cls_token.expand(B, -1, -1), self.reg_token.expand(B, -1, -1),
                       self.patch_embed.proj(x).flatten(2).transpose(1, 2)], 1)
        mids = []
        for i, b in enumerate(self.blocks):
            if self.ckpt and self.training and torch.is_grad_enabled() and any(p_.requires_grad for p_ in b.parameters()):
                x = torch.utils.checkpoint.checkpoint(b, x, cos, sin, n_pre, use_reentrant=False)
            else: x = b(x, cos, sin, n_pre)
            if i in taps: mids.append(x[:, n_pre:])
        return self.norm(x), n_pre, mids
    def features(self, x, taps=()):
        x, n_pre, mids = self.tokens(x, taps); pt = x[:, n_pre:]
        f = torch.cat([x[:, 0], pt.mean(1)] + (_stripe_means(pt, self.grid) if self.parts else []), 1)
        return f, pt, mids
    def forward(self, x, return_tokens=False):
        f, pt, _ = self.features(x)
        return (f, pt) if return_tokens else f

_EVA_CHECKED = set()
def load_dinov3_eva_pretrained(vit, repo, timm_name=None):
    # Chỉ dùng lúc TRAIN (cần mạng; repo timm không gated). Lần nạp đầu: so đầu ra với timm (nếu có) trên ảnh ngẫu nhiên, fp32 CPU.
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    sd = load_file(hf_hub_download(repo, "model.safetensors"))
    sd = {k: v for k, v in sd.items() if not k.startswith(("head.", "fc_norm.", "rope."))}
    missing, unexpected = vit.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, ("DINOv3 Eva khoá không khớp", missing[:5], unexpected[:5])
    print(f"Loaded {repo} (DINOv3 Eva, {len(sd)} tensor, {sum(v.numel() for v in sd.values()) / 1e6:.1f}M)")
    if repo in _EVA_CHECKED: return
    _EVA_CHECKED.add(repo)
    try:
        import timm
        tm = timm.create_model(timm_name or repo.split("/")[-1].split(".")[0], pretrained=False, num_classes=0, dynamic_img_size=True)
        tm.load_state_dict(sd, strict=False); tm.eval()
        g0, was = vit.grid, vit.training; vit.eval()
        with torch.no_grad():
            x = torch.randn(1, 3, 256, 128)
            ref = tm.forward_features(x); ours, n_pre, _ = vit.tokens(x)
        vit.grid = g0; vit.train(was)
        err = float((ref - ours).abs().max()); rel = err / float(ref.abs().max())
        print(f"[DINOv3 Eva] so với timm {timm.__version__}: lệch tuyệt đối max {err:.2e}, tương đối {rel:.2e}")
        assert rel < 1e-3, "DINOv3 Eva KHÔNG khớp timm"
        del tm
    except AssertionError: raise
    except Exception as e:
        print(f"[DINOv3 Eva] bỏ qua so sánh timm ({type(e).__name__}: {e})")

# ------------------------------ EVA-02 (BAAI, MIM; timm eva02_large_patch14_224.mim_m38m, MIT) ------------------------------
# 14/10: họ pretrain khác DINO (MIM theo đặc trưng EVA-CLIP) để thay DINOv2-L trong hệ ghép (cỡ 304M, cùng tốc độ). Như timm Eva cấu hình eva02:
# q / k / v tách (k không bias), SwiGLU có LayerNorm trong MLP (scale_mlp, ẩn 2730), không layer scale, cls + pos_embed tuyệt đối (pretrain lưới 16x16,
# nội suy bicubic antialias về lưới train như timm dynamic_img_size), RoPE 2D "cat" xen kẽ chỉ cho patch: theta 10000, lưới ij, toạ độ co về ref 16x16.
# LayerNorm eps 1e-6, chuẩn hoá CLIP. LayerNorm cuối tên "norm" (checkpoint: fc_norm) để LLRD xếp nó lên tầng trên cùng như DINO.
EVA02_CFG = {"eva02_large_m38m": (1024, 24, 16, 2730, "timm/eva02_large_patch14_224.mim_m38m"),   # dim, depth, heads, ẩn MLP, repo
             "eva02_dbg": (64, 3, 2, 170, None)}                                                     # chỉ để chạy thử code trên CPU

def _eva02_rope(gh, gw, head_dim, device, ref=(16, 16), theta=10000.0):
    # = timm build_rotary_pos_embed(in_pixels=False, ref_feat_shape=ref, grid_indexing="ij", grid_offset=0) -> (sin, cos), mỗi cái (N, head_dim)
    nb = head_dim // 4
    bands = 1.0 / (theta ** (torch.arange(nb, device=device, dtype=torch.float32) / nb))
    th = torch.arange(gh, device=device, dtype=torch.float32) / gh * ref[0]
    tw = torch.arange(gw, device=device, dtype=torch.float32) / gw * ref[1]
    pos = (torch.stack(torch.meshgrid(th, tw, indexing="ij"), -1).unsqueeze(-1) * bands).reshape(gh * gw, -1)
    return pos.sin().repeat_interleave(2, -1), pos.cos().repeat_interleave(2, -1)

def _rot_pair(t):    # = timm rot(): [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]
    return torch.stack([-t[..., 1::2], t[..., ::2]], -1).reshape(t.shape)

class _Eva02Attn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__(); self.h = heads
        self.q_proj = nn.Linear(dim, dim); self.k_proj = nn.Linear(dim, dim, bias=False); self.v_proj = nn.Linear(dim, dim); self.proj = nn.Linear(dim, dim)
    def forward(self, x, sin, cos, n_pre):
        B, N, C = x.shape; sh = lambda t: t.reshape(B, N, self.h, C // self.h).transpose(1, 2)
        q, k, v = sh(self.q_proj(x)), sh(self.k_proj(x)), sh(self.v_proj(x))
        sin, cos = sin.to(q.dtype), cos.to(q.dtype)
        rope = lambda t: torch.cat([t[:, :, :n_pre], t[:, :, n_pre:] * cos + _rot_pair(t[:, :, n_pre:]) * sin], 2)
        o = F.scaled_dot_product_attention(rope(q), rope(k).to(q.dtype), v.to(q.dtype))
        return self.proj(o.transpose(1, 2).reshape(B, N, C))

class _SwiGLUNorm(nn.Module):
    def __init__(self, dim, hidden, eps):
        super().__init__(); self.fc1_g = nn.Linear(dim, hidden); self.fc1_x = nn.Linear(dim, hidden); self.norm = nn.LayerNorm(hidden, eps=eps); self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x): return self.fc2(self.norm(F.silu(self.fc1_g(x)) * self.fc1_x(x)))

class _Eva02Block(nn.Module):
    def __init__(self, dim, heads, hidden, dp, eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps); self.attn = _Eva02Attn(dim, heads); self.norm2 = nn.LayerNorm(dim, eps=eps); self.mlp = _SwiGLUNorm(dim, hidden, eps)
        self.dp = dp
    def forward(self, x, sin, cos, n_pre):
        x = x + _drop_path(self.attn(self.norm1(x), sin, cos, n_pre), self.dp, self.training)
        return x + _drop_path(self.mlp(self.norm2(x)), self.dp, self.training)

class Eva02ViT(nn.Module):
    # Cùng giao diện DinoV3Eva (features / forward / grid / out_dim / blocks / norm). pos_embed theo lưới hiện tại (grid); run.py đổi tier bằng
    # nội suy pos_embed như DINOv2 (arch "eva02*" đi nhánh dinov2), RoPE tính lại theo kích thước ảnh vào.
    def __init__(self, dim, depth, heads, hidden, grid, drop_path=0.0, parts=False, patch=14, eps=1e-6):
        super().__init__()
        self.grid, self.parts, self.patch, self.heads, self.ckpt = grid, parts, patch, heads, False
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)); self.pos_embed = nn.Parameter(torch.zeros(1, 1 + grid[0] * grid[1], dim))
        self.patch_embed = nn.Module(); self.patch_embed.proj = nn.Conv2d(3, dim, patch, patch)
        dps = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList([_Eva02Block(dim, heads, hidden, d, eps) for d in dps])
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.out_dim = (2 + 3 * parts) * dim
        self._rope = {}
    def tokens(self, x, taps=()):
        B, _, H, W = x.shape; g = (H // self.patch, W // self.patch)
        assert self.pos_embed.shape[1] == 1 + g[0] * g[1], f"Eva02: pos_embed {tuple(self.pos_embed.shape)} không khớp lưới {g}"
        key = (g, str(x.device))
        if key not in self._rope: self._rope[key] = _eva02_rope(*g, self.norm.normalized_shape[0] // self.heads, x.device)
        sin, cos = self._rope[key]
        x = torch.cat([self.cls_token.expand(B, -1, -1).to(x.dtype), self.patch_embed.proj(x).flatten(2).transpose(1, 2)], 1) + self.pos_embed
        mids = []
        for i, b in enumerate(self.blocks):
            if self.ckpt and self.training and torch.is_grad_enabled() and any(p_.requires_grad for p_ in b.parameters()):
                x = torch.utils.checkpoint.checkpoint(b, x, sin, cos, 1, use_reentrant=False)
            else: x = b(x, sin, cos, 1)
            if i in taps: mids.append(x[:, 1:])
        return self.norm(x), 1, mids
    def features(self, x, taps=()):
        x, n_pre, mids = self.tokens(x, taps); pt = x[:, n_pre:]
        f = torch.cat([x[:, 0], pt.mean(1)] + (_stripe_means(pt, self.grid) if self.parts else []), 1)
        return f, pt, mids
    def forward(self, x, return_tokens=False):
        f, pt, _ = self.features(x)
        return (f, pt) if return_tokens else f

def load_eva02_pretrained(vit, repo, timm_name="eva02_large_patch14_224"):
    # Chỉ dùng lúc TRAIN (cần mạng; repo timm không gated). Nội suy pos_embed 16x16 -> lưới train; so đầu ra với timm (nếu có), fp32 CPU, 1 ảnh.
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    raw = load_file(hf_hub_download(repo, "model.safetensors"))
    sd = {("norm." + k[len("fc_norm."):] if k.startswith("fc_norm.") else k): v for k, v in raw.items() if not k.startswith(("head.", "rope."))}
    pe = sd["pos_embed"]; s0 = int(round((pe.shape[1] - 1) ** 0.5)); gh, gw = vit.grid
    if (s0, s0) != (gh, gw):
        g = F.interpolate(pe[:, 1:].float().reshape(1, s0, s0, -1).permute(0, 3, 1, 2), size=(gh, gw), mode="bicubic", align_corners=False, antialias=True)
        sd["pos_embed"] = torch.cat([pe[:, :1].float(), g.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)], 1).to(pe.dtype)
    missing, unexpected = vit.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, ("EVA-02 khoá không khớp", missing[:5], unexpected[:5])
    print(f"Loaded {repo} (EVA-02, {len(sd)} tensor, {sum(v.numel() for v in sd.values()) / 1e6:.1f}M) | pos_embed {s0}x{s0} -> {gh}x{gw}")
    try:
        import timm
        tm = timm.create_model(timm_name, pretrained=False, num_classes=0, dynamic_img_size=True)
        tm.load_state_dict(raw, strict=False); tm.eval(); was = vit.training; vit.eval()
        with torch.no_grad():
            x = torch.randn(1, 3, gh * vit.patch, gw * vit.patch)
            ref = tm.fc_norm(tm.forward_features(x)); ours = vit.tokens(x)[0]
        vit.train(was); rel = float((ref - ours).abs().max()) / float(ref.abs().max())
        print(f"[EVA-02] so với timm {timm.__version__}: lệch tương đối max {rel:.2e}"); assert rel < 1e-3, "EVA-02 KHÔNG khớp timm"
        del tm
    except AssertionError: raise
    except Exception as e:
        print(f"[EVA-02] bỏ qua so sánh timm ({type(e).__name__}: {e})")

# ------------------------------ C-RADIOv4 (NVIDIA, 01/2026; NVIDIA Open Model License, weights HF không gated) ------------------------------
# 11/10: backbone chưng cất từ SigLIP2-g + DINOv3-7B + SAM3 (IN kNN 86.6 > DINOv3-H+ 85.8). Kiến trúc = timm vit_huge_patch16 (qkv có bias, MLP GELU,
# LN eps 1e-6, không layer scale, KHÔNG có norm cuối) + CPE: pos_embed học được trên lưới 128×128; khi suy luận nội suy song tuyến về
# (max(h,w))² rồi cắt góc trên-trái h×w (đúng ViTPatchGenerator._get_pos_embeddings, nhánh eval). Patch embed = Linear (c, yy, xx) không bias.
# 10 token đầu = 3 CLS theo teacher + 7 register; đặc trưng tóm tắt = CLS ở summary_idxs (2 teacher có summary). Đầu vào chuẩn hoá CLIP OpenAI.
RADIO_CFG = {"radio_v4h": (1280, 32, 16, 5120, 10, 128, "nvidia/C-RADIOv4-H"),      # dim, depth, heads, ẩn MLP, số token đầu, lưới pos, repo
             "radio_dbg": (64, 3, 2, 128, 10, 16, None)}                              # chỉ để chạy thử code trên CPU

class _VitAttn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__(); self.h = heads; self.qkv = nn.Linear(dim, dim * 3); self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4).unbind(0)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C))

class _VitMlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__(); self.fc1 = nn.Linear(dim, hidden); self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x): return self.fc2(F.gelu(self.fc1(x)))

class _VitBlock(nn.Module):
    def __init__(self, dim, heads, hidden, dp, eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps); self.attn = _VitAttn(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=eps); self.mlp = _VitMlp(dim, hidden); self.dp = dp
    def forward(self, x):
        x = x + _drop_path(self.attn(self.norm1(x)), self.dp, self.training)
        return x + _drop_path(self.mlp(self.norm2(x)), self.dp, self.training)

class RadioViT(nn.Module):
    # Cùng giao diện với DinoV3Eva (tokens / features / forward / grid / out_dim / blocks / norm / ckpt).
    def __init__(self, dim, depth, heads, hidden, n_pre, pos_grid, grid, drop_path=0.0, parts=False, patch=16, eps=1e-6, n_sum=2):
        super().__init__()
        self.grid, self.n_pre, self.parts, self.patch, self.pos_grid, self.ckpt = grid, n_pre, parts, patch, pos_grid, False
        self.cls_token = nn.Parameter(torch.zeros(n_pre, dim))                      # 3 CLS theo teacher + 7 register
        self.patch_embed = nn.Linear(3 * patch * patch, dim, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_grid * pos_grid, dim))
        self.register_buffer("summary_idxs", torch.arange(n_sum, dtype=torch.long))
        dps = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList([_VitBlock(dim, heads, hidden, d, eps) for d in dps])
        self.norm = nn.Identity()
        self.out_dim = (n_sum + 1 + 3 * parts) * dim
        self._pos = {}
    def pos(self, gh, gw, dtype):
        # nhánh eval của CPE; khi train tính lại mỗi lần (pos_embed còn học), khi eval lưu đệm theo lưới
        key = (gh, gw, str(self.pos_embed.device), dtype)
        if not self.training and key in self._pos: return self._pos[key]
        P, C = self.pos_grid, self.pos_embed.shape[-1]; m = max(gh, gw)
        pe = self.pos_embed.reshape(1, P, P, C).permute(0, 3, 1, 2)
        if (gh, gw) != (P, P):
            pe = F.interpolate(pe.float(), size=(m, m), mode="bilinear", align_corners=False)[..., :gh, :gw]
        pe = pe.flatten(2).transpose(1, 2).to(dtype)
        if not self.training: self._pos[key] = pe
        else: self._pos.clear()
        return pe
    def tokens(self, x, taps=()):
        B, _, H, W = x.shape; p = self.patch; self.grid = (H // p, W // p)
        t = F.conv2d(x, self.patch_embed.weight.reshape(-1, 3, p, p).to(x.dtype), stride=p).flatten(2).transpose(1, 2)
        t = t + self.pos(*self.grid, t.dtype)
        x = torch.cat([self.cls_token.unsqueeze(0).expand(B, -1, -1).to(t.dtype), t], 1)
        mids = []
        for i, b in enumerate(self.blocks):
            if self.ckpt and self.training and torch.is_grad_enabled() and any(p_.requires_grad for p_ in b.parameters()):
                x = torch.utils.checkpoint.checkpoint(b, x, use_reentrant=False)
            else: x = b(x)
            if i in taps: mids.append(x[:, self.n_pre:])
        return self.norm(x), self.n_pre, mids
    def features(self, x, taps=()):
        x, n_pre, mids = self.tokens(x, taps); pt = x[:, n_pre:]
        f = torch.cat([x[:, self.summary_idxs].flatten(1), pt.mean(1)] + (_stripe_means(pt, self.grid) if self.parts else []), 1)
        return f, pt, mids
    def forward(self, x, return_tokens=False):
        f, pt, _ = self.features(x)
        return (f, pt) if return_tokens else f

_RADIO_CHECKED = set()
def load_radio_pretrained(vit, repo):
    # Chỉ dùng lúc TRAIN (cần mạng; repo không gated). Khoá HF: radio_model.model.{blocks.*, patch_generator.{cls_token.token, embedder.weight, pos_embed}},
    # radio_model.summary_idxs, radio_model.input_conditioner.{norm_mean, norm_std}. Lần nạp đầu: so với code gốc NVIDIA (torch.hub) nếu tải được.
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    raw = load_file(hf_hub_download(repo, "model.safetensors")); sd = {}
    for k, v in raw.items():
        if k.startswith("radio_model.model.blocks."): sd[k[len("radio_model.model."):]] = v
        elif k == "radio_model.model.patch_generator.cls_token.token": sd["cls_token"] = v
        elif k == "radio_model.model.patch_generator.embedder.weight": sd["patch_embed.weight"] = v
        elif k == "radio_model.model.patch_generator.pos_embed": sd["pos_embed"] = v
        elif k == "radio_model.summary_idxs": sd["summary_idxs"] = v.long()
    mean, std = raw["radio_model.input_conditioner.norm_mean"].flatten(), raw["radio_model.input_conditioner.norm_std"].flatten()
    assert torch.allclose(mean, torch.tensor(CLIP_NORM[0]), atol=1e-4) and torch.allclose(std, torch.tensor(CLIP_NORM[1]), atol=1e-4), ("RADIO chuẩn hoá khác CLIP", mean, std)
    missing, unexpected = vit.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, ("RADIO khoá không khớp", missing[:5], unexpected[:5])
    print(f"Loaded {repo} (C-RADIO, {len(sd)} tensor, {sum(v.numel() for v in sd.values()) / 1e6:.1f}M, summary_idxs {sd['summary_idxs'].tolist()})")
    if repo in _RADIO_CHECKED: return
    _RADIO_CHECKED.add(repo)
    try:   # code gốc qua torch.hub (cần timm + einops; bản transformers đòi thêm open_clip). Đã kiểm local 11/10: lệch ≤ 1.1e-5 ở 256/224/192
        ref = torch.hub.load("NVlabs/RADIO", "radio_model", version="c-radio_v4-h", progress=False, skip_validation=True, trust_repo=True)
        ref.load_state_dict({k[len("radio_model."):]: v for k, v in raw.items() if k.startswith("radio_model.")}, strict=False); ref = ref.float().eval()
        g0, was = vit.grid, vit.training; vit.eval()
        with torch.no_grad():
            x = torch.rand(1, 3, 256, 128)
            out = ref(x); summ, feat = (out.summary, out.features) if hasattr(out, "summary") else (out[0], out[1])
            mn, sd_ = torch.tensor(CLIP_NORM[0]).view(1, 3, 1, 1), torch.tensor(CLIP_NORM[1]).view(1, 3, 1, 1)
            tok, n_pre, _ = vit.tokens((x - mn) / sd_)
            ours_s, ours_f = tok[:, vit.summary_idxs].flatten(1), tok[:, n_pre:]
        vit.grid = g0; vit.train(was)
        rel = max(float((summ - ours_s).abs().max() / summ.abs().max()), float((feat - ours_f).abs().max() / feat.abs().max()))
        print(f"[C-RADIO] so với code NVIDIA: lệch tương đối max {rel:.2e}")
        assert rel < 1e-3, "C-RADIO KHÔNG khớp code gốc"
        del ref
    except AssertionError: raise
    except Exception as e:
        print(f"[C-RADIO] bỏ qua so sánh code gốc ({type(e).__name__}: {e})")

# ------------------------------ PE-Core (Meta Perception Encoder, Apache-2.0), khoá khớp timm vit_pe_core_* ------------------------------
# Như timm Eva cấu hình PE: cls + pos_embed tuyệt đối + norm_pre, khối qkv gộp với RoPE 2D kiểu "cat" (xen kẽ, chỉ cho patch, lưới lệch 1,
# quy về lưới pretrain 14×14), MLP GELU, không layer scale; norm cuối; pooling bằng attention (1 latent, 8 head). Patch embed không bias.
PE_CFG = {"pe_core_b16": ("timm/vit_pe_core_base_patch16_224.fb", 768, 12, 12, 16, 14),    # repo, dim, depth, heads, patch, lưới pretrain
          "pe_core_l14": ("timm/vit_pe_core_large_patch14_336.fb", 1024, 24, 16, 14, 24)}
PE_NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def _pe_rope(gh, gw, head_dim, ref, device, temp=10000.0, offset=1.0):
    # = timm build_rotary_pos_embed(in_pixels=False, grid_indexing="xy", grid_offset=1, ref_feat_shape=(ref, ref)) -> sin, cos (N, head_dim)
    nb = head_dim // 4
    bands = 1.0 / temp ** (torch.arange(nb, dtype=torch.float32, device=device) / nb)
    tx = (torch.arange(gw, dtype=torch.float32, device=device) + offset) / gw * ref
    ty = (torch.arange(gh, dtype=torch.float32, device=device) + offset) / gh * ref
    gx, gy = torch.meshgrid(tx, ty, indexing="xy")
    pos = torch.stack([gx, gy], -1).unsqueeze(-1) * bands
    return pos.sin().reshape(gh * gw, -1).repeat_interleave(2, -1), pos.cos().reshape(gh * gw, -1).repeat_interleave(2, -1)

def _pe_rot(x):
    return torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)

class _PEAttn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__(); self.h = heads; self.qkv = nn.Linear(dim, dim * 3); self.proj = nn.Linear(dim, dim)
    def forward(self, x, sin, cos):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4).unbind(0)
        sin, cos = sin.to(q.dtype), cos.to(q.dtype)
        rope = lambda t: torch.cat([t[:, :, :1], t[:, :, 1:] * cos + _pe_rot(t[:, :, 1:]) * sin], 2)   # RoPE chỉ cho patch (bỏ cls)
        o = F.scaled_dot_product_attention(rope(q), rope(k), v)
        return self.proj(o.transpose(1, 2).reshape(B, N, C))

class _PEMlp(nn.Module):
    def __init__(self, dim, hidden): super().__init__(); self.fc1 = nn.Linear(dim, hidden); self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x): return self.fc2(F.gelu(self.fc1(x)))

class _PEBlock(nn.Module):
    def __init__(self, dim, heads, dp):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5); self.attn = _PEAttn(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5); self.mlp = _PEMlp(dim, dim * 4); self.dp = dp
    def forward(self, x, sin, cos):
        x = x + _drop_path(self.attn(self.norm1(x), sin, cos), self.dp, self.training)
        return x + _drop_path(self.mlp(self.norm2(x)), self.dp, self.training)

class _PEPool(nn.Module):
    # = timm AttentionPoolLatent (1 latent, pool "token")
    def __init__(self, dim, heads=8):
        super().__init__(); self.h = heads
        self.latent = nn.Parameter(torch.zeros(1, 1, dim)); self.q = nn.Linear(dim, dim); self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim); self.norm = nn.LayerNorm(dim, eps=1e-5); self.mlp = _PEMlp(dim, dim * 4)
    def forward(self, x):
        B, N, C = x.shape; hd = C // self.h
        q = self.q(self.latent.expand(B, -1, -1)).reshape(B, 1, self.h, hd).transpose(1, 2)
        k, v = self.kv(x).reshape(B, N, 2, self.h, hd).permute(2, 0, 3, 1, 4).unbind(0)
        o = self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, 1, C))
        o = o + self.mlp(self.norm(o))
        return o[:, 0]

class _PEPatch(nn.Module):
    def __init__(self, dim, patch): super().__init__(); self.proj = nn.Conv2d(3, dim, patch, patch, bias=False)
    def forward(self, x): return self.proj(x).flatten(2).transpose(1, 2)

class PECoreViT(nn.Module):
    # Cùng giao diện với DinoV3ViT (features / forward / grid / out_dim / blocks / norm). Đặc trưng = [pooling attention ; TB patch].
    # pos_embed tuyệt đối theo lưới lúc tạo model -> mọi ảnh phải đúng kích thước đó (pipeline luôn resize về IMG).
    def __init__(self, dim, depth, heads, grid, patch=16, ref=14, drop_path=0.0, parts=False):
        super().__init__()
        self.grid, self.patch, self.heads, self.ref, self.parts = tuple(grid), patch, heads, ref, parts
        self.patch_embed = _PEPatch(dim, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + grid[0] * grid[1], dim))
        self.norm_pre = nn.LayerNorm(dim, eps=1e-5)
        dps = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList([_PEBlock(dim, heads, dps[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-5); self.attn_pool = _PEPool(dim)
        self.out_dim = dim * (2 + (3 if parts else 0)); self._rope = {}
    def features(self, x, taps=()):
        B, _, H, W = x.shape; g = (H // self.patch, W // self.patch)
        assert g == self.grid, f"PE-Core: lưới ảnh {g} khác lưới lúc tạo model {self.grid}"
        key = str(x.device)
        if key not in self._rope: self._rope[key] = _pe_rope(*g, self.pos_embed.shape[-1] // self.heads, self.ref, x.device)
        sin, cos = self._rope[key]
        x = torch.cat([self.cls_token.expand(B, -1, -1), self.patch_embed(x)], 1) + self.pos_embed
        x = self.norm_pre(x); mids = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, sin, cos)
            if i in taps: mids.append(x[:, 1:])
        x = self.norm(x); pt = x[:, 1:]
        f = torch.cat([self.attn_pool(x), pt.mean(1)] + (_stripe_means(pt, self.grid) if self.parts else []), 1)
        return f, pt, mids
    def forward(self, x, return_tokens=False):
        f, pt, _ = self.features(x)
        return (f, pt) if return_tokens else f

def load_pe_pretrained(vit, repo):
    # Chỉ dùng lúc TRAIN (cần mạng). pos_embed pretrain (lưới 14×14) nội suy bicubic + antialias sang lưới hiện tại như timm resample_abs_pos_embed.
    import os
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    sd = load_file(hf_hub_download(repo, "model.safetensors", token=os.environ.get("HF_TOKEN") or None))
    sd = {k: v for k, v in sd.items() if not k.startswith(("head.", "fc_norm."))}
    pe = sd["pos_embed"]; n0 = int(round((pe.shape[1] - 1) ** 0.5)); gh, gw = vit.grid
    if (n0, n0) != (gh, gw):
        g = pe[:, 1:].reshape(1, n0, n0, -1).permute(0, 3, 1, 2).float()
        g = F.interpolate(g, size=(gh, gw), mode="bicubic", antialias=True, align_corners=False)
        sd["pos_embed"] = torch.cat([pe[:, :1].float(), g.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)], 1).to(pe.dtype)
    missing, unexpected = vit.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, ("PE-Core khoá không khớp", missing[:5], unexpected[:5])
    print(f"Loaded {repo} (PE-Core, {len(sd)} tensor, pos_embed {n0}x{n0} -> {gh}x{gw})")

# ------------------------------ E12: nhánh màu ------------------------------
# Lý do: DINOv2 được pretrain với ColorJitter(hue 0.1, saturation 0.2, p 0.8) + 20 % ảnh xám, nên đặc trưng cuối được dạy để bất biến
# với màu (L tăng mọi nhóm trừ màu). Hai nhánh cộng thêm logit cho 24 thuộc tính màu, khởi tạo bằng 0 (lúc đầu = model gốc).
UB_COLOR = list(range(8, 20))       # UpperBody-Color-* theo thứ tự 40 thuộc tính của BTC
LB_COLOR = list(range(21, 33))      # LowerBody-Color-*

class MidColorHead(nn.Module):
    # Head màu từ patch token của các khối giữa (còn giữ thông tin màu), gộp theo dải thân trên / thân dưới
    def __init__(self, dim, n_taps, n_attr=40):
        super().__init__()
        self.n_attr = n_attr
        self.norm_u, self.norm_l = nn.LayerNorm(dim * n_taps), nn.LayerNorm(dim * n_taps)
        self.ub, self.lb = nn.Linear(dim * n_taps, len(UB_COLOR)), nn.Linear(dim * n_taps, len(LB_COLOR))
        for l_ in (self.ub, self.lb): nn.init.zeros_(l_.weight); nn.init.zeros_(l_.bias)
        self.register_buffer("iu", torch.tensor(UB_COLOR), persistent=False)
        self.register_buffer("il", torch.tensor(LB_COLOR), persistent=False)
    def forward(self, mids, grid):
        st = [_stripe_means(t, grid) for t in mids]
        u = self.ub(self.norm_u(torch.cat([a[1] for a in st], 1))); l_ = self.lb(self.norm_l(torch.cat([a[2] for a in st], 1)))
        return u.new_zeros(u.shape[0], self.n_attr).index_copy(1, self.iu, u).index_copy(1, self.il, l_)

class ColorStats(nn.Module):
    # Thống kê màu tường minh: histogram mềm trên lưới k×k×k màu RGB của pixel thật (đã bỏ chuẩn hoá), vùng áo và vùng quần
    # (bỏ 20 % mỗi bên để bớt nền), qua MLP nhỏ. Rẻ: ảnh được avg-pool 4×4 trước khi tính.
    def __init__(self, mean, std, k=4, hidden=128, n_attr=40):
        super().__init__()
        g = torch.linspace(0, 1, k)
        self.register_buffer("c", torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3), persistent=False)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.inv_tau, self.n_attr = 2.0 * (k - 1) ** 2, n_attr            # 1 / (2σ²), σ = nửa khoảng cách lưới
        self.mu = nn.Sequential(nn.Linear(k ** 3, hidden), nn.GELU(), nn.Linear(hidden, len(UB_COLOR)))
        self.ml = nn.Sequential(nn.Linear(k ** 3, hidden), nn.GELU(), nn.Linear(hidden, len(LB_COLOR)))
        for l_ in (self.mu[2], self.ml[2]): nn.init.zeros_(l_.weight); nn.init.zeros_(l_.bias)
        self.register_buffer("iu", torch.tensor(UB_COLOR), persistent=False)
        self.register_buffer("il", torch.tensor(LB_COLOR), persistent=False)
    def hist(self, img):
        px = img.flatten(2).transpose(1, 2)                                 # (B, P, 3)
        d2 = ((px.unsqueeze(2) - self.c) ** 2).sum(-1)                      # (B, P, k³)
        return torch.softmax(-d2 * self.inv_tau, -1).mean(1).sqrt()
    def forward(self, x):
        img = F.avg_pool2d((x.float() * self.std + self.mean).clamp(0, 1), 4)
        H, W = img.shape[-2:]; c0 = int(W * 0.2); c1 = max(int(W * 0.8), c0 + 1)
        u = self.mu(self.hist(img[:, :, int(H * 0.2):max(int(H * 0.6), int(H * 0.2) + 1), c0:c1]))
        l_ = self.ml(self.hist(img[:, :, int(H * 0.5):, c0:c1]))
        return u.new_zeros(u.shape[0], self.n_attr).index_copy(1, self.iu, u).index_copy(1, self.il, l_.to(u.dtype))

# ------------------------------ Mô hình PAR ------------------------------
def patch_of(arch):
    return (CLIP_CFG[arch][4] if arch.startswith("clip") else SIGLIP_CFG[arch][4] if arch.startswith("siglip")
            else PE_CFG[arch][4] if arch.startswith("pe_core") else 16 if arch.startswith(("dinov3", "radio")) else 14)

class DinoPartMasker(nn.Module):
    # E27 (H8): DINO đã fine-tune, đóng băng, khoanh vùng áo / quần cho model khác. Logit màu theo từng patch fc([CLS ; token_i]);
    # điểm vùng = max logit 12 màu áo / quần; mask = softmax(điểm / τ) trong khung dọc (áo 10–70 %, quần 40–100 % chiều cao).
    # fixed=True: mask đều trong cùng khung (đối chứng, không cần DINO). Ảnh vào đã chuẩn hoá theo model chính, đổi sang đầu vào của DINO.
    def __init__(self, pm, in_norm):
        super().__init__()
        self.tau, self.fixed = float(pm.get("TAU", 1.5)), bool(pm.get("FIXED", False))
        self.size = (int(pm.get("src_h", 256)), int(pm.get("src_w", 128)))
        self.net = None if self.fixed else PARNet(pm["src_arch"], pretrained=False, img_h=self.size[0], img_w=self.size[1])
        sm = pm.get("src_mean", IMAGENET_NORM[0]); ss = pm.get("src_std", IMAGENET_NORM[1])
        for n_, v in (("mi", in_norm[0]), ("si", in_norm[1]), ("md", sm), ("sd", ss)):
            self.register_buffer(n_, torch.tensor(v, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
    def train(self, mode=True):
        return super().train(False)                      # luôn ở chế độ eval (không dropout / drop path)
    @torch.no_grad()
    def forward(self, x, grid):
        B = x.shape[0]
        if self.fixed:
            GH, GW = 16, 8; su = sl = x.new_zeros(B, GH * GW, dtype=torch.float32)
        else:
            xd = F.interpolate(x.float() * self.si + self.mi, size=self.size, mode="bilinear", align_corners=False)
            fe, pt, _ = self.net.backbone.features((xd - self.md) / self.sd)
            GH, GW = self.net.backbone.grid; N, C = pt.shape[1], pt.shape[2]
            zi = self.net.fc(torch.cat([fe[:, :C].unsqueeze(1).expand(B, N, C), pt], 2)).float()
            su, sl = zi[..., UB_COLOR].amax(-1), zi[..., LB_COLOR].amax(-1)
        rows = (torch.arange(GH, device=x.device, dtype=torch.float32) / GH).view(GH, 1).expand(GH, GW).reshape(-1)
        pu = torch.where((rows >= 0.10) & (rows < 0.70), 0.0, -1e4); pl = torch.where(rows >= 0.40, 0.0, -1e4)
        def shrink(m_):
            m_ = F.adaptive_avg_pool2d(m_.view(B, 1, GH, GW), tuple(grid)).flatten(1)
            return m_ / m_.sum(1, keepdim=True).clamp(min=1e-12)
        return shrink(torch.softmax(su / self.tau + pu, 1)), shrink(torch.softmax(sl / self.tau + pl, 1))

class PartMaskColorHead(nn.Module):
    # E27: token của model chính gộp theo mask áo / quần -> logit 12 màu áo / 12 màu quần (lớp cuối khởi tạo 0 = lúc đầu như model gốc)
    def __init__(self, dim, n_src, n_attr=40):
        super().__init__()
        self.n_attr = n_attr
        self.nu, self.nl = nn.LayerNorm(dim * n_src), nn.LayerNorm(dim * n_src)
        self.ub, self.lb = nn.Linear(dim * n_src, len(UB_COLOR)), nn.Linear(dim * n_src, len(LB_COLOR))
        for l_ in (self.ub, self.lb): nn.init.zeros_(l_.weight); nn.init.zeros_(l_.bias)
        self.register_buffer("iu", torch.tensor(UB_COLOR), persistent=False)
        self.register_buffer("il", torch.tensor(LB_COLOR), persistent=False)
    def forward(self, toks, mu, ml):
        u = torch.cat([torch.einsum("bn,bnc->bc", mu.to(t.dtype), t) for t in toks], 1).float()
        l_ = torch.cat([torch.einsum("bn,bnc->bc", ml.to(t.dtype), t) for t in toks], 1).float()
        u, l_ = self.ub(self.nu(u)), self.lb(self.nl(l_))
        return u.new_zeros(u.shape[0], self.n_attr).index_copy(1, self.iu, u).index_copy(1, self.il, l_)

class PARNet(nn.Module):
    def __init__(self, arch="dinov2_vitb14_reg", num_attr=40, pretrained=False, drop=0.2,
                 img_h=252, img_w=126, drop_path=0.0, parts=False, qdec=False, cmid=(), chist=False, pmask=None, qdec2=None):
        super().__init__()
        self.arch, self.parts = arch, parts
        self.qdec = None                                   # K11: QueryDecoder nếu qdec=True
        self.qdec2 = None                                  # 14/10 QDEC2 (dict cấu hình): đầu truy vấn kép riêng, chỉ chạy khi forward(return_q=True)
        assert not qdec2 or arch.startswith(("dinov2", "dinov3", "pe_core", "radio", "eva02")), "QDEC2: cần backbone có features(x, taps)"
        self.cmid, self.chead, self.cstats = tuple(cmid), None, None      # E12: nhánh màu (chỉ DINOv2)
        self.pmasker, self.phead, self.ptaps = None, None, tuple((pmask or {}).get("TAPS", ()))   # E27: màu theo vùng do DINO khoanh
        if arch.startswith(("dinov2", "dinov3", "pe_core", "radio", "eva02")):
            if arch.startswith("pe_core"):                 # PE-Core: patch 16, pos_embed tuyệt đối + RoPE 2D, pooling attention
                repo, dim, depth, heads, patch, ref = PE_CFG[arch]
                self.backbone = PECoreViT(dim, depth, heads, (img_h // patch, img_w // patch), patch, ref, drop_path, parts)
                if pretrained: load_pe_pretrained(self.backbone, repo)
            elif arch in RADIO_CFG:                        # C-RADIOv4 (11/10): ViT-H/16 + CPE, 10 token đầu, chuẩn hoá CLIP
                dim, depth, heads, hidden, n_pre, pgrid, repo = RADIO_CFG[arch]
                self.backbone = RadioViT(dim, depth, heads, hidden, n_pre, pgrid, (img_h // 16, img_w // 16), drop_path, parts)
                if pretrained and repo: load_radio_pretrained(self.backbone, repo)
            elif arch in EVA02_CFG:                        # 14/10 EVA-02 (MIM, khoá timm, patch 14, chuẩn hoá CLIP)
                dim, depth, heads, hidden, repo = EVA02_CFG[arch]
                self.backbone = Eva02ViT(dim, depth, heads, hidden, (img_h // 14, img_w // 14), drop_path, parts)
                if pretrained and repo: load_eva02_pretrained(self.backbone, repo)
            elif arch in DINO3_EVA_CFG:                    # DINOv3 ViT-H+ (khoá timm)
                dim, depth, heads, hidden, repo = DINO3_EVA_CFG[arch]
                self.backbone = DinoV3Eva(dim, depth, heads, hidden, (img_h // 16, img_w // 16), 4, drop_path, parts)
                if pretrained and repo: load_dinov3_eva_pretrained(self.backbone, repo)
            elif arch.startswith("dinov3"):                # DINOv3: patch 16, RoPE, 4 register
                dim, depth, heads, repo = DINO3_CFG[arch]
                self.backbone = DinoV3ViT(dim, depth, heads, (img_h // 16, img_w // 16), 4, drop_path, parts)
                if pretrained: load_dinov3_pretrained(self.backbone, repo)
            else:
                name = arch.replace("_reg", ""); n_reg = 4 if arch.endswith("_reg") else 0
                dim, depth, heads = DINO_CFG[name]
                self.backbone = DinoViT(dim, depth, heads, (img_h // 14, img_w // 14), n_reg, 14, drop_path, parts)
                if pretrained: load_dinov2_pretrained(self.backbone, name, n_reg)
            self.fc = nn.Sequential(nn.LayerNorm(self.backbone.out_dim), nn.Dropout(drop),
                                    nn.Linear(self.backbone.out_dim, num_attr))
            if qdec: self.qdec = QueryDecoder(dim, num_attr, dim // 64)
            if qdec2:
                assert not (cmid or pmask), "QDEC2 chưa ghép với CMID / PMASK (cùng dùng token khối giữa / nhánh forward riêng)"
                L_ = len(self.backbone.blocks); mid = L_ * 2 // 3 if qdec2.get("MID") is None else qdec2["MID"]
                assert 0 <= mid < L_, f"QDEC2: MID {mid} ngoài 0..{L_ - 1}"
                with torch.random.fork_rng(devices=[]):    # không rút RNG CPU: dòng ngẫu nhiên phía sau (dữ liệu, augmentation) như bản không QDEC2
                    self.qdec2 = QueryDecoder2(dim, mid, qdec2.get("DIM", 256), qdec2.get("HEADS", 8), num_attr)
            if cmid: self.chead = MidColorHead(dim, len(cmid), num_attr)
            if chist: self.cstats = ColorStats(*norm_for(arch), n_attr=num_attr)
            if pmask:
                assert arch.startswith("pe_core") or not self.ptaps, "E27: token khối giữa chỉ có ở PE-Core / DINO"
                self.pmasker, self.phead = DinoPartMasker(pmask, norm_for(arch)), PartMaskColorHead(dim, 1 + len(self.ptaps), num_attr)
            return
        if arch.startswith("clip"):
            _, dim, depth, heads, patch, pdim = CLIP_CFG[arch]
            self.backbone = ClipViT(dim, depth, heads, (img_h // patch, img_w // patch), patch, pdim, drop_path, parts)
            if pretrained: load_clip_pretrained(self.backbone, arch)
            # Không LayerNorm ở head: phần đầu của feature là embedding CLIP đã chuẩn hoá, head khởi tạo bằng text embedding
            self.fc = nn.Sequential(nn.Identity(), nn.Dropout(drop), nn.Linear(self.backbone.out_dim, num_attr))
            if qdec: self.qdec = QueryDecoder(dim, num_attr, dim // 64)
            if pmask:
                assert not self.ptaps, "E27: CLIP chỉ dùng token khối cuối"
                self.pmasker, self.phead = DinoPartMasker(pmask, norm_for(arch)), PartMaskColorHead(dim, 1, num_attr)
            return
        if arch.startswith("siglip"):
            _, dim, depth, heads, patch = SIGLIP_CFG[arch][:5]; hid = SIGLIP_CFG[arch][5] if len(SIGLIP_CFG[arch]) > 5 else None
            self.backbone = SiglipViT(dim, depth, heads, (img_h // patch, img_w // patch), patch, drop_path, parts, hid)
            if pretrained and SIGLIP_CFG[arch][0]: load_siglip_pretrained(self.backbone, arch)
            self.fc = nn.Sequential(nn.Identity(), nn.Dropout(drop), nn.Linear(self.backbone.out_dim, num_attr))
            if qdec: self.qdec = QueryDecoder(dim, num_attr, dim // 64)
            return
        m = getattr(torchvision.models, arch)(weights="DEFAULT" if pretrained else None)
        if arch.startswith("convnext"):
            dim = m.classifier[2].in_features
            m.classifier[2] = nn.Sequential(nn.Dropout(drop), nn.Linear(dim, num_attr))
        elif arch.startswith("resnet"):
            dim = m.fc.in_features
            m.fc = nn.Sequential(nn.Dropout(drop), nn.Linear(dim, num_attr))
        else:
            raise ValueError(arch)
        self.net = m

    @property
    def head(self):
        if self.arch.startswith(("dinov2", "dinov3", "clip", "siglip", "pe_core", "radio", "eva02")): return self.fc[2]
        return self.net.classifier[2][1] if self.arch.startswith("convnext") else self.net.fc[1]

    def forward(self, x, return_feat=False, return_q=False):
        # return_feat=True: trả thêm (đặc trưng gộp, patch token cuối) cho chưng cất đặc trưng (E11); mặc định chỉ trả logit như cũ
        # return_q=True (QDEC2): thêm z_q ở CUỐI tuple -> (z, z_q) hoặc (z, f, tok, z_q); z không đổi
        if self.arch.startswith(("dinov2", "dinov3", "clip", "siglip", "pe_core", "radio", "eva02")):
            mids = None
            if self.phead is not None:                     # E27: token gộp theo mask vùng của DINO
                if self.ptaps: f, tok, mids = self.backbone.features(x, self.ptaps)
                else: f, tok = self.backbone(x, return_tokens=True)
                z = self.fc(f) + self.phead([tok] + list(mids or []), *self.pmasker(x, self.backbone.grid))
                return (z, f, tok) if return_feat else z
            if self.chead is not None: f, tok, mids = self.backbone.features(x, self.cmid)
            elif return_q: f, tok, mids = self.backbone.features(x, (self.qdec2.mid,))
            elif self.qdec is not None or return_feat: f, tok = self.backbone(x, return_tokens=True)
            else: f, tok = self.backbone(x), None
            z = self.fc(f)
            if self.qdec is not None: z = z + self.qdec(tok)
            if self.chead is not None: z = z + self.chead(mids, self.backbone.grid)
            if self.cstats is not None: z = z + self.cstats(x)
            if return_q:
                zq = self.qdec2([tok, mids[0]], self.backbone.grid)
                return (z, f, tok, zq) if return_feat else (z, zq)
            return (z, f, tok) if return_feat else z
        return self.net(x)
'''
(CODE / "model.py").write_text(MODEL_PY, encoding="utf-8")
sys.path.insert(0, str(CODE))
import model as upar_model; importlib.reload(upar_model)
from model import PARNet, preprocess, norm_for, patch_of, crop_frac
assert upar_model.QD2_GROUP == [-1 if a == "Gender-Female" else 2 if a.startswith("LowerBody") else 1 if a.startswith(("UpperBody", "Accessory-Backpack", "Accessory-Bag"))
                                else 0 for a in ATTR], "QDEC2: QD2_GROUP lệch thứ tự thuộc tính"   # 0 = tuổi / tóc / kính / mũ

def open_rgb(path, tries=7):
    # 12/10: Input Kaggle (ổ gắn qua mạng) đôi khi báo FileNotFoundError / OSError tạm thời giữa chừng (L3 clean chết ở epoch 4 với ảnh
    # đã đọc được 3 epoch trước) -> đọc lại có chờ (0.5 .. 32 s), hết lượt mới báo lỗi.
    for k in range(tries):
        try:
            with Image.open(path) as im: return im.convert("RGB")
        except OSError:
            if k == tries - 1: raise
            time.sleep(0.5 * 2 ** k)

# %% [markdown]
# ## 4. Text embedding cho head CLIP

# %%
TEMPLATES = ["a photo of {}.", "a low resolution surveillance photo of {}.", "a blurry cropped photo of {}.",
             "a photo of a pedestrian, {}."]
COLORS = ["Black", "Blue", "Brown", "Green", "Grey", "Orange", "Pink", "Purple", "Red", "White", "Yellow", "Other"]
CW = {c: c.lower() for c in COLORS}; CW["Other"] = "multicolored"

def attribute_phrases():
    P = {
        "Age-Young": ("a child or a teenager", ["an adult", "an elderly person"]),
        "Age-Adult": ("an adult", ["a child or a teenager", "an elderly person"]),
        "Age-Old": ("an elderly person", ["a child or a teenager", "an adult"]),
        "Gender-Female": ("a woman", ["a man"]),
        "Hair-Length-Short": ("a person with short hair", ["a person with long hair", "a bald person"]),
        "Hair-Length-Long": ("a person with long hair", ["a person with short hair", "a bald person"]),
        "Hair-Length-Bald": ("a bald person", ["a person with short hair", "a person with long hair"]),
        "UpperBody-Length-Short": ("a person wearing a short-sleeved top", ["a person wearing a long-sleeved top"]),
        "LowerBody-Length-Short": ("a person wearing shorts or a short skirt", ["a person wearing long trousers"]),
        "LowerBody-Type-Trousers&Shorts": ("a person wearing trousers or shorts", ["a person wearing a skirt or a dress"]),
        "LowerBody-Type-Skirt&Dress": ("a person wearing a skirt or a dress", ["a person wearing trousers or shorts"]),
        "Accessory-Backpack": ("a person carrying a backpack", ["a person"]),
        "Accessory-Bag": ("a person carrying a handbag or a shoulder bag", ["a person"]),
        "Accessory-Glasses-Normal": ("a person wearing glasses", ["a person"]),
        "Accessory-Glasses-Sun": ("a person wearing sunglasses", ["a person wearing glasses", "a person"]),
        "Accessory-Hat": ("a person wearing a hat", ["a person"]),
    }
    for c in COLORS:
        P[f"UpperBody-Color-{c}"] = (f"a person wearing a {CW[c]} top", [f"a person wearing a {CW[o]} top" for o in COLORS if o != c])
        P[f"LowerBody-Color-{c}"] = (f"a person wearing {CW[c]} trousers", [f"a person wearing {CW[o]} trousers" for o in COLORS if o != c])
    assert set(P) == set(ATTR), set(ATTR) ^ set(P)
    return P

def clip_text_head(arch, scale):
    # W[a] = scale * (e(câu dương) - trung bình e(câu âm)), e = text embedding CLIP đã chuẩn hoá (trung bình nhiều template)
    import torch
    from transformers import AutoModel, AutoTokenizer
    sig = arch.startswith("siglip")
    name = (upar_model.SIGLIP_CFG if sig else upar_model.CLIP_CFG)[arch][0]
    m = AutoModel.from_pretrained(name).eval(); tok = AutoTokenizer.from_pretrained(name)
    pad = dict(padding="max_length", max_length=64, truncation=True) if sig else dict(padding=True)
    cache = {}
    @torch.no_grad()
    def emb(phrase):
        if phrase not in cache:
            t = tok([tp.format(phrase) for tp in TEMPLATES], return_tensors="pt", **pad)
            o = m.get_text_features(**t)
            o = o if torch.is_tensor(o) else (o.pooler_output if hasattr(o, 'pooler_output') else o[0])
            e = torch.nn.functional.normalize(o.float(), dim=-1).mean(0)
            cache[phrase] = torch.nn.functional.normalize(e, dim=0)
        return cache[phrase]
    P = attribute_phrases()
    W = torch.stack([emb(P[a][0]) - torch.stack([emb(n) for n in P[a][1]]).mean(0) for a in ATTR]) * scale
    del m; gc.collect()
    return W

# %% [markdown]
# ## 5. Dataset, augmentation, loss, metric, calibrate, cross-fit

# %%
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True
seed_all(CFG["SEED"])
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if IS_KAGGLE and not CFG["DEBUG"]:   # 14/10: DEBUG (smoke Colab CPU) được chạy CPU
    assert dev.type == "cuda", "Không có GPU: Settings -> Accelerator -> GPU T4 x2"
print("Device:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] or "CPU", "| torch", torch.__version__)

class LowRes:
    '''Thu nhỏ về chiều rộng ngẫu nhiên rồi phóng lại: mô phỏng crop giám sát độ phân giải thấp.'''
    def __init__(self, p, wmin=20, wmax=64): self.p, self.wmin, self.wmax = p, wmin, wmax
    def __call__(self, img):
        if random.random() > self.p or img.width <= self.wmin: return img
        w = random.randint(self.wmin, min(self.wmax, img.width)); h = max(1, round(img.height * w / img.width))
        return img.resize((w, h), random.choice([0, 2, 3])).resize(img.size, random.choice([2, 3]))

class RandJPEG:
    def __init__(self, p, qmin=25, qmax=90): self.p, self.qmin, self.qmax = p, qmin, qmax
    def __call__(self, img):
        if random.random() > self.p: return img
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=random.randint(self.qmin, self.qmax)); buf.seek(0)
        return Image.open(buf).convert("RGB")

class MotionBlur:
    """Mờ chuyển động 1 hướng ngẫu nhiên (ngang / dọc / chéo), kernel 3 hoặc 5: mô phỏng video giám sát."""
    def __init__(self, p): self.p = p
    def __call__(self, img):
        if random.random() > self.p: return img
        from PIL import ImageFilter
        k = random.choice([3, 5]); d = random.choice(["h", "v", "d1", "d2"]); w = [0.0] * (k * k)
        for i in range(k):
            r, c = {"h": (k // 2, i), "v": (i, k // 2), "d1": (i, i), "d2": (i, k - 1 - i)}[d]
            w[r * k + c] = 1.0
        return img.filter(ImageFilter.Kernel((k, k), w, scale=k))

class CropJitter:
    """Cắt một khung con ngẫu nhiên (cao/rộng độc lập trong [cmin, 1]) rồi mới resize: đổi tỷ lệ khung và cách crop như detector khác."""
    def __init__(self, p, cmin=0.8): self.p, self.cmin = p, cmin
    def __call__(self, img):
        if random.random() > self.p: return img
        W0, H0 = img.size; sh, sw = random.uniform(self.cmin, 1), random.uniform(self.cmin, 1)
        h, w = max(1, int(H0 * sh)), max(1, int(W0 * sw)); y, x = random.randint(0, H0 - h), random.randint(0, W0 - w)
        return img.crop((x, y, x + w, y + h))

class WBJitter:
    """Nhân mỗi kênh màu một hệ số ngẫu nhiên nhỏ (mô phỏng ám màu của camera khác)."""
    def __init__(self, p, r=0.15): self.p, self.r = p, r
    def __call__(self, t):
        if random.random() > self.p: return t
        g = torch.empty(3).uniform_(1 - self.r, 1 + self.r); g = g / g.mean()
        return (t * g[:, None, None]).clamp(0, 1)

class GrayWorld:
    def __call__(self, t): return upar_model.grayworld(t)

import torchvision.transforms.functional as TF
def _kelvin_rgb(k):
    # màu RGB của vật đen ở nhiệt độ k (xấp xỉ Tanner Helland, 1000–40000 K), thang 0..1
    t = k / 100.0
    r = 1.0 if t <= 66 else min(1.0, max(0.0, 329.698727446 * (t - 60) ** -0.1332047592 / 255))
    g = (min(1.0, max(0.0, (99.4708025861 * math.log(t) - 161.1195681661) / 255)) if t <= 66
         else min(1.0, max(0.0, 288.1221695283 * (t - 60) ** -0.0755148492 / 255)))
    b = 1.0 if t >= 66 else (0.0 if t <= 19 else min(1.0, max(0.0, (138.5177312231 * math.log(t - 10) - 305.0447927307) / 255)))
    return np.array([r, g, b], np.float32)

class CTempJitter:
    """E31: lệch đèn kiểu camera thật — ánh sáng nhiệt độ màu K (quỹ tích Planck, đèn vàng 2 500 K .. trời râm 7 500 K) mà camera cân bằng cho 6 500 K,
    cộng lệch xanh lá / tím nhẹ (đèn huỳnh quang) và gamma phơi sáng. Giữ nguyên nhãn (nhãn MEVID làm từ ảnh check-in sáng rõ: phải đọc đúng màu thật)."""
    def __init__(self, p, kr=(2500, 7500), gamma=(0.75, 1.35), tint=0.06):
        self.p, self.kr, self.gamma, self.tint = p, kr, gamma, tint; self.ref = _kelvin_rgb(6500)
    def __call__(self, t):
        if random.random() > self.p: return t
        g = _kelvin_rgb(math.exp(random.uniform(math.log(self.kr[0]), math.log(self.kr[1])))) / self.ref
        g[1] *= 1 + random.uniform(-self.tint, self.tint); g = g / g.mean()
        t = (t * torch.as_tensor(g)[:, None, None]).clamp(0, 1)
        return t.pow(random.uniform(*self.gamma))

class TruncOcc:
    """E31: che / cắt giữ nhãn — MEVID gán nhãn theo bộ trang phục nên phần bị che (dưới hông, cạnh khung) vẫn dương.
    Cắt bỏ 10–40 % đáy ảnh (người bị che chân / khung detector cụt) hoặc một dải cạnh, KHÔNG đổi nhãn."""
    def __init__(self, p, fr=(0.1, 0.4), side_p=0.3, side=(0.1, 0.25)): self.p, self.fr, self.side_p, self.side = p, fr, side_p, side
    def __call__(self, img):
        if random.random() > self.p: return img
        W0, H0 = img.size
        if random.random() < self.side_p:
            c = int(W0 * random.uniform(*self.side)); return img.crop((c, 0, W0, H0)) if random.random() < 0.5 else img.crop((0, 0, W0 - c, H0))
        return img.crop((0, 0, W0, max(8, int(H0 * (1 - random.uniform(*self.fr))))))

class MiniAugMix:
    """E31: AugMix (Hendrycks 2020) chỉ với phép KHÔNG phá màu (xoay / xô / dịch / tự cân tương phản / san bằng / posterize 6–8 bit / độ nét).
    Trộn 3 chuỗi (sâu 1–3) theo Dirichlet(1), rồi trộn với ảnh gốc theo Beta(1, 1). Bỏ solarize / color / invert vì 24/40 thuộc tính là màu."""
    def __init__(self, p, width=3): self.p, self.width = p, width
    @staticmethod
    def _op(img):
        o = random.randint(0, 8); W0, H0 = img.size; m = random.uniform(0.1, 1.0)
        if o == 0: return TF.affine(img, 0, (0, 0), 1.0, (random.choice([-1, 1]) * 15 * m, 0))
        if o == 1: return TF.affine(img, 0, (0, 0), 1.0, (0, random.choice([-1, 1]) * 15 * m))
        if o == 2: return TF.affine(img, 0, (int(random.choice([-1, 1]) * W0 * 0.15 * m), 0), 1.0, (0, 0))
        if o == 3: return TF.affine(img, 0, (0, int(random.choice([-1, 1]) * H0 * 0.1 * m)), 1.0, (0, 0))
        if o == 4: return TF.rotate(img, random.choice([-1, 1]) * 15 * m)
        if o == 5: return TF.autocontrast(img)
        if o == 6: return TF.equalize(img)
        if o == 7: return TF.posterize(img, random.randint(6, 8))
        return TF.adjust_sharpness(img, 1 + random.choice([-1, 1]) * 0.9 * m)
    def __call__(self, img):
        if random.random() > self.p: return img
        ws = np.random.dirichlet([1.0] * self.width); mix = np.zeros((img.height, img.width, 3), np.float32)
        for w_ in ws:
            x = img
            for _ in range(random.randint(1, 3)): x = self._op(x)
            mix += w_ * np.asarray(x, np.float32)
        b = np.random.beta(1.0, 1.0)
        return Image.fromarray(np.clip((1 - b) * np.asarray(img, np.float32) + b * mix, 0, 255).astype(np.uint8))

class LetterboxResize:
    def __init__(self, H, W, mean): self.H, self.W, self.mean = H, W, mean
    def __call__(self, img): return upar_model.letterbox(img, self.H, self.W, self.mean)

class MixStyle:
    """K10: forward hook trên đầu ra một block ViT, chỉ khi train. Trộn mean/std (theo token) của từng kênh giữa các ảnh
    trong batch: model bớt phụ thuộc "phong cách" camera / dataset (Zhou et al., ICLR 2021)."""
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6): self.p, self.beta, self.eps = p, torch.distributions.Beta(alpha, alpha), eps
    def __call__(self, mod, inp, out):
        if not mod.training or out.shape[0] < 2 or random.random() > self.p: return None
        x = out.float(); mu = x.mean(1, keepdim=True).detach(); sig = (x.var(1, keepdim=True) + self.eps).sqrt().detach()
        lam = self.beta.sample((x.shape[0], 1, 1)).to(x.device); perm = torch.randperm(x.shape[0], device=x.device)
        x = (x - mu) / sig * (lam * sig + (1 - lam) * sig[perm]) + lam * mu + (1 - lam) * mu[perm]
        return x.to(out.dtype)

class GaussNoise:
    """E24: nhiễu cảm biến (tensor [0, 1]), σ ngẫu nhiên trong khoảng cho trước."""
    def __init__(self, p, smin=0.01, smax=0.05): self.p, self.smin, self.smax = p, smin, smax
    def __call__(self, t):
        if random.random() > self.p: return t
        return (t + torch.randn_like(t) * random.uniform(self.smin, self.smax)).clamp(0, 1)

def make_train_tf(H, W, mean, std, aug=None, wb=False):
    # Mặc định giống hệt pipeline cũ; các khoá E24 (LOWRES_W, JPEG_Q, CJ_*, HUE, GRAY_P, NOISE_*, ERASE_P, ...) chỉ thêm phép biến đổi khi được bật
    a = dict(LOWRES_P=CFG["AUG_LOWRES_P"], JPEG_P=CFG["AUG_JPEG_P"], MOTION_P=0.0, CROP_P=0.0, CROP_MIN=0.8, WB_P=0.0,
             LOWRES_W=(20, 64), JPEG_Q=(25, 90), CJ_P=0.8, BRIGHT=0.25, CONTRAST=0.25, SAT=0.1, HUE=0.0, GRAY_P=0.0,
             BLUR_P=0.15, NOISE_P=0.0, NOISE_S=(0.01, 0.05), WB_R=0.15, ERASE_P=0.3,
             CT_P=0.0, CT_K=(2500, 7500), CT_GAMMA=(0.75, 1.35), TRUNC_P=0.0, TRUNC=(0.1, 0.4), AUGMIX_P=0.0, LBOX=False,   # E31 (11/10)
             SHARP_P=0.0, SHARP_F=2.0)   # 15/10: làm nét sau resize (ảnh gốc lớn bị THU xuống thì sắc hơn ảnh nhỏ bị phóng) — mặc định tắt
    a.update(aug or {})
    return T.Compose([
        CropJitter(a["CROP_P"], a["CROP_MIN"]), *([TruncOcc(a["TRUNC_P"], a["TRUNC"])] if a["TRUNC_P"] > 0 else []),
        LowRes(a["LOWRES_P"], *a["LOWRES_W"]), MotionBlur(a["MOTION_P"]), RandJPEG(a["JPEG_P"], *a["JPEG_Q"]),
        LetterboxResize(H, W, mean) if a["LBOX"] else T.Resize((H, W)), *([MiniAugMix(a["AUGMIX_P"])] if a["AUGMIX_P"] > 0 else []),
        *([T.RandomAdjustSharpness(a["SHARP_F"], p=a["SHARP_P"])] if a["SHARP_P"] > 0 else []), T.RandomHorizontalFlip(),
        T.Pad(8, padding_mode="edge"), T.RandomCrop((H, W)),
        T.RandomApply([T.ColorJitter(brightness=a["BRIGHT"], contrast=a["CONTRAST"], saturation=a["SAT"], hue=a["HUE"])], p=a["CJ_P"]),  # mặc định không đổi hue: 24/40 thuộc tính là màu
        *([T.RandomGrayscale(a["GRAY_P"])] if a["GRAY_P"] > 0 else []),
        T.RandomApply([T.GaussianBlur(5, (0.1, 1.2))], p=a["BLUR_P"]),
        T.ToTensor(), *([GaussNoise(a["NOISE_P"], *a["NOISE_S"])] if a["NOISE_P"] > 0 else []),
        *([CTempJitter(a["CT_P"], a["CT_K"], a["CT_GAMMA"])] if a["CT_P"] > 0 else []), WBJitter(a["WB_P"], a["WB_R"]), *([GrayWorld()] if wb else []), T.Normalize(mean, std),
        T.RandomErasing(p=a["ERASE_P"], scale=(0.02, 0.12), value="random"),
    ])

# ---- E24: augmentation theo nhóm thuộc tính + đổi màu có đổi nhãn (cân bằng màu hiếm) ----
# Nhóm theo "cần nhìn gì": H = chi tiết nhỏ ở đầu (kính, mũ, tóc), C = 24 màu áo / quần, P = toàn thân (tuổi, giới, dáng, túi)
AGRP = np.array(["C" if "-Color-" in a else "H" if a.startswith(("Hair", "Accessory-Glasses", "Accessory-Hat")) else "P" for a in ATTR])
def grp_mask(g): return np.isin(AGRP, list(g)).astype(np.float32)

def _rgb2hsv(a):
    r, g, b = a[..., 0], a[..., 1], a[..., 2]; mx = a.max(-1); mn = a.min(-1); d = mx - mn
    h = np.zeros_like(mx); nz = d > 1e-6
    rm = nz & (mx == r); gm = nz & (mx == g) & ~rm; bm = nz & ~rm & ~gm
    h[rm] = ((g - b)[rm] / d[rm]) % 6; h[gm] = (b - r)[gm] / d[gm] + 2; h[bm] = (r - g)[bm] / d[bm] + 4
    return np.stack([h * 60, np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0), mx], -1)
def _hsv2rgb(x):
    h, s, v = x[..., 0] / 60 % 6, x[..., 1], x[..., 2]; c = v * s; xx = c * (1 - np.abs(h % 2 - 1)); m0 = v - c
    z = np.zeros_like(h); i = np.clip(np.floor(h).astype(int), 0, 5)
    r = np.choose(i, [c, xx, z, z, xx, c]); g = np.choose(i, [xx, c, c, xx, z, z]); b = np.choose(i, [z, z, xx, c, c, xx])
    return np.stack([r + m0, g + m0, b + m0], -1)

class Recolor:
    """E24: đổi màu áo HOẶC quần bằng xoay hue trong dải dọc của vùng, chỉ trên điểm ảnh gần màu nguồn, rồi đổi nhãn màu theo.
    Nguồn chỉ là Blue / Green / Purple (xa màu da); vùng phải có đúng 1 màu; áo và quần không cùng màu nguồn; điểm ảnh được tô
    phải phủ >= COVER dải; màu đích chọn theo V trung bình (vd Yellow cần sáng, Brown cần tối) và phải được dataset gán nhãn.
    Đích được bốc ∝ tần suất^(-POW) -> tạo thêm mẫu dương cho màu hiếm (LB-Purple 167 ảnh, LB-Orange 259, ...)."""
    HUE = dict(Red=0, Orange=28, Yellow=55, Green=125, Blue=220, Purple=280, Pink=335, Brown=25)
    # v / s: khoảng V, S trung bình của vùng nguồn để kết quả thật sự là màu đích (vd áo xanh nhạt S thấp xoay sang đỏ sẽ ra hồng -> chỉ cho Pink)
    TGT = dict(Red=dict(v=(0.25, 1), s=(0.35, 1)), Green=dict(v=(0.2, 1)), Blue=dict(v=(0.2, 1)), Purple=dict(v=(0.2, 1)),
               Yellow=dict(v=(0.45, 1), s=(0.3, 1), sv=(1.2, 1.35)), Orange=dict(v=(0.4, 1), s=(0.35, 1), sv=(1.3, 1.25)),
               Pink=dict(v=(0.4, 1), sv=(0.55, 1.35)), Brown=dict(v=(0.22, 0.6), s=(0.3, 1), sv=(1.0, 0.75)))
    SRC = ("Blue", "Green", "Purple")
    BAND = dict(UpperBody=(0.10, 0.68), LowerBody=(0.48, 1.0))
    def __init__(self, cfg, pos_rate):
        self.p, self.cover, self.smin, self.vmin = cfg.get("P", 0.5), cfg.get("COVER", 0.22), cfg.get("SMIN", 0.25), cfg.get("VMIN", 0.22)
        self.reg = {}
        for reg in ("UpperBody", "LowerBody"):
            idx = {c: ATTR.index(f"{reg}-Color-{c}") for c in self.HUE}
            self.reg[reg] = dict(idx=idx, all=[j for j, a in enumerate(ATTR) if a.startswith(f"{reg}-Color-")],
                                 w={c: float(pos_rate[j]) ** -cfg.get("POW", 0.5) for c, j in idx.items()})
    def __call__(self, img, y, m):
        if random.random() > self.p: return img, y
        regs = ["UpperBody", "LowerBody"]; random.shuffle(regs)
        for reg in regs:
            R = self.reg[reg]; pos = [j for j in R["all"] if y[j] > 0.5]
            src = next((c for c in self.SRC if pos == [R["idx"][c]]), None)
            if src is None or m[R["idx"][src]] == 0: continue
            oth = self.reg["LowerBody" if reg == "UpperBody" else "UpperBody"]
            if y[oth["idx"][src]] > 0.5: continue                # áo và quần cùng màu nguồn: dải chồng lấn sẽ tô nhầm vùng kia
            a = np.asarray(img, np.float32) / 255; Hh = a.shape[0]; hsv = _rgb2hsv(a)
            r0, r1 = int(self.BAND[reg][0] * Hh), int(self.BAND[reg][1] * Hh)
            dh = np.abs((hsv[..., 0] - self.HUE[src] + 180) % 360 - 180)
            mk = (dh < 40) & (hsv[..., 1] > 0.15) & (hsv[..., 2] > 0.08); mk[:r0] = False; mk[r1:] = False
            if r1 <= r0 or mk[r0:r1].mean() < self.cover: continue
            S_, V_ = hsv[..., 1][mk].mean(), hsv[..., 2][mk].mean()
            if S_ < self.smin or V_ < self.vmin: continue        # vùng quá xám / quá tối: đổi hue không ra màu rõ
            ok = [t for t, c in self.TGT.items() if t != src and c["v"][0] <= V_ <= c["v"][1] and c.get("s", (0, 1))[0] <= S_ <= c.get("s", (0, 1))[1]
                  and m[R["idx"][t]] > 0]
            if not ok: continue
            tgt = random.choices(ok, weights=[R["w"][t] for t in ok])[0]
            sv = self.TGT[tgt].get("sv", (1.0, 1.0)); rot = self.HUE[tgt] - self.HUE[src] + random.uniform(-6, 6)
            hsv[..., 0] = np.where(mk, (hsv[..., 0] + rot) % 360, hsv[..., 0])
            hsv[..., 1] = np.where(mk, np.clip(hsv[..., 1] * sv[0], 0, 1), hsv[..., 1])
            hsv[..., 2] = np.where(mk, np.clip(hsv[..., 2] * sv[1], 0, 1), hsv[..., 2])
            y = y.copy(); y[R["idx"][src]] = 0.0; y[R["idx"][tgt]] = 1.0
            return Image.fromarray((np.clip(_hsv2rgb(hsv), 0, 1) * 255 + 0.5).astype(np.uint8)), y
        return img, y

# ---- E35 (12/10): ảnh train phục hồi offline + fidelity gate ----
_RS_CACHE = {}
_RS_UB = [i for i, a in enumerate(ATTR) if a.startswith("UpperBody-Color-")]
_RS_LB = [i for i, a in enumerate(ATTR) if a.startswith("LowerBody-Color-")]
_RS_H = [i for i, a in enumerate(ATTR) if a.startswith(("Hair", "Accessory-Glasses", "Accessory-Hat"))]
def restore_manifest(method):
    """Bảng ảnh phục hồi của 1 phương pháp: restore_<pp>.csv (+ restored_<pp>.zip) = output upar_restore_gen.ipynb (Add Input; nhiều shard thì gộp).
    Zip chưa giải nén -> giải nén vào WORK/restored. Thêm cột path = đường dẫn tuyệt đối của ảnh phục hồi."""
    if method in _RS_CACHE: return _RS_CACHE[method]
    roots = [Path(p) for p in os.environ.get("UPAR_RESTORE_DIRS", "").split(os.pathsep) if p] + [WORK] + list(input_dirs())
    # 12/10: nhiều shard trong 1 dataset -> restore_<pp>_<tag>.csv đi cặp với restored_<pp>_<tag>.zip
    csvs = list(dict.fromkeys(c for d in roots for c in sorted(d.glob(f"restore_{method}*.csv")) if c.stem in (f"restore_{method}",) or c.stem.startswith(f"restore_{method}_")))
    assert csvs, f"E35: không thấy restore_{method}.csv trong Input (Add Input output của notebook upar_restore_gen)"
    parts = []
    for k_, c in enumerate(csvs):
        suf = c.stem[len(f"restore_{method}"):]; d, base, zp = c.parent, None, c.parent / f"restored_{method}{suf}.zip"
        for cand in [d / f"restored_{method}{suf}", zp, d]:          # thư mục sẵn / Kaggle tự giải nén zip của dataset upload
            if (cand / method).is_dir(): base = cand; break
        if base is None and zp.is_file():
            base = (Path("/tmp/upar_restored") if IS_KAGGLE else WORK / "restored") / f"{method}_{k_}"   # ngoài /kaggle/working: không vào Output
            if not (base / ".ok").exists():
                t0 = time.time(); zipfile.ZipFile(zp).extractall(base); (base / ".ok").touch()
                print(f"E35: giải nén {zp} -> {base} ({time.time() - t0:.0f}s)", flush=True)
        assert base is not None, f"E35: có {c} nhưng không thấy ảnh phục hồi ({method}/ hoặc restored_{method}.zip) cạnh nó"
        mp = pd.read_csv(c); mp["path"] = [str(base / f) for f in mp.file]; parts.append(mp)
    man = pd.concat(parts, ignore_index=True).drop_duplicates(["image", "s"]).reset_index(drop=True)
    miss = [p_ for p_ in man.path.sample(min(50, len(man)), random_state=0) if not Path(p_).exists()]
    assert not miss, f"E35: thiếu file ảnh phục hồi, vd {miss[:2]}"
    print(f"E35 {method}: {man.image.nunique()} ảnh phục hồi ({len(man)} mẫu) từ {len(csvs)} bảng", flush=True)
    _RS_CACHE[method] = man; return man

def restore_lookup(frame, rs):
    """-> danh sách theo từng ảnh của frame: [(đường dẫn ảnh phục hồi, cổng 40 chiều: 1 = nhãn / consistency còn dùng được)], rỗng = không có."""
    if not rs.get("METHOD"): return [[] for _ in range(len(frame))]
    man = restore_manifest(rs["METHOD"]); g = np.ones((len(man), 40), np.float32); gt = rs.get("GATE")
    if gt:
        q = gt.get("Q", 0.9)                        # chỉ che khi vừa thuộc nhóm tệ nhất, vừa lệch thật (ngưỡng tuyệt đối)
        t_up, t_lo = max(man.dc_up.quantile(q), gt.get("DE_MIN", 3.0)), max(man.dc_low.quantile(q), gt.get("DE_MIN", 3.0))
        t_h = min(man.ps_head.quantile(1 - q), gt.get("PS_MAX", 25.0))
        bu, bl, bh = man.dc_up.values > t_up, man.dc_low.values > t_lo, man.ps_head.values < t_h
        g[np.ix_(bu, _RS_UB)] = 0; g[np.ix_(bl, _RS_LB)] = 0; g[np.ix_(bh, _RS_H)] = 0
        print(f"E35 cổng: che màu áo khi ΔE > {t_up:.2f} ({bu.mean():.1%}), màu quần khi ΔE > {t_lo:.2f} ({bl.mean():.1%}), "
              f"tóc / kính / mũ khi PSNR đầu < {t_h:.1f} dB ({bh.mean():.1%})", flush=True)
    look = {}
    for k, (im, p_) in enumerate(zip(man.image.values, man.path.values)): look.setdefault(im, []).append((p_, g[k]))
    items = [look.get(im, []) for im in frame.image.values]
    cov = np.mean([bool(x) for x in items])
    print(f"E35 {rs['METHOD']}: {cov:.1%} ảnh train có bản phục hồi | " + ", ".join(
          f"{d} {np.mean([bool(items[i]) for i in np.where(frame.ds.values == d)[0]]):.0%}" for d in sorted(frame.ds.unique())), flush=True)
    assert cov > 0, "E35: không ảnh train nào có bản phục hồi (sai SUBSET khi sinh ảnh?)"
    return items

# 17/10 BGAUG: tăng cường nền — mặt nạ người DeepLabV3 (research/bg_masks.py, 224x112, packbits) cho ảnh train; khi train ngẫu nhiên thay NỀN
# (ngoài mặt nạ) bằng màu trơn / bản mờ của chính ảnh / nền ảnh khác, nhãn giữ nguyên -> model học bỏ qua cảnh xung quanh. Không đổi suy luận.
CUR_BGAUG = None; _BGM = {}
def bg_masks():
    if not _BGM:
        mt = next((d / "masks_train.npy" for d in input_dirs() if (d / "masks_train.npy").exists()), None) if Path("/kaggle/input").exists() else None   # chỉ mục, không glob đệ quy
        if mt is None and os.environ.get("UPAR_BG_MASKS"): mt = Path(os.environ["UPAR_BG_MASKS"]) / "masks_train.npy"
        assert mt is not None and mt.exists(), "BGAUG: không thấy masks_train.npy (Input dataset mặt nạ)"
        for split in ("train", "val"):
            f = mt.parent / f"masks_{split}.npy"
            if not f.exists(): continue
            M = np.load(f, mmap_mode="r")
            # mặt nạ tạo theo thứ tự ảnh của chú thích RESPLIT 0710 -> đọc đúng file đó (LODO / split gốc dùng ANN khác, khớp theo đường dẫn ảnh)
            ra = [q for q in [ANN] + sorted(d.parent for d in (input_dirs() if Path("/kaggle/input").exists() else []) if d.name == "train" and d.parent.name == "task1")
                  if (q / split / "gt.csv").exists() and "resplit" in str(q).lower() and "resplitclean" not in str(q).lower()]
            if os.environ.get("UPAR_BG_ANN"): ra = [Path(os.environ["UPAR_BG_ANN"])]
            assert ra, "BGAUG: không thấy chú thích resplit_0710 để khớp mặt nạ"
            rows = pd.read_csv(ra[0] / split / "gt.csv", usecols=[0]).iloc[:, 0].tolist()
            assert len(rows) == len(M), (split, str(ra[0]), len(rows), len(M))
            _BGM.update({r: (M, i) for i, r in enumerate(rows)})
        print(f"[BGAUG] mặt nạ người: {len(_BGM)} ảnh từ {mt.parent}", flush=True)
        if "TR" in globals(): print(f"[BGAUG] phủ ảnh train hiện tại: {np.mean([r in _BGM for r in TR.image]) * 100:.1f} %", flush=True)
    return _BGM

class PARDataset(Dataset):
    def __init__(self, frame, H, W, mean, std, train, aug=None, wb=False, soft=None, ym=None, gaug=None, recolor=None, crop=None, lbox=False, hswap=None,
                 restore=None):
        self.rs = restore if train else None           # E35: dict(items, P, PAIR) từ restore_lookup
        self.paths = [str(DATA / p) for p in frame.image]; self.crop = tuple(crop) if crop else (); self.lbox = bool(lbox)
        self.rel = list(frame.image); self.bg = dict(CUR_BGAUG) if (train and CUR_BGAUG) else None
        if self.bg: bg_masks()
        self.y = frame[ATTR].values.astype(np.float32)
        self.m = MASK[frame.ds.map(DS_ID).values]
        if ym is not None: self.y, self.m = ym                 # E8: nhãn / mask đã thống nhất quy ước
        self.d = frame.ds.map(DS_ID).values.astype(np.int64)
        self.soft = soft.astype(np.float32) if soft is not None else np.zeros((len(frame), 40), np.float32)
        self.H, self.W, self.mean, self.std, self.wb = H, W, mean, std, wb
        self.tf = make_train_tf(H, W, mean, std, aug, wb) if train else None
        # E24: mỗi ảnh bốc 1 chính sách augmentation; loss (và KD) chỉ tính trên nhóm thuộc tính mà chính sách đó KHÔNG làm hỏng nhãn
        self.pol = [(make_train_tf(H, W, mean, std, {**(aug or {}), **p.get("AUG", {})}, wb), grp_mask(p.get("LOSS", "HPC")),
                     grp_mask(p.get("KD", p.get("LOSS", "HPC")))) for p in gaug] if (train and gaug) else None
        self.pw = [p.get("P", 1.0) for p in gaug] if self.pol else None
        self.rc = Recolor(recolor, ((self.y * self.m).sum(0) / np.maximum(self.m.sum(0), 1)).clip(1e-4, 1)) if (train and recolor) else None
        self.ones = np.ones(40, np.float32)
        # E34 (12/10): ghép nửa trên ảnh i + nửa dưới ảnh j CÙNG dataset; j bốc ∝ (tần suất màu quần hiếm nhất của j)^-POW -> thêm mẫu màu quần hiếm
        # trong nhiều bối cảnh áo / người khác. Nhãn: áo, tóc, kính, mũ theo i; LowerBody-* theo j; tuổi, giới, ba lô, túi bị che loss (người lai).
        self.hs = dict(hswap) if (train and hswap) else None
        if self.hs:
            lbc = [j for j, a in enumerate(ATTR) if a.startswith("LowerBody-Color-")]
            pr = ((self.y * self.m).sum(0) / np.maximum(self.m.sum(0), 1)).clip(1e-4, 1)
            wj = np.maximum((self.y[:, lbc] * pr[lbc][None, :] ** -self.hs.get("POW", 0.5)).max(1), 1.0)
            self.hs.update(lb=np.array([j for j, a in enumerate(ATTR) if a.startswith("LowerBody-")]),
                           gl=np.array([ATTR.index(a) for a in ("Age-Young", "Age-Adult", "Age-Old", "Gender-Female", "Accessory-Backpack", "Accessory-Bag")]))
            self.hs_idx = {d: np.where(self.d == d)[0] for d in np.unique(self.d)}
            self.hs_cum = {d: np.cumsum(wj[ix]) for d, ix in self.hs_idx.items()}
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = crop_frac(open_rgb(self.paths[i]), self.crop); y, m, km = self.y[i], self.m[i], self.ones
        if self.bg and not self.crop and random.random() < self.bg.get("P", 0.4):
            hit = _BGM.get(self.rel[i])
            if hit is not None:
                M_, r_ = hit; mk = np.unpackbits(M_[r_])[:224 * 112].reshape(224, 112)
                if mk.mean() > 0.08:                               # bỏ qua ảnh gần như không thấy người (mặt nạ hỏng)
                    msk = Image.fromarray(mk * 255).resize(img.size, Image.NEAREST).filter(ImageFilter.MaxFilter(5))   # nới 2 px giữ mép người / túi
                    mode = random.choices(("solid", "blur", "swap"), weights=self.bg.get("W", (1, 1, 1)))[0]
                    if mode == "solid": bgi = Image.new("RGB", img.size, tuple(random.randint(0, 255) for _ in range(3)))
                    elif mode == "blur": bgi = img.filter(ImageFilter.GaussianBlur(radius=max(4, img.size[0] // 8)))
                    else: bgi = open_rgb(self.paths[random.randrange(len(self.paths))]).resize(img.size, Image.BILINEAR)
                    img = Image.composite(img, bgi, msk)
        if self.hs and random.random() < self.hs.get("P", 0.3):
            d = self.d[i]; cum = self.hs_cum[d]
            j = int(self.hs_idx[d][min(bisect.bisect(cum, random.random() * cum[-1]), len(cum) - 1)])
            if j != i:
                im2 = crop_frac(open_rgb(self.paths[j]), self.crop).resize(img.size, Image.BILINEAR)
                cut = int(img.size[1] * random.uniform(*self.hs.get("CUT", (0.47, 0.55))))
                img = img.copy(); img.paste(im2.crop((0, cut, img.size[0], img.size[1])), (0, cut))
                y, m = y.copy(), m.copy(); lb, gl = self.hs["lb"], self.hs["gl"]
                y[lb] = self.y[j][lb]; m[lb] = self.m[j][lb]; m[gl] = 0
        g2 = None
        if self.rs is not None:                        # E35
            its = self.rs["items"][i]
            if self.rs["PAIR"]:                        # view 2 = ảnh phục hồi (không có thì ảnh gốc, augment lần nữa)
                if its: p2, g2 = random.choice(its); img2 = crop_frac(open_rgb(p2), self.crop)
                else: img2, g2 = img, self.ones
            elif its and random.random() < self.rs["P"]:
                p2, g_ = random.choice(its); img = crop_frac(open_rgb(p2), self.crop); m = m * g_
        if self.rc: img, y = self.rc(img, y, m)
        if self.pol:
            tf, lm, km = self.pol[random.choices(range(len(self.pol)), weights=self.pw)[0]]; x = tf(img); m = m * lm
        else:
            x = self.tf(img) if self.tf else preprocess(img, self.H, self.W, self.mean, self.std, self.wb, self.lbox)
        if g2 is not None: return x, y, m, self.d[i], self.soft[i], km, self.tf(img2), g2
        return x, y, m, self.d[i], self.soft[i], km

def _worker_init(_): torch.set_num_threads(1)

class PairBatchSampler(torch.utils.data.Sampler):
    """E36 (12/10): mỗi batch dành 2K chỗ ĐẦU cho K cặp ảnh Market cùng người (4 số đầu tên file) KHÁC camera (cX), xếp [a1, b1, a2, b2, ...];
    batch - 2K chỗ còn lại bốc như WeightedRandomSampler (trọng số w). Vòng train lấy z[0:2K:2] vs z[1:2K:2] để tính loss nhất quán.
    MEVID: video nhiều camera, nhãn theo bộ trang phục -> dự đoán của cùng người phải bất biến theo camera / góc nhìn / che khuất."""
    def __init__(self, frame, batch, k, w, seed=42):
        self.batch, self.k, self.n = batch, k, len(frame) // batch
        self.w = torch.as_tensor(w / w.sum(), dtype=torch.double); self.g = torch.Generator().manual_seed(seed)
        mk = frame.ds.values == "Market1501"; fn = frame.image.str.rsplit("/", n=1).str[-1]
        pid, cam = fn.str[:4].values, fn.str.extract(r"_c(\d)")[0].values
        grp = {}
        for i in np.where(mk)[0]:
            if pid[i].startswith(("-1", "0000")) or not isinstance(cam[i], str): continue      # ảnh rác / nhiễu của Market
            grp.setdefault(pid[i], {}).setdefault(cam[i], []).append(i)
        self.ids = [g for g in grp.values() if len(g) >= 2]                 # người có ảnh ở >= 2 camera
        print(f"E36 cặp khác camera: {len(self.ids)} người Market, K = {k} cặp / batch {batch}", flush=True)
    def __len__(self): return self.n
    def __iter__(self):
        rng = random.Random(int(torch.randint(0, 2**31 - 1, (1,), generator=self.g)))
        for _ in range(self.n):
            out = []
            for g in rng.sample(self.ids, self.k):
                c1, c2 = rng.sample(sorted(g), 2); out += [rng.choice(g[c1]), rng.choice(g[c2])]
            out += torch.multinomial(self.w, self.batch - 2 * self.k, replacement=True, generator=self.g).tolist()
            yield out


class BalancedBatchSampler(torch.utils.data.Sampler):
    """E33 (11/10): mỗi batch chia đều theo nhóm, số batch / epoch như cũ (len(frame) // batch).
    MODE "ds": batch chia đều cho các dataset (dư thì bốc ngẫu nhiên dataset nhận thêm).
    MODE "attr" (class-aware sampling, Shen et al. ECCV'16): mỗi chỗ trong batch thuộc về 1 thuộc tính, tỷ lệ chỗ của thuộc tính a ∝ n_a^POW
    (POW 0 = 40 thuộc tính bằng nhau, 0.5 = căn bậc hai, 1 ≈ tự nhiên), rồi bốc 1 ảnh có nhãn dương a (ô có gán nhãn ở dataset đó);
    MAX_REP: mỗi ảnh trong nhóm bị bốc tối đa chừng này lần / epoch (kỳ vọng).
    Trong mỗi nhóm, ảnh được bốc theo hoán vị xoay vòng (hết vòng mới lặp lại) để lặp đều, không bốc trùng ngẫu nhiên."""
    def __init__(self, frame, batch, mode, pow_=0.0, seed=0, max_rep=None):
        self.batch, self.mode, self.n_batches, self.seed, self.ep = batch, mode, max(1, len(frame) // batch), seed, 0
        if mode == "ds":
            self.names = sorted(frame.ds.unique()); self.pools = [np.where(frame.ds.values == d)[0] for d in self.names]
            self.q = np.full(len(self.pools), 1.0 / len(self.pools))
        elif mode == "attr":
            Y, Mk = frame[ATTR].values.astype(np.float64), MASK[frame.ds.map(DS_ID).values]
            pos = (Y * Mk) > 0; n = pos.sum(0); keep = np.where(n > 0)[0]
            self.names = [ATTR[j] for j in keep]; self.pools = [np.where(pos[:, j])[0] for j in keep]
            q = n[keep].astype(np.float64) ** pow_; q = q / q.sum()
            if max_rep:                # chặn hệ số lặp / epoch (fold LODO bỏ PA100k: LB-Orange chỉ 8 ảnh -> lặp 112×), phần thừa chia lại cho nhóm khác
                cap = max_rep * n[keep] / (self.n_batches * batch)
                for _ in range(50):
                    over = q > cap
                    if not over.any(): break
                    free = q[over].sum() - cap[over].sum(); q[over] = cap[over]; q[~over] += free * q[~over] / q[~over].sum()
            self.q = q
        else: raise ValueError(f"SAMPLER MODE {mode!r}")
        exp_n = self.q * self.n_batches * batch                       # số lần bốc kỳ vọng mỗi epoch theo nhóm
        rep = exp_n / np.array([len(p) for p in self.pools])
        print(f"E33 sampler {mode}" + (f" POW {pow_}" if mode == "attr" else "") + f": {len(self.pools)} nhóm, mỗi batch {batch} ảnh, "
              f"hệ số lặp ảnh trong nhóm min {rep.min():.2f} / trung vị {np.median(rep):.2f} / max {rep.max():.1f} "
              f"({self.names[int(rep.argmax())]}, {len(self.pools[int(rep.argmax())])} ảnh)", flush=True)
        if mode == "attr":
            cnt = np.zeros(len(frame)); share = np.zeros(len(frame))
            for p_, e_ in zip(self.pools, exp_n): cnt[p_] += 1; share[p_] += e_ / len(p_)
            dsn = frame.ds.values; tot = share.sum()
            print("  tỷ lệ dataset kỳ vọng trong batch: " + ", ".join(f"{d} {share[dsn == d].sum() / tot:.2f}" for d in sorted(set(dsn)))
                  + f" | ảnh không thuộc nhóm nào: {(cnt == 0).sum()}", flush=True)
    def __len__(self): return self.n_batches
    def __iter__(self):
        rng = np.random.default_rng(self.seed + 7919 * self.ep); self.ep += 1
        perm = [rng.permutation(p) for p in self.pools]; ptr = [0] * len(self.pools)
        def take(g, k):
            out = []
            while k > 0:
                if ptr[g] >= len(perm[g]): perm[g] = rng.permutation(self.pools[g]); ptr[g] = 0
                s = perm[g][ptr[g]:ptr[g] + k]; ptr[g] += len(s); k -= len(s); out.append(s)
            return out
        for _ in range(self.n_batches):
            base = np.floor(self.q * self.batch).astype(int)          # phần chia đều cố định trong mỗi batch
            rest = self.batch - base.sum()
            if rest > 0: base += rng.multinomial(rest, self.q)       # phần dư bốc theo đúng tỷ lệ
            idx = np.concatenate([s for g in range(len(self.pools)) if base[g] for s in take(g, base[g])])
            yield rng.permutation(idx).tolist()

def make_loader(frame, H, W, mean, std, train, batch, aug=None, wb=False, rfs_t=None, soft=None, ym=None, gaug=None, recolor=None, crop=None, lbox=False,
                ds_bal=None, sampler=None, hswap=None, restore=None, idpair=None):
    kw = dict(num_workers=CFG["WORKERS"], worker_init_fn=_worker_init, pin_memory=dev.type == "cuda")
    if CFG["WORKERS"] > 0: kw.update(prefetch_factor=2, persistent_workers=train)
    ds = PARDataset(frame, H, W, mean, std, train, aug, wb, soft, ym, gaug, recolor, crop, lbox, hswap, restore)
    if not train: return DataLoader(ds, batch_size=batch, shuffle=False, **kw)
    if sampler:                   # E33: batch chia đều theo dataset / theo thuộc tính (thay WeightedRandomSampler, không kèm RFS)
        bs_ = BalancedBatchSampler(frame, batch, sampler["MODE"], sampler.get("POW", 0.0), CFG["SEED"], sampler.get("MAX_REP"))
        return DataLoader(ds, batch_sampler=bs_, **kw)
    n_ds = frame.ds.map(frame.ds.value_counts()).values.astype(np.float64)
    w = n_ds ** (-(CFG["DS_BALANCE"] if ds_bal is None else ds_bal))       # E33: DS_BAL riêng biến thể (0 = tự nhiên, 1 = đều theo dataset)
    if rfs_t:     # K4: lặp ảnh có thuộc tính hiếm, hệ số max(1, sqrt(t / tần suất)) theo thuộc tính hiếm nhất của ảnh
        Y, Mk = frame[ATTR].values.astype(np.float64), MASK[frame.ds.map(DS_ID).values]
        f = ((Y * Mk).sum(0) / np.maximum(Mk.sum(0), 1)).clip(1e-6, 1)
        r = np.maximum(1.0, np.sqrt(rfs_t / f))[None, :] * (Y * Mk > 0)
        w = w * np.maximum(1.0, r.max(1))
        print(f"RFS t={rfs_t}: hệ số lặp trung bình {np.maximum(1.0, r.max(1)).mean():.2f}, tối đa {r.max():.1f}")
    if idpair:                    # E36: cặp cùng người khác camera đặt đầu batch
        return DataLoader(ds, batch_sampler=PairBatchSampler(frame, batch, idpair.get("K", 8), w, CFG["SEED"]), **kw)
    sampler = WeightedRandomSampler(torch.as_tensor(w / w.sum()), num_samples=len(frame), replacement=True)
    return DataLoader(ds, batch_size=batch, sampler=sampler, drop_last=True, **kw)

class MaskedLoss(nn.Module):
    def __init__(self, pos_rate, smooth=0.0, pow_=1.0, asl=None, soft_w=False):
        super().__init__()
        pr = torch.as_tensor(pos_rate)                  # K13: pow_ < 1 làm trọng số wBCE bớt dồn về thuộc tính hiếm
        self.register_buffer("wp", torch.exp((1 - pr) * pow_)); self.register_buffer("wn", torch.exp(pr * pow_))
        self.smooth = smooth
        self.soft_w = bool(soft_w)                      # T2 (13/10): nhãn mềm -> trọng số wBCE nội suy y·wp + (1 − y)·wn (y ∈ {0, 1}: y hệt cách cũ)
        # E29 ASL (Ridnik et al. ICCV'21): -t (1-p)^γ+ log p - (1-t) p_m^γ- log(1-p_m), p_m = max(p - clip, 0); vẫn nhân trọng số wBCE (pow 0 = tắt)
        self.asl = None if not asl else (float(asl.get("GP", 0.0)), float(asl.get("GN", 4.0)), float(asl.get("CLIP", 0.05)))
    def forward(self, logits, y, m, floor=None):
        t = y * (1 - self.smooth) + 0.5 * self.smooth
        if floor is not None: t = torch.maximum(t, floor)          # K12: nhãn âm nhưng gần màu đúng -> đích > 0
        z = logits.float()
        if self.asl is None:
            l = F.binary_cross_entropy_with_logits(z, t, reduction="none")
        else:
            gp, gn, clip = self.asl; p = torch.sigmoid(z); pm = (p - clip).clamp(min=0) if clip > 0 else p
            l = -t * (1 - p).pow(gp) * F.logsigmoid(z) - (1 - t) * pm.pow(gn) * torch.log((1 - pm).clamp(min=1e-8))
        l = l * ((y * self.wp + (1 - y) * self.wn) if self.soft_w else torch.where(y > 0.5, self.wp, self.wn))
        return (l * m).sum() / m.sum().clamp(min=1)

GROUPS_EXCL = [[0, 1, 2], [33, 34]]           # tuổi, loại quần: đúng 1 nhãn ở Market/PA100k
COLOR_GROUPS = [list(range(8, 20)), list(range(21, 33))]   # màu áo, màu quần: đúng 1 màu ở Market/PA100k (>= 99.8 %), PETA 88 % / 98 %
def group_ce(logits, y, m, groups=GROUPS_EXCL):
    # K5 / K8: cross-entropy trên nhóm loại trừ, chỉ cho ảnh có đúng 1 nhãn dương trong nhóm (và nhãn đó được annotate).
    # Thuộc tính dataset không annotate bị loại khỏi softmax (logit = -1e4), vd Market chỉ annotate 9/12 màu áo.
    tot, cnt = logits.new_zeros(()), 0
    for g in groups:
        yg, mg = y[:, g], m[:, g]
        ok = (yg.sum(1) == 1) & ((yg * mg).sum(1) == 1) & (mg.sum(1) >= 2)
        if ok.any():
            zg = logits[ok][:, g].float().masked_fill(mg[ok] == 0, -1e4)
            tot = tot + F.cross_entropy(zg, yg[ok].argmax(1), reduction="sum"); cnt += int(ok.sum())
    return tot / max(cnt, 1)

def cons_loss(za, zb, g):
    # E35: BCE đối xứng giữa 2 view, mỗi phía học theo xác suất (ngắt gradient) của phía kia; g = thuộc tính qua cổng (cả thuộc tính dataset không gán nhãn)
    pa, pb = torch.sigmoid(za.detach()), torch.sigmoid(zb.detach())
    l = F.binary_cross_entropy_with_logits(za, pb, reduction="none") + F.binary_cross_entropy_with_logits(zb, pa, reduction="none")
    return 0.5 * (l * g).sum() / g.sum().clamp(min=1)

def soft_f1_loss(logits, y, m):
    # K9: 1 - Instance-F1 mềm (xác suất thay cho dự đoán cứng), tính từng ảnh rồi lấy trung bình như scorer; chỉ thuộc tính được annotate
    p, yy = torch.sigmoid(logits.float()) * m, y * m
    return 1 - (2 * (p * yy).sum(1) / (p.sum(1) + yy.sum(1)).clamp(min=1e-6)).mean()

COLOR_SIM = [("Black", "Grey"), ("Grey", "White"), ("Blue", "Black")]
def color_sim_matrix():
    # K12: S[i, j] = 1 nếu 2 màu dễ lẫn (cùng vùng áo hoặc quần)
    S_ = torch.zeros(40, 40)
    for reg in ("UpperBody", "LowerBody"):
        for a, b in COLOR_SIM:
            i, j = ATTR.index(f"{reg}-Color-{a}"), ATTR.index(f"{reg}-Color-{b}"); S_[i, j] = S_[j, i] = 1
    return S_

UB_COL, LB_COL = list(range(8, 20)), list(range(21, 33))     # 12 màu áo / 12 màu quần, phần tử cuối là Other
def label_fix(frame, cf):
    # E8: đưa nhãn màu về quy ước của test (Market / PA100k: mỗi vùng đúng 1 màu, áo hoạ tiết = Other)
    # H1: PETA vùng có >= 2 màu -> không biết là áo hoạ tiết (lẽ ra Other) hay áo khoác (lẽ ra 1 màu): bỏ cả nhóm màu của vùng khỏi loss
    # H2: PETA vùng 1 màu (áo trơn) -> Other = 0 là nhãn hợp lệ (gốc bị mask vì PETA không dùng Other)
    y = frame[ATTR].values.astype(np.float32); m = MASK[frame.ds.map(DS_ID).values].copy()
    peta = (frame.ds == "PETA").values
    for nm, g in (("áo", UB_COL), ("quần", LB_COL)):
        k = y[:, g].sum(1); multi = peta & (k >= 2)
        m[np.ix_(multi, g)] = 0
        if cf.get("PETA_OTHER_NEG", True): m[peta & (k == 1), g[-1]] = 1
        print(f"  label_fix {nm}: PETA {int(multi.sum())} ảnh >= 2 màu bỏ khỏi loss | {int((peta & (k == 1)).sum())} ảnh có thêm nhãn Other = 0")
    return y, m

def age_mask(frame, y=None, m=None):
    # E32 (11/10): gỡ đường tắt "Young = học sinh đồng phục". Bỏ 3 thuộc tính tuổi khỏi loss ở ảnh mà nhãn tuổi tự mâu thuẫn / gắn với nguồn:
    #   PA100k có mẫu áo trắng + đồ ngắn + ba lô (55 % ảnh Young, 0.8 % Adult; học sinh 17 / 18 tuổi cùng đồng phục bị gán khác nhau),
    #   PETA-CUHK (chỉ số 2 233–6 795: 342 / 507 ảnh Young PETA là sinh viên trong bộ con này).
    y = frame[ATTR].values.astype(np.float32) if y is None else y
    m = MASK[frame.ds.map(DS_ID).values].copy() if m is None else m.copy()
    A = {a: i for i, a in enumerate(ATTR)}
    uni = ((frame.ds == "PA100k") & (frame["UpperBody-Color-White"] > 0.5) & (frame["LowerBody-Length-Short"] > 0.5) & (frame["Accessory-Backpack"] > 0.5)).values
    pid = frame.image.str.extract(r"PETA/images/(\d+)")[0].astype(float).values
    cuhk = (frame.ds == "PETA").values & (pid >= 2233) & (pid <= 6795)
    sel = uni | cuhk; m[np.ix_(sel, [A["Age-Young"], A["Age-Adult"], A["Age-Old"]])] = 0
    print(f"  age_mask: bỏ loss tuổi ở {int(uni.sum())} ảnh PA100k đồng phục + {int(cuhk.sum())} ảnh PETA-CUHK")
    return y, m

def clean_mask(frame, y=None, m=None):
    # 18/10 CMASK: đặt m = 0 ở các ô (ảnh, thuộc tính) mà OOF L3 nghi nhãn sai (cờ up | down của E30 bản B: biên > 2 so với ngưỡng, Platt p >= 0.9 / <= 0.1,
    # không hạ dương Backpack / Bag / Glasses / Sunglasses / Hat). Nhãn GIỮ NGUYÊN — ô đó chỉ không đóng góp loss (E30 thì kéo nhãn về dự đoán).
    f = next((d / "cmask_flags.npz" for d in input_dirs() if (d / "cmask_flags.npz").exists()), None) if Path("/kaggle/input").exists() else None
    if f is None and os.environ.get("UPAR_CMASK"): f = Path(os.environ["UPAR_CMASK"]) / "cmask_flags.npz"
    assert f is not None and f.exists(), "CMASK: không thấy cmask_flags.npz (Input dataset upar-cmask-1018 hoặc UPAR_CMASK)"
    z = np.load(f, allow_pickle=False)
    assert [str(a) for a in z["attr"]] == list(ATTR), "CMASK: thứ tự thuộc tính trong file cờ khác ATTR"
    F = np.unpackbits(z["flag"], axis=1, count=len(ATTR)).astype(bool)
    idx = {r: i for i, r in enumerate(z["images"].tolist())}
    rows = np.array([idx.get(r, -1) for r in frame.image])
    assert (rows >= 0).all(), f"CMASK: {int((rows < 0).sum())} / {len(rows)} ảnh train không có trong file cờ (khác split resplit_0710?)"
    y = frame[ATTR].values.astype(np.float32) if y is None else y
    m = MASK[frame.ds.map(DS_ID).values].copy() if m is None else m.copy()
    sel = F[rows] & (m > 0); m[sel] = 0
    print(f"  clean_mask ({f}): bỏ {int(sel.sum())} ô khỏi loss ({sel.sum() / max((m + sel).sum(), 1):.2%} ô có nhãn) ở {int(sel.any(1).sum())} ảnh | "
          + ", ".join(f"{d} {int(sel[(frame.ds == d).values].sum())}" for d in sorted(frame.ds.unique())), flush=True)
    return y, m

def t2_find(t2):
    # T2: thư mục chứa cả 2 file teacher (dataset upar-t2-teacher-1013); UPAR_T2_DIR (os.pathsep) để chạy thử ngoài Kaggle
    roots = [Path(p) for p in os.environ.get("UPAR_T2_DIR", "").split(os.pathsep) if p] + list(input_dirs())
    hit = next((d for d in roots if (d / t2["FILE"]).exists() and (d / t2["ROWS"]).exists()), None)
    if hit is None:
        print(f"!! T2: KHÔNG thấy {t2['FILE']} + {t2['ROWS']} trong Input -> Add Input dataset upar-t2-teacher-1013 (hoặc đặt UPAR_T2_DIR)", flush=True)
        raise FileNotFoundError(f"T2: thiếu {t2['FILE']} / {t2['ROWS']}")
    return hit

def t2_cols(t2):
    # chỉ số thuộc tính nhận teacher: ATTRS (None = mọi thuộc tính) trừ EXCLUDE; mỗi phần tử là tên đầy đủ hoặc tiền tố
    inc, exc = t2.get("ATTRS"), tuple(t2.get("EXCLUDE") or ())
    cols = [j for j, a in enumerate(ATTR) if (inc is None or a.startswith(tuple(inc))) and not (exc and a.startswith(exc))]
    assert cols, f"T2: không thuộc tính nào khớp ATTRS {inc} / EXCLUDE {exc}"
    return cols

def t2_targets(frame, t2, y=None, m=None, P=None, rows=None):
    # 13/10 T2: ảnh có dòng teacher (khớp theo ĐƯỜNG DẪN ảnh, không theo tr_row: DEBUG / train+val đổi thứ tự) nhận y = (1 − LAM)·y + LAM·P1 trên cột
    # đã chọn; giữ y cũ ở ô MASK = 0 (5 màu Market không gán: teacher vẫn có số) và ô teacher NaN. Chỉ đổi mảng nhãn đưa vào PARDataset (frame giữ nguyên).
    y = frame[ATTR].values.astype(np.float32) if y is None else np.array(y, np.float32, copy=True)
    m = MASK[frame.ds.map(DS_ID).values].copy() if m is None else m
    if P is None or rows is None:
        d = t2_find(t2); P = np.load(d / t2["FILE"]).astype(np.float32); rows = pd.read_csv(d / t2["ROWS"], dtype={"pid": str})
        print(f"  T2 teacher: {d}", flush=True)
    assert P.ndim == 2 and P.shape == (len(rows), len(ATTR)), f"T2: teacher {P.shape} không khớp {len(rows)} dòng × {len(ATTR)} thuộc tính"
    assert rows.image.is_unique, "T2: ảnh trùng trong file dòng teacher"
    cols, lam = t2_cols(t2), float(t2["LAM"])
    ti = frame.image.map(pd.Series(np.arange(len(rows)), index=rows.image.values)).values     # dòng teacher của từng ảnh (NaN = không có)
    ok = ~pd.isna(ti) & (frame.ds == "Market1501").values
    r, k = np.where(ok)[0], ti[ok].astype(np.int64)
    old, Pc = y[np.ix_(r, cols)], P[np.ix_(k, cols)]
    use = (m[np.ix_(r, cols)] > 0) & np.isfinite(Pc)
    new = np.where(use, (1 - lam) * old + lam * np.nan_to_num(Pc), old).astype(np.float32)
    y[np.ix_(r, cols)] = new
    dlt = np.abs(new - old)[use]; flip = ((new > 0.5) != (old > 0.5)) & use
    print(f"  T2 LAM {lam}: {len(r)} / {len(rows)} ảnh teacher khớp ảnh train ({len(r)} ảnh đổi nhãn) | {len(cols)} thuộc tính "
          f"{[ATTR[j] for j in cols] if len(cols) < 12 else ''} | {int(use.sum())} ô đổi (bỏ {int((~use).sum())} ô MASK 0 / NaN), "
          f"|Δy| TB {dlt.mean() if dlt.size else 0:.4f}, max {dlt.max() if dlt.size else 0:.3f} | {int(flip.sum())} ô qua ngưỡng 0.5", flush=True)
    if not CFG["DEBUG"]:
        assert len(r) >= 0.99 * len(rows), f"T2: chỉ {len(r)} / {len(rows)} ảnh teacher khớp ảnh train -> sai annotation (cần ANN_PREFER 'resplit')?"
    return y, m

def fill_loss(logits, soft, fm, m):
    # K19: nhãn mềm của teacher cho các ô dataset không gán nhãn; chia cùng mẫu số với loss nhãn thật -> mỗi ô nặng ngang nhau
    l = F.binary_cross_entropy_with_logits(logits.float(), soft, reduction="none")
    return (l * fm).sum() / m.sum().clamp(min=1)

def kd_weights(kd, n):
    # Trọng số (n teacher, 40 thuộc tính). E24 GROUPS: teacher t chỉ dạy nhóm GROUPS[t] (vd ["C", "HP"]: teacher màu + teacher còn lại);
    # thuộc tính không teacher nào nhận thì chia đều. Không có GROUPS: trọng số vô hướng W (mặc định chia đều) như trước.
    if kd.get("GROUPS"):
        w_ = np.asarray(kd.get("W") or [1.0] * n, np.float32)[:, None]          # E26: GROUPS + W -> trọng số không đều trong cùng nhóm
        Wm = np.stack([grp_mask(g) for g in kd["GROUPS"]]) * w_; cov = Wm.sum(0)
        return np.where(cov > 0, Wm / np.maximum(cov, 1e-12), 1.0 / n).astype(np.float32)
    return np.repeat(np.asarray(kd.get("W") or [1 / n] * n, np.float32)[:, None], 40, 1)

def kd_loss(logits, soft, m):
    # K1: BCE với xác suất mềm của teacher (chỉ trên thuộc tính được annotate)
    l = F.binary_cross_entropy_with_logits(logits.float(), soft, reduction="none")
    return (l * m).sum() / m.sum().clamp(min=1)

EPS = 1e-20
def par_metrics(gt, pr, valid):
    g, p = gt[:, valid].astype(bool), pr[:, valid].astype(bool)
    tp = (g & p).sum(0); fn = (g & ~p).sum(0); tn = (~g & ~p).sum(0); fp = (~g & p).sum(0)
    rp = tp / (tp + fn + EPS); rn = tn / (tn + fp + EPS)
    mA = np.mean((rp + rn) / 2)
    prec_l = tp / (tp + fp + EPS); lf1 = np.mean(2 * prec_l * rp / (prec_l + rp + EPS))
    inter = (g & p).sum(1); union = (g | p).sum(1)
    ip = inter / (p.sum(1) + EPS); ir = inter / (g.sum(1) + EPS)
    IP, IR = np.mean(ip), np.mean(ir)
    if1 = np.mean(2 * ip * ir / (ip + ir + EPS))      # như scorer BTC (Jia et al.): F1 từng ảnh rồi lấy trung bình
    return dict(Challenge_Avg=2 * mA * if1 / (mA + if1 + EPS), mA=mA, Instance_F1=if1, Label_F1=lf1,
                Instance_Acc=np.mean(inter / (union + EPS)), Instance_Prec=IP, Instance_Rec=IR)

def evaluate(frame, probs, thr=0.5):
    res = {}
    for d, idx in frame.groupby("ds").indices.items():
        gt = frame[ATTR].values[idx]
        res[d] = par_metrics(gt, probs[idx] > thr, gt.sum(0) > 0)
    r = pd.DataFrame(res).T; r.loc["MEAN"] = r.mean()
    return r

class _Scorer:
    def __init__(self, frame, probs):
        self.parts = []
        for d, idx in frame.groupby("ds").indices.items():
            gt = frame[ATTR].values[idx].astype(bool); v = gt.sum(0) > 0
            self.parts.append((gt[:, v], probs[idx][:, v], v))
    def __call__(self, thr):
        return float(np.mean([par_metrics(g, p > thr[v], np.ones(g.shape[1], bool))["Challenge_Avg"] for g, p, v in self.parts]))

def calibrate(frame, probs, grid=np.round(np.arange(0.05, 0.951, 0.025), 3), passes=2):
    # coordinate descent trên ngưỡng từng thuộc tính, mục tiêu = MEAN Challenge_Avg qua các dataset
    sc = _Scorer(frame, probs)
    thr = np.full(40, 0.5); best = sc(thr)
    for _ in range(passes):
        for a in range(40):
            for t in grid:
                old = thr[a]; thr[a] = t
                s = sc(thr)
                if s > best + 1e-6: best = s
                else: thr[a] = old
    return thr, best

logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sigm = lambda z: 1 / (1 + np.exp(-z))

def crossfit(frame, z, folds, grid):
    # calibrate trên fold này, chấm trên fold kia (ở mọi mức shrink k), đổi vai, lấy trung bình -> điểm trung thực
    tab = {k: [] for k in grid}; per_ds = {k: [] for k in grid}
    for f in (0, 1):
        cal, ev = folds != f, folds == f
        fc, fe = frame[cal].reset_index(drop=True), frame[ev].reset_index(drop=True)
        thr, _ = calibrate(fc, sigm(z[cal]))
        off = logit(thr)
        for k in grid:
            r = evaluate(fe, sigm(z[ev] - k * off))
            tab[k].append(r.loc["MEAN", "Challenge_Avg"]); per_ds[k].append(r)
    score = {k: float(np.mean(v)) for k, v in tab.items()}
    k_best = max(grid, key=lambda k: (round(score[k], 6), -k))       # hoà điểm -> chọn k nhỏ (gần 0.5, an toàn hơn)
    detail = (per_ds[k_best][0] + per_ds[k_best][1]) / 2
    return k_best, score, detail

# %% [markdown]
# ## 6. Hàm train một ứng viên

# %%
def layer_id(n, L):
    m = re.match(r"backbone\.(?:blocks|layer|encoder\.layers)\.(\d+)\.", n)
    if m: return int(m.group(1)) + 1
    if n.startswith(("backbone.norm.", "backbone.post_layernorm.", "backbone.visual_projection.", "backbone.head.", "backbone.attn_pool.")): return L + 1
    return 0

def build_param_groups(m, ac):
    groups, vit = {}, m.arch.startswith(("dinov2", "dinov3", "clip", "siglip", "pe_core", "radio", "eva02"))
    L = len(m.backbone.blocks) if vit else 0
    for n, p_ in m.named_parameters():
        no_wd = p_.ndim <= 1 or any(k in n for k in ("cls_token", "pos_embed", "register_tokens", "reg_token", "class_embedding", "position_embedding", "probe", "latent"))
        if n.startswith(("fc.", "net.classifier.2", "net.fc.")):
            lid, lr = "head", ac["HEAD_LR"]
        elif n.startswith(("qdec.", "qdec2.")):                       # 14/10: QDEC2 cùng LR với decoder K11
            lid, lr = "qdec", ac.get("QDEC_LR", 3e-4)
        elif n.startswith(("chead.", "cstats.", "phead.")):           # E12: head mới -> LR của head (layer_id sẽ xếp nhầm vào tầng 0, LR gần 0)
            lid, lr = "chead", ac["HEAD_LR"]
        elif vit:
            i = layer_id(n, L); lid, lr = f"layer{i}", ac["LR"] * ac["LLRD"] ** (L + 1 - i)
        else:
            lid, lr = "backbone", ac["LR"]
        g = groups.setdefault((lid, no_wd), {"params": [], "lr": lr, "weight_decay": 0.0 if no_wd else CFG["WD"], "name": lid})
        g["params"].append(p_)
    return list(groups.values())

@torch.no_grad()
def predict(m, loader, q=False):
    m.eval(); out = []
    fw = (lambda x_: torch.cat(m(x_, return_q=True), 1)) if q else m      # 14/10 QDEC2: 1 lượt forward ra [z | z_q] (80 cột)
    for x, *_ in loader:
        x = x.to(dev, non_blocking=True)
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            z = fw(x).float()
            if CFG["TTA_FLIP"]: z = (z + fw(torch.flip(x, dims=[3])).float()) / 2
        out.append(z.cpu())
    return torch.cat(out).numpy()

def sysinfo():
    import psutil
    s_ = f"RAM {psutil.virtual_memory().used / 2**30:.1f}GB"
    if dev.type == "cuda":
        s_ += " | " + " ".join(f"GPU{i} {torch.cuda.memory_allocated(i) / 2**30:.1f}GB (đỉnh {torch.cuda.max_memory_allocated(i) / 2**30:.1f})" for i in range(torch.cuda.device_count()))   # đỉnh GPU1 > 0 = DataParallel có dùng GPU thứ 2
    return s_

def input_runs(name, tag, root="/kaggle/input"):
    """Các thư mục kết quả cũ trong Input; bản LODO lấy epoch cuối (meta "select" = "last") đứng trước."""
    qs = [d for d in input_dirs(root) if d.name == f"{name}_{tag}" and d.parent.name == "runs_v2" and (d / "meta.json").exists()]
    qs = [q for q in qs if ((q / "best_ema.pt").exists() or CFG.get("REUSE_LOGITS_ONLY")) and (q / "val_logits.npy").exists()]
    return sorted(qs, key=lambda q: (json.load(open(q / "meta.json")).get("select") != "last", str(q)))

def run_tag():
    # 08/10: run train trên split chia lại mang tag "val_rs" để không bao giờ lẫn với run split chính thức ("val") cùng tên trong Input
    rs = os.environ.get("UPAR_ANN_PREFER", CFG.get("ANN_PREFER", ""))
    return ("val_rs" if rs else "val") + ("_debug" if CFG["DEBUG"] else "")

def _n_val(q):
    try: return len(np.load(q / "val_logits.npy", mmap_mode="r"))
    except Exception: return -1

def find_run_dir(name, tag):
    q = RUNS / f"{name}_{tag}"
    if (q / "best_ema.pt").exists() and (q / "meta.json").exists(): return q
    qs = input_runs(name, tag)
    if not qs and tag.startswith("val_rs"):
        # tag cũ "val" có thể là run split mới train bằng notebook trước 08/10 (vd teacher L3, bản 1008): nhận ra bằng số ảnh val
        old = input_runs(name, tag.replace("val_rs", "val", 1))
        same = [q_ for q_ in old if _n_val(q_) == len(VA)]
        qs = same or old
        if same: print(f"  {name}: dùng {qs[0]} (tag cũ 'val' nhưng train trên split mới, val {len(VA)} ảnh)")
        elif qs: print(f"  !! {name}: không có run split mới ({name}_{tag}) -> dùng run split chính thức {qs[0]}")
    return qs[0] if qs else None

def reuse_runs(name, tag, n_val):
    """Run trong Input dùng lại được: đúng tag; split mới thì nhận cả tag cũ "val" nếu val cùng số ảnh."""
    prev = input_runs(name, tag)
    if not prev and tag.startswith("val_rs"):
        prev = [q_ for q_ in input_runs(name, tag.replace("val_rs", "val", 1)) if _n_val(q_) == n_val]
    return prev

@torch.no_grad()
def teacher_logits(tdir, frame):
    meta = json.load(open(tdir / "meta.json"))
    m = PARNet(meta["arch"], pretrained=False, img_h=meta["img_h"], img_w=meta["img_w"], parts=meta.get("parts", False), qdec=meta.get("qdec", False), cmid=tuple(meta.get("cmid", ())), chist=bool(meta.get("chist", False)), pmask=meta.get("pmask"),
               qdec2=meta.get("QDEC2"))    # 14/10: best_ema.pt của run QDEC2 có cả đầu z_q (teacher vẫn chỉ dùng z)
    m.load_state_dict({k: v.float() for k, v in torch.load(tdir / "best_ema.pt", map_location="cpu").items()})
    z = predict(m.to(dev), make_loader(frame, meta["img_h"], meta["img_w"], tuple(meta["mean"]), tuple(meta["std"]), False, CFG["BATCH"] * 2, wb=meta.get("wb", False), crop=meta.get("CROP"),
                                       lbox=bool((meta.get("AUG") or {}).get("LBOX"))))
    del m; gc.collect()
    if dev.type == "cuda": torch.cuda.empty_cache()
    return z

class _TeacherAdapter(nn.Module):
    # E26 (H7): teacher khác kích thước ảnh / chuẩn hoá với student. Batch đã augment của student được bỏ chuẩn hoá, resize bilinear (antialias)
    # về kích thước của teacher rồi chuẩn hoá theo teacher -> teacher vẫn nhìn đúng ảnh student đang thấy (KD nhất quán).
    def __init__(self, m, s_norm, t_norm, t_size):
        super().__init__(); self.m, self.size = m, tuple(t_size)
        for n_, v in (("ms", s_norm[0]), ("ss", s_norm[1]), ("mt", t_norm[0]), ("st", t_norm[1])):
            self.register_buffer(n_, torch.tensor(v, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
    @property
    def backbone(self): return self.m.backbone
    def forward(self, x, return_feat=False):
        x = x.float() * self.ss + self.ms
        if tuple(x.shape[-2:]) != self.size: x = F.interpolate(x, size=self.size, mode="bilinear", antialias=True, align_corners=False)
        return self.m((x - self.mt) / self.st, return_feat=return_feat)

def load_teacher(tdir, size, norm, wb):
    # E11: teacher online nhìn đúng batch đã augment của student. Cùng white balance là bắt buộc; khác kích thước / chuẩn hoá thì đi qua
    # _TeacherAdapter (E26). Chưng cất đặc trưng (FEAT / FEAT_REG) vẫn cần teacher đầu tiên cùng kích thước (cùng lưới patch).
    meta = json.load(open(tdir / "meta.json"))
    assert bool(meta.get("wb", False)) == bool(wb), f"teacher {tdir.name} khác white balance với student"
    assert not meta.get("CROP"), f"teacher online {tdir.name} dùng ảnh cắt (CROP): chỉ dùng được làm teacher offline"
    same = ((meta["img_h"], meta["img_w"]) == tuple(size) and tuple(meta["mean"]) == tuple(norm[0]) and tuple(meta["std"]) == tuple(norm[1]))
    m = PARNet(meta["arch"], pretrained=False, img_h=meta["img_h"], img_w=meta["img_w"], parts=meta.get("parts", False), qdec=meta.get("qdec", False), cmid=tuple(meta.get("cmid", ())), chist=bool(meta.get("chist", False)), pmask=meta.get("pmask"),
               qdec2=meta.get("QDEC2"))    # 14/10: như teacher_logits
    m.load_state_dict({k: v.float() for k, v in torch.load(tdir / "best_ema.pt", map_location="cpu").items()})
    m = m.to(dev).eval()
    for p_ in m.parameters(): p_.requires_grad_(False)
    if not same:
        print(f"  teacher {tdir.name}: {meta['img_h']}x{meta['img_w']} -> đổi batch student {size[0]}x{size[1]} sang kích thước / chuẩn hoá của teacher")
        m = _TeacherAdapter(m, norm, (tuple(meta["mean"]), tuple(meta["std"])), (meta["img_h"], meta["img_w"])).to(dev).eval()
    return nn.DataParallel(m) if (CFG["MULTI_GPU"] and torch.cuda.device_count() > 1) else m

def prep_trainable(model, ac):
    # Đóng băng embedding + khối 1..FREEZE (E13). FROZEN_FP16 (H+, 10/10): ma trận của phần đóng băng giữ fp16 trên GPU (autocast vẫn chạy fp16;
    # cls / register / LayerNorm / gamma giữ fp32 nên dòng residual vẫn fp32). CKPT: gradient checkpointing cho khối còn train (DinoV3Eva).
    L_vit = len(model.backbone.blocks) if model.arch.startswith(("dinov2", "dinov3", "clip", "siglip", "pe_core", "radio", "eva02")) else 0
    fz = {n_ for n_, _ in model.named_parameters()
          if L_vit and ac.get("FREEZE", 0) and n_.startswith("backbone.") and layer_id(n_, L_vit) <= ac["FREEZE"]}
    for n_, p_ in model.named_parameters():
        if n_ in fz:
            p_.requires_grad_(False)
            if ac.get("FROZEN_FP16") and p_.is_cuda and p_.ndim >= 2: p_.data = p_.data.half()
    if hasattr(model.backbone, "ckpt"): model.backbone.ckpt = bool(ac.get("CKPT"))
    return fz

def make_ema(model, fz, build):
    # EMA dựng trên CPU rồi mới đưa lên GPU; tham số đóng băng dùng chung bộ nhớ với model (không đổi nên không cần bản riêng: H+ tiết kiệm ~1.7 GB)
    ema = build(); ema.load_state_dict(model.state_dict())
    mp = dict(model.named_parameters())
    for n_, p_ in ema.named_parameters():
        p_.requires_grad_(False)
        if n_ in fz: p_.data = mp[n_].data
    return ema.to(next(model.parameters()).device).eval()

def auto_fit(name, ac, arch, H, W, BS, parts, n_train, n_val):
    # H+ (10/10): thử lần lượt (FREEZE, CKPT) trên lô ngẫu nhiên đúng kích thước (đủ train step + EMA + DataParallel), lấy cấu hình đầu tiên
    # có đỉnh bộ nhớ ≤ FIT_MAX_GB trên mọi GPU. AUTO_EP: số epoch ≤ EPOCHS sao cho train xong trước TRAIN_BUDGET_H (tính từ đầu notebook).
    ok = False
    for fz_n, ck in ac["AUTO_FIT"]:
        a_ = dict(ac, FREEZE=fz_n, CKPT=ck); m = e = opt = net = loss = None; ips, peak, err = 0.0, [0.0], ""
        try:
            m = PARNet(arch, pretrained=True, drop_path=ac["DROP_PATH"], img_h=H, img_w=W, parts=parts, drop=ac.get("HEAD_DROP", 0.2)).to(dev)
            fz = prep_trainable(m, a_)
            e = make_ema(m, fz, lambda: PARNet(arch, pretrained=False, img_h=H, img_w=W, parts=parts))
            opt = torch.optim.AdamW(build_param_groups(m, a_))
            net = nn.DataParallel(m) if (CFG["MULTI_GPU"] and torch.cuda.device_count() > 1) else m
            scaler = torch.amp.GradScaler("cuda"); net.train()
            for i_ in range(torch.cuda.device_count()): torch.cuda.reset_peak_memory_stats(i_)
            x = torch.randn(BS, 3, H, W, device=dev); y = (torch.rand(BS, 40, device=dev) < 0.2).float(); ts = []
            for it in range(6):
                torch.cuda.synchronize(); t0 = time.time()
                with torch.autocast("cuda"): loss = F.binary_cross_entropy_with_logits(net(x).float(), y)
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(m.parameters(), 5.0); scaler.step(opt); scaler.update()
                with torch.no_grad():
                    msd = m.state_dict()
                    for k_, v_ in e.state_dict().items():
                        if k_ not in fz and v_.dtype.is_floating_point: v_.mul_(0.999).add_(msd[k_], alpha=0.001)
                torch.cuda.synchronize(); ts.append(time.time() - t0)
            peak = [torch.cuda.max_memory_allocated(i_) / 2**30 for i_ in range(torch.cuda.device_count())]
            ips = BS / float(np.median(ts[2:])); ok = max(peak) <= CFG["FIT_MAX_GB"]
        except torch.cuda.OutOfMemoryError:
            err = " | OOM"
        finally:
            del m, e, opt, net, loss; gc.collect(); torch.cuda.empty_cache()
        print(f"[{name}] AUTO_FIT FREEZE {fz_n} CKPT {ck}: {'VỪA' if ok else 'không vừa'} | đỉnh "
              + " / ".join(f"{v:.1f}" for v in peak) + f" GB | {ips:.1f} img/s{err}", flush=True)
        if ok: break
    assert ok, "AUTO_FIT: không cấu hình nào vừa bộ nhớ GPU"
    ac = dict(ac, FREEZE=fz_n, CKPT=ck, FIT_IPS=round(ips, 2))
    if ac.get("AUTO_EP") and not CFG["DEBUG"]:
        ips_r = 0.92 * ips                                  # dữ liệu thật + augment + log: chậm hơn lô ngẫu nhiên một chút
        ep_s = n_train / ips_r + n_val / (2.5 * ips_r)      # val chỉ forward
        rem = CFG["TRAIN_BUDGET_H"] * 3600 - (time.time() - NB_T0)
        ep = int(max(1, min(ac.get("EPOCHS", CFG["EPOCHS"]), rem // ep_s)))
        print(f"[{name}] AUTO_EP: ~{ep_s / 60:.0f} phút / epoch, còn {rem / 3600:.1f} giờ trong ngân sách -> {ep} epoch", flush=True)
        ac["EPOCHS"] = ep
    return ac

def train_candidate(name, train_frame, val_frame, tag, full=False, sel_ds=None, reuse=True, force_last=False):
    ac = ARCH_CFG[name]; arch = ac.get("ARCH", name); parts = ac.get("PARTS", False); H, W = ac["IMG"]; mean, std = norm_for(arch)
    wb, qd = bool(ac.get("WB", False)), bool(ac.get("QDEC", False))
    q2 = ac.get("QDEC2")                                     # 14/10: đầu truy vấn kép (z_q giám sát riêng)
    if q2:            # chỉ ghép với loss BCE thường (crit): các nhánh dưới gọi model theo đường khác hoặc chỉ giám sát z (chưa kiểm với z_q)
        bad = [k_ for k_ in ("DS_BIAS", "KD", "RESTORE", "IDPAIR", "GROUP_CE", "COLOR_CE", "SOFTF1") if ac.get(k_)]
        assert not bad, f"QDEC2 chưa ghép với {bad}"
    cm, ch = tuple(ac.get("CMID", ())), bool(ac.get("CHIST", False))        # E12: nhánh màu
    out = RUNS / f"{name}_{tag}"; out.mkdir(parents=True, exist_ok=True)
    last_ep = force_last or bool(ac.get("LAST")) or (bool(sel_ds) and CFG.get("LODO_LAST_EPOCH", False))
    if not reuse:
        print(f"[{name}] train lại (không dùng kết quả cũ trong Input)")
        for fn in ["best_ema.pt", "val_logits.npy", "meta.json", "history.csv"]: (out / fn).unlink(missing_ok=True)
    if not CFG["DEBUG"] and not full and reuse:
        prev = reuse_runs(name, tag, len(val_frame))
        if prev and not (out / "val_logits.npy").exists():
            n_prev = len(np.load(prev[0] / "val_logits.npy", mmap_mode="r")) if (prev[0] / "val_logits.npy").exists() else -1
            if n_prev != len(val_frame):                    # 08/10: run cũ chấm trên val khác (vd split chính thức) -> không dùng lại, train mới
                print(f"[{name}] !! bỏ qua run trong Input {prev[0]}: val có {n_prev} ảnh, val hiện tại {len(val_frame)} ảnh -> train mới")
            else:
                for fn in ["best_ema.pt", "val_logits.npy", "meta.json", "history.csv"]:
                    if (prev[0] / fn).exists(): shutil.copy(prev[0] / fn, out / fn)
                print(f"[{name}] dùng lại kết quả train từ Input: {prev[0]}")
    if (out / "val_logits.npy").exists() and ((out / "best_ema.pt").exists() or CFG.get("REUSE_LOGITS_ONLY")) and not CFG["DEBUG"]:
        print(f"[{name}] đã có kết quả ở {out}, bỏ qua train"); return out
    seed_all(ac.get("SEED", CFG["SEED"]))
    BS = min(ac.get("BATCH", CFG["BATCH"]), CFG["BATCH"])        # batch train riêng của biến thể (vd ViT-L), không vượt CFG
    ym = label_fix(train_frame, ac["CFIX"]) if ac.get("CFIX") is not None else None
    if ac.get("AGEMASK"): ym = age_mask(train_frame, *(ym or (None, None)))
    if ac.get("CMASK"): ym = clean_mask(train_frame, *(ym or (None, None)))
    pr = pos_rate(train_frame) if ym is None else ((ym[0] * ym[1]).sum(0) / np.maximum(ym[1].sum(0), 1)).clip(1e-4, 1 - 1e-4).astype(np.float32)
    if ac.get("T2"):          # 13/10 T2: sau pr -> trọng số wBCE + bias đầu ra vẫn tính từ nhãn gốc (A/B sạch với v11); val_frame không đổi
        bad = [k_ for k_ in ("GROUP_CE", "COLOR_CE", "SOFTF1", "COLOR_SOFT", "RECOLOR", "HSWAP", "KD", "RESTORE", "GAUG") if ac.get(k_)]
        assert not bad, f"T2 (nhãn mềm) chưa ghép với {bad} (các nhánh này coi nhãn là 0/1)"
        print(f"[{name}] T2 nhãn mềm từ teacher nhiều góc nhìn: {ac['T2']}", flush=True)
        ym = t2_targets(train_frame, ac["T2"], *(ym or (None, None)))
    EP = 1 if CFG["DEBUG"] else ac.get("EPOCHS", CFG["EPOCHS"])
    soft, kd = None, ac.get("KD")
    online, tnets = bool(kd and kd.get("ONLINE")), []
    if online:                                     # E11: teacher chạy trên chính batch đã augment (không tính nhãn mềm trước)
        for tn in kd["TEACHERS"]:
            tdir = find_run_dir(tn, tag)
            assert tdir is not None, f"KD: không tìm thấy teacher {tn}_{tag} (train teacher trước hoặc Add Input)"
            tnets.append(load_teacher(tdir, (H, W), (mean, std), wb)); print(f"[{name}] teacher online {tn}: {tdir}")
            if kd.get("T_FP16") and dev.type == "cuda":   # 18/10: ma trận teacher fp16 như FROZEN_FP16 (teacher chạy trong autocast; LayerNorm / bias / token giữ fp32)
                for p_ in (tnets[-1].module if isinstance(tnets[-1], nn.DataParallel) else tnets[-1]).parameters():
                    if p_.ndim >= 2: p_.data = p_.data.half()
                print(f"[{name}] teacher {tn}: ma trận fp16, GPU0 đang giữ {torch.cuda.memory_allocated(0) / 2**30:.2f} GB", flush=True)
    elif kd:                                       # K1: nhãn mềm từ ensemble teacher cùng fold (cùng tag)
        zs = []
        for tn in kd["TEACHERS"]:
            tdir = find_run_dir(tn, tag)
            assert tdir is not None, f"KD: không tìm thấy teacher {tn}_{tag} (train teacher trước hoặc Add Input)"
            zs.append(teacher_logits(tdir, train_frame)); print(f"[{name}] teacher {tn}: {tdir}")
        wt = kd_weights(kd, len(zs))
        soft = 1 / (1 + np.exp(-sum(w_[None] * z_ for w_, z_ in zip(wt, zs)) / kd.get("T", 1.0)))
    if online: WT = torch.as_tensor(kd_weights(kd, len(tnets)), device=dev)
    LB = bool((ac.get("AUG") or {}).get("LBOX"))                  # E31: letterbox cả train lẫn val / teacher / run.py
    rs, rs_pack = ac.get("RESTORE"), None                          # E35: ảnh phục hồi offline
    if rs:
        assert not (rs.get("PAIR") and (kd or ac.get("GAUG") or ac.get("RECOLOR") or ac.get("HSWAP") or ac.get("DS_BIAS"))), "E35 PAIR chưa ghép với KD / GAUG / RECOLOR / HSWAP / DS_BIAS"
        rs_pack = dict(items=restore_lookup(train_frame, rs), P=rs.get("P", 0.5), PAIR=bool(rs.get("PAIR")))
    global CUR_BGAUG; CUR_BGAUG = ac.get("BGAUG")          # 17/10 tăng cường nền (chỉ loader train)
    dl_tr = make_loader(train_frame, H, W, mean, std, True, BS, ac.get("AUG"), wb, ac.get("RFS_T"), soft, ym, ac.get("GAUG"), ac.get("RECOLOR"), crop=ac.get("CROP"),
                        ds_bal=ac.get("DS_BAL"), sampler=ac.get("SAMPLER"), hswap=ac.get("HSWAP"), restore=rs_pack, idpair=ac.get("IDPAIR"))
    if rs: print(f"[{name}] E35 ảnh phục hồi: {rs}", flush=True)
    if ac.get("HSWAP"): print(f"[{name}] E34 ghép nửa trên / nửa dưới: {ac['HSWAP']}")
    if ac.get("GAUG"): print(f"[{name}] E24 augmentation theo nhóm: " + " | ".join(f"P {p_.get('P', 1)} loss {p_.get('LOSS', 'HPC')} kd {p_.get('KD', p_.get('LOSS', 'HPC'))} {p_.get('AUG', {})}" for p_ in ac["GAUG"]))
    if ac.get("RECOLOR"): print(f"[{name}] E24 đổi màu có đổi nhãn: {ac['RECOLOR']}")
    MASK_T = torch.as_tensor(MASK, device=dev)        # K19: ô gốc không gán nhãn (theo dataset)
    dl_va = make_loader(val_frame, H, W, mean, std, False, CFG["BATCH"] * 2, wb=wb, crop=ac.get("CROP"), lbox=LB)
    pm, pm_dir = (dict(ac["PMASK"]) if ac.get("PMASK") else None), None
    if pm and not pm.get("FIXED"):                     # E27: DINO đã fine-tune khoanh vùng (cùng fold / cùng split với model này)
        pm_dir = find_run_dir(pm["SRC"], tag)
        assert pm_dir is not None, f"E27: không thấy model khoanh vùng {pm['SRC']}_{tag} (train trước hoặc Add Input)"
        sm_ = json.load(open(pm_dir / "meta.json"))
        pm.update(src_arch=sm_["arch"], src_h=sm_["img_h"], src_w=sm_["img_w"], src_mean=list(sm_["mean"]), src_std=list(sm_["std"]))
        print(f"[{name}] E27 khoanh vùng áo / quần bằng {pm_dir} (τ {pm.get('TAU', 1.5)}, token khối {pm.get('TAPS', ())})")
    if ac.get("AUTO_FIT") and dev.type == "cuda":
        ac = auto_fit(name, ac, arch, H, W, BS, parts, len(train_frame), len(val_frame))
        if not CFG["DEBUG"]: EP = ac.get("EPOCHS", EP)
    model = PARNet(arch, pretrained=True, drop_path=ac["DROP_PATH"], img_h=H, img_w=W, parts=parts, qdec=qd, cmid=cm, chist=ch, pmask=pm,
                   drop=ac.get("HEAD_DROP", 0.2), qdec2=q2)      # E31 drop8: dropout rất lớn trên đặc trưng gộp (chỉ khi train; suy luận không đổi)
    if pm_dir is not None:
        model.pmasker.net.load_state_dict({k: v.float() for k, v in torch.load(pm_dir / "best_ema.pt", map_location="cpu").items()})
    if model.pmasker is not None:
        for p_ in model.pmasker.parameters(): p_.requires_grad_(False)
    with torch.no_grad():
        model.head.bias.copy_(torch.as_tensor(logit(pr), dtype=torch.float32))      # khởi tạo đầu ra = prior
        if q2: model.qdec2.b.copy_(torch.as_tensor(logit(pr), dtype=torch.float32))  # QDEC2: z_q = prior lúc đầu (w = 0)
        if arch.startswith(("clip", "siglip")) and CFG["TEXT_INIT"]:
            try:
                Wt = clip_text_head(arch, CFG["TEXT_SCALE"])
                model.head.weight.zero_(); model.head.weight[:, :Wt.shape[1]].copy_(Wt)
                print(f"[{arch}] head khởi tạo bằng text embedding, |W| = {Wt.norm(dim=1).mean():.2f}")
            except Exception as e:
                print(f"[{arch}] KHÔNG khởi tạo được text head ({type(e).__name__}: {e}) -> dùng khởi tạo ngẫu nhiên")
    model = model.to(dev)
    if ac.get("FP16_CHECK") and dev.type == "cuda":     # H+ (pretrain bf16, T4 chỉ có fp16): so đặc trưng backbone fp16 vs fp32 trên ảnh val thật
        with torch.no_grad():
            xb = next(iter(dl_va))[0][:16].to(dev); model.eval(); f32 = model.backbone(xb).float()
            with torch.autocast("cuda"): f16 = model.backbone(xb).float()
            model.train()
        rel_, fin_ = float((f16 - f32).abs().max() / f32.abs().max()), bool(torch.isfinite(f16).all())
        print(f"[{name}] kiểm fp16 trên 16 ảnh val: lệch tương đối max {rel_:.2e} | hữu hạn {fin_}", flush=True)
        assert fin_ and rel_ < 0.05, "fp16 tràn số / lệch lớn so với fp32 -> không train được bằng fp16 trên T4"
        del xb, f32, f16
    fz = prep_trainable(model, ac); frozen = lambda n: n in fz                    # E13: embedding (0) và khối 1..FREEZE
    if ac.get("FREEZE"): print(f"[{name}] đóng băng {sum(p_.numel() for p_ in model.parameters() if not p_.requires_grad) / 1e6:.1f}M tham số"
                               + (" (fp16)" if ac.get("FROZEN_FP16") and dev.type == "cuda" else "") + (" | gradient checkpointing" if ac.get("CKPT") else ""))
    if arch.startswith(("clip", "siglip")):  # zero-shot nhanh để chắc text head hợp lý (chỉ in ra, không dùng để chọn)
        sub = val_frame.groupby("ds").head(300 if not CFG["DEBUG"] else 20).reset_index(drop=True)
        z0 = predict(model, make_loader(sub, H, W, mean, std, False, CFG["BATCH"] * 2, wb=wb, crop=ac.get("CROP"), lbox=LB))
        r0 = evaluate(sub, sigm(z0)).loc["MEAN"]
        print(f"[{arch}] zero-shot + prior trên {len(sub)} ảnh val: Avg {r0.Challenge_Avg:.4f} | mA {r0.mA:.4f} | F1 {r0.Instance_F1:.4f}")
    ema = make_ema(model, fz, lambda: PARNet(arch, pretrained=False, img_h=H, img_w=W, parts=parts, qdec=qd, cmid=cm, chist=ch, pmask=pm, qdec2=q2))

    pg = build_param_groups(model, ac); opt = torch.optim.AdamW(pg)
    steps_ep = len(dl_tr); total = steps_ep * EP; warm = int(steps_ep * CFG["WARMUP_EP"])
    base_lrs = [g["lr"] for g in opt.param_groups]
    def set_lr(step):
        f = step / max(1, warm) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
        for g, b in zip(opt.param_groups, base_lrs): g["lr"] = b * f
    @torch.no_grad()
    def ema_update(step):
        d = min(CFG["EMA"], (1 + step) / (10 + step)); msd = model.state_dict()
        for k, v in ema.state_dict().items():
            if k in fz: continue                       # đóng băng: dùng chung bộ nhớ với model
            if v.dtype.is_floating_point: v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else: v.copy_(msd[k])
    crit = MaskedLoss(pr, CFG["LABEL_SMOOTH"], ac.get("WBCE_POW", 1.0), ac.get("ASL"), soft_w=bool(ac.get("T2"))).to(dev)
    csim = color_sim_matrix().to(dev) if ac.get("COLOR_SOFT") else None
    ms_hooks = []
    if ac.get("MIXSTYLE"):                          # K10: hook chỉ gắn vào model đang train (EMA không có)
        mx = ac["MIXSTYLE"]
        ms_hooks = [model.backbone.blocks[b].register_forward_hook(MixStyle(mx.get("P", 0.5), mx.get("ALPHA", 0.1))) for b in mx.get("BLOCKS", [1, 3, 5])]
    use_dsb = bool(ac.get("DS_BIAS", False))       # K2: bias riêng cho từng dataset, gộp trung bình vào head khi lưu
    dsb = nn.Parameter(torch.zeros(len(DS_NAMES), 40, device=dev)); ema_dsb = torch.zeros(len(DS_NAMES), 40, device=dev)
    if use_dsb: opt.add_param_group({"params": [dsb], "lr": ac["HEAD_LR"], "weight_decay": 0.0, "name": "ds_bias"}); base_lrs.append(ac["HEAD_LR"])
    feat_w = float(kd.get("FEAT", 0.0)) if online else 0.0
    featr_w = float(kd.get("FEAT_REG", 0.0)) if online else 0.0
    need_feat = bool(feat_w or featr_w)
    if need_feat: t0m = tnets[0].module if isinstance(tnets[0], nn.DataParallel) else tnets[0]
    if feat_w:                                     # E11: lớp chiếu đặc trưng student -> teacher, chỉ dùng lúc train (không lưu vào model)
        proj = nn.Linear(model.backbone.out_dim, t0m.backbone.out_dim).to(dev)
        opt.add_param_group({"params": list(proj.parameters()), "lr": ac["HEAD_LR"], "weight_decay": 0.0, "name": "feat_proj"}); base_lrs.append(ac["HEAD_LR"])
    if featr_w:                                    # E11: lớp chiếu cho đặc trưng 3 vùng (đầu / thân trên / thân dưới)
        projr = nn.Linear(model.backbone.norm.normalized_shape[0], t0m.backbone.norm.normalized_shape[0]).to(dev)
        opt.add_param_group({"params": list(projr.parameters()), "lr": ac["HEAD_LR"], "weight_decay": 0.0, "name": "feat_proj_reg"}); base_lrs.append(ac["HEAD_LR"])
    seen_ids = sorted(set(train_frame.ds.map(DS_ID)))
    def dsb_mean(): return ema_dsb[seen_ids].mean(0)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    net = nn.DataParallel(model) if (CFG["MULTI_GPU"] and torch.cuda.device_count() > 1) else model
    print(f"{'=' * 70}\n[{name}] {'FULL (train+val)' if full else 'train'} | {EP} epoch × {steps_ep} step × batch {BS} | seed {ac.get('SEED', CFG['SEED'])} | aug {ac.get('AUG')} | "
          f"{len(train_frame)} ảnh | IMG {H}x{W} | LR {ac['LR']:.0e} LLRD {ac['LLRD']} | "
          f"{sum(p_.numel() for p_ in model.parameters()) / 1e6:.1f}M tham số\n{'=' * 70}", flush=True)
    hist, best, step, t_stop, n_bad = [], -1, 0, False, 0
    mres = ac.get("MRES"); mres_rng = np.random.RandomState(ac.get("SEED", CFG["SEED"]) + 7)   # 14/10: nhiều độ phân giải lúc train (RoPE / CPE)
    if mres:
        assert arch.startswith(("dinov3", "radio")), "MRES: chỉ cho backbone lưới động (RoPE / CPE)"
        MRES_SIZES = [tuple(s_) for s_ in mres["SIZES"]]; MRES_P = np.asarray(mres.get("P", [1.0] * len(MRES_SIZES)), float); MRES_P = MRES_P / MRES_P.sum()
        print(f"[{name}] MRES: mỗi lô chọn 1 trong {MRES_SIZES} (p {np.round(MRES_P, 3).tolist()}), resize trên GPU", flush=True)
    rf = ac.get("RFREEZE")                          # 17/10: đóng băng ngẫu nhiên khối (chỉ pha fine-tune, chỉ khối không bị FREEZE cố định)
    if rf:
        L_rf = len(model.backbone.blocks); rf_rng = np.random.RandomState(ac.get("SEED", CFG["SEED"]) + 11)   # RNG riêng: thứ tự dữ liệu / augment như bản gốc
        rf_par = [[p_ for n_, p_ in model.named_parameters() if n_.startswith("backbone.") and layer_id(n_, L_rf) == b + 1 and not frozen(n_)] for b in range(L_rf)]
        rf_cnt, rf_n = np.zeros(L_rf), 0
        rf_when = "đầu mỗi epoch" if not rf.get("EVERY") else f"mỗi {rf['EVERY']} bước"
        print(f"[{name}] RFREEZE: P {rf['P']} | bốc lại {rf_when} | {sum(map(bool, rf_par))}/{L_rf} khối tham gia", flush=True)
    def rf_sample():
        off = rf_rng.rand(L_rf) < rf["P"]
        if off.all(): off[rf_rng.randint(L_rf)] = False          # luôn còn ít nhất 1 khối train
        for b, ps in enumerate(rf_par):
            for p_ in ps: p_.requires_grad_(not off[b])           # grad None -> AdamW bỏ qua cả cập nhật lẫn weight decay (zero_grad set_to_none)
        return off
    for ep in range(EP):
        if ac.get("LPFT"):                          # E13 LP-FT: LPFT epoch đầu chỉ train head (fc / qdec / nhánh màu), sau đó mở khoá
            lp = ep < ac["LPFT"]
            for n_, p_ in model.named_parameters():
                if n_.startswith("backbone."): p_.requires_grad_(not lp and not frozen(n_))
            print(f"[{name}] epoch {ep}: {'chỉ train head (LP)' if lp else 'fine-tune'}", flush=True)
        net.train(); t0 = time.time(); run_loss = 0
        for i, bt in enumerate(dl_tr):
            x, y, m, did, sft, kdm = bt[:6]
            x, y, m, kdm = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), m.to(dev, non_blocking=True), kdm.to(dev, non_blocking=True)
            x2, g2 = (bt[6].to(dev, non_blocking=True), bt[7].to(dev, non_blocking=True)) if len(bt) > 6 else (None, None)   # E35 PAIR
            if mres:                                 # 14/10 MRES: ảnh đã chuẩn hoá -> nội suy tuyến tính tương đương resize rồi chuẩn hoá
                hw_ = MRES_SIZES[mres_rng.choice(len(MRES_SIZES), p=MRES_P)]
                if hw_ != tuple(x.shape[-2:]): x = F.interpolate(x, size=hw_, mode="bilinear", align_corners=False, antialias=True)
            if rf and not (ac.get("LPFT") and ep < ac["LPFT"]):
                if i == 0 or (rf.get("EVERY") and step % rf["EVERY"] == 0):
                    rf_off = rf_sample()
                    if i == 0: print(f"[{name}] ep{ep} RFREEZE: đóng băng {int(rf_off.sum())}/{L_rf} khối {(np.flatnonzero(rf_off) + 1).tolist()}", flush=True)
                rf_cnt += ~rf_off; rf_n += 1
            set_lr(step)
            floor = ac["COLOR_SOFT"] * (y @ csim).clamp(max=1) * (1 - y) if csim is not None else None
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                if need_feat: zl, fs, ts = net(x, return_feat=True); zl = zl.float()
                elif x2 is not None: zl, zr = net(torch.cat([x, x2])).float().chunk(2)     # E35: 1 lượt forward cho cả 2 view
                elif q2: zl, zq_ = net(x, return_q=True); zl = zl.float()                    # 14/10 QDEC2: thêm z_q (1 lượt forward)
                else: zl = net(x).float()
                if use_dsb: zl = zl + dsb[did.to(dev)]
                loss = crit(zl, y, m, floor)
                if q2: loss = loss + q2.get("LAM", 1.0) * crit(zq_.float(), y, m, floor)       # 14/10 QDEC2: z_q giám sát riêng, cùng mask
                if x2 is not None: loss = loss + rs.get("ALPHA", 0.5) * crit(zr, y, m * g2) + rs.get("CONS", 1.0) * cons_loss(zl, zr, g2)
                if ac.get("IDPAIR"):                         # E36: cùng người khác camera -> cùng dự đoán (mọi thuộc tính, kể cả Market không gán)
                    k2 = 2 * ac["IDPAIR"].get("K", 8); za_, zb_ = zl[0:k2:2], zl[1:k2:2]
                    loss = loss + ac["IDPAIR"].get("W", 1.0) * cons_loss(za_, zb_, torch.ones_like(za_))
                if online:                                    # E11: teacher nhìn đúng ảnh student đang thấy
                    with torch.no_grad():
                        outs = [tn_(x, return_feat=True) for tn_ in tnets]
                        zt = sum(w_[None] * o[0].float() for w_, o in zip(WT, outs))     # E24: trọng số teacher theo từng thuộc tính
                    sft = torch.sigmoid(zt / kd.get("T", 1.0))
                if kd and kd.get("MODE") == "fill":          # K19: chỉ ô dataset gốc không gán và chưa có nhãn (vd 5 màu Market, Other của PETA áo 2 màu)
                    loss = loss + kd.get("LAMBDA", 1.0) * fill_loss(zl, sft.to(dev), (1 - MASK_T[did.to(dev)]) * (1 - m), m)
                elif kd:                                      # nhiệt độ T cho cả student, ×T² (T = 1: giống hệt kdL)
                    T_ = kd.get("T", 1.0)
                    km = (torch.ones_like(m) if kd.get("ALL") else m) * kdm   # ALL: teacher cho nhãn mềm cả thuộc tính dataset không gán (như bản 0.7831); kdm: E24 nhóm KD của chính sách aug
                    if kd.get("ONLY"): km = km * torch.as_tensor(grp_mask(kd["ONLY"]), device=km.device)   # E26: chỉ chưng cất nhóm này, nhóm khác học nhãn thật
                    loss = (1 - kd.get("LAMBDA", 0.5)) * loss + kd.get("LAMBDA", 0.5) * T_ * T_ * kd_loss(zl / T_, sft.to(dev), km)
                if feat_w:                                    # E11: chưng cất đặc trưng (1 − cos)
                    loss = loss + feat_w * (1 - F.cosine_similarity(proj(fs.float()), outs[0][1].float(), dim=1)).mean()
                if featr_w:                                   # E11: chưng cất theo vùng (cùng lưới vì teacher cùng kích thước ảnh)
                    s_reg = upar_model._stripe_means(ts.float(), model.backbone.grid)
                    t_reg = upar_model._stripe_means(outs[0][2].float(), model.backbone.grid)
                    loss = loss + featr_w * sum((1 - F.cosine_similarity(projr(a), b, dim=1)).mean() for a, b in zip(s_reg, t_reg)) / len(s_reg)
                if ac.get("GROUP_CE"): loss = loss + ac["GROUP_CE"] * group_ce(zl, y, m)
                if ac.get("COLOR_CE"): loss = loss + ac["COLOR_CE"] * group_ce(zl, y, m, COLOR_GROUPS)
                if ac.get("SOFTF1"): loss = loss + ac["SOFTF1"] * soft_f1_loss(zl, y, m)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), ac["GRAD_CLIP"])
            scaler.step(opt); scaler.update(); ema_update(step); step += 1
            if use_dsb:
                with torch.no_grad(): ema_dsb.mul_(min(CFG["EMA"], (1 + step) / (10 + step))).add_(dsb.detach(), alpha=1 - min(CFG["EMA"], (1 + step) / (10 + step)))
            lv_ = loss.item(); n_bad = 0 if math.isfinite(lv_) else n_bad + 1
            if n_bad >= 50: raise RuntimeError(f"[{name}] loss không hữu hạn 50 bước liền (ep{ep} it{i}): fp16 tràn số?")
            run_loss += lv_ if math.isfinite(lv_) else 0.0
            if time.time() - NB_T0 > CFG["TIME_LIMIT_H"] * 3600:
                t_stop = True; print(f"[{name}] !! quá TIME_LIMIT_H {CFG['TIME_LIMIT_H']} giờ ở ep{ep} it{i}: dừng train, lưu EMA hiện tại", flush=True); break
            if i % CFG["LOG_EVERY"] == 0:
                el = time.time() - t0; ips = (i + 1) * BS / el
                print(f"[{arch}] ep{ep} it{i}/{steps_ep} loss {run_loss / (i + 1):.4f} | {ips:.0f} img/s | "
                      f"ETA epoch {(steps_ep - i - 1) * BS / ips / 60:.1f} min | {sysinfo()}", flush=True)
        if rf and rf_n: print(f"[{name}] ep{ep} RFREEZE cộng dồn: TB {rf_cnt.sum() / rf_n:.1f}/{L_rf} khối train mỗi bước | "
                              f"tỉ lệ train theo khối min {rf_cnt.min() / rf_n:.2f} / max {rf_cnt.max() / rf_n:.2f}", flush=True)
        z_va = predict(ema, dl_va, q=bool(q2))
        if q2: z_va, zq_va = z_va[:, :40], z_va[:, 40:]        # 14/10 QDEC2: z (chọn epoch, val_logits.npy, zip) | z_q (val_logits_q*.npy, ghép offline)
        if use_dsb: z_va = z_va + dsb_mean().cpu().numpy()[None]
        if ac.get("SAVE_EP_LOGITS"): np.save(out / f"val_logits_ep{ep}.npy", z_va.astype(np.float16))   # 13/10 M2/M5/E2: logit val từng epoch (nhiễu chọn epoch)
        if ac.get("SAVE_EP_LOGITS") and q2: np.save(out / f"val_logits_q_ep{ep}.npy", zq_va.astype(np.float16))
        if ep in (ac.get("SAVE_EP_EXTRA") or ()) and ac.get("EXTRA_VAL") and not full:   # 13/10 M2: thêm logit EXTRA_VAL (224x112) ở epoch này
            with torch.random.fork_rng(devices=[]):                                      # tạo DataLoader rút RNG CPU -> khôi phục để các epoch sau giữ đúng dòng ngẫu nhiên
                for (h2, w2) in ac["EXTRA_VAL"]:
                    z2 = predict(ema, make_loader(val_frame, h2, w2, mean, std, False, CFG["BATCH"] * 2, wb=wb, crop=ac.get("CROP"), lbox=LB), q=bool(q2))
                    if q2: np.save(out / f"val_logits_q_ep{ep}_{h2}x{w2}.npy", z2[:, 40:].astype(np.float16)); z2 = z2[:, :40]
                    if use_dsb: z2 = z2 + dsb_mean().cpu().numpy()[None]
                    np.save(out / f"val_logits_ep{ep}_{h2}x{w2}.npy", z2.astype(np.float16)); del z2
        r = evaluate(val_frame, sigm(z_va))
        sel = r.loc[sel_ds, "Challenge_Avg"].mean() if sel_ds else r.loc["MEAN", "Challenge_Avg"]   # LODO: chỉ chọn epoch theo dataset đã thấy
        row = {"epoch": ep, "loss": run_loss / max(1, steps_ep), "val_raw_avg": sel, "min": (time.time() - t0) / 60,
               **{f"{d}_avg": r.loc[d, "Challenge_Avg"] for d in r.index if d != "MEAN"}}
        if q2: rq = evaluate(val_frame, sigm(zq_va)); row["q_val_raw_avg"] = rq.loc[sel_ds, "Challenge_Avg"].mean() if sel_ds else rq.loc["MEAN", "Challenge_Avg"]
        hist.append(row); print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        if sel > best or full or last_ep or t_stop:   # full: val nằm trong train; last_ep: LODO lấy epoch cuối
            best = sel
            sd_save = {k: v.detach().to("cpu", copy=True) for k, v in ema.state_dict().items()}     # chép ra CPU (H+: tránh thêm ~2.5 GB trên GPU0)
            if use_dsb: hk = [k for k in sd_save if k.endswith(("fc.2.bias", "classifier.2.1.bias", "fc.1.bias"))][-1]; sd_save[hk] = sd_save[hk] + dsb_mean().cpu()
            torch.save({k: v.half() if v.dtype == torch.float32 else v for k, v in sd_save.items()}, out / "best_ema.pt")   # QDEC2: có cả qdec2.* (zip bỏ ra)
            np.save(out / "val_logits.npy", z_va)
            if q2: np.save(out / "val_logits_q.npy", zq_va)
        if ep in (ac.get("SNAP_EP") or ()):            # 13/10 FULL: thêm bản EMA ở epoch này (vd RADIO v17 = EMA epoch 5 / 8), không đổi best_ema.pt
            sd_s = {k: v.detach().to("cpu", copy=True) for k, v in ema.state_dict().items()}
            if use_dsb: hk_s = [k for k in sd_s if k.endswith(("fc.2.bias", "classifier.2.1.bias", "fc.1.bias"))][-1]; sd_s[hk_s] = sd_s[hk_s] + dsb_mean().cpu()   # gộp bias theo dataset như best_ema.pt
            torch.save({k: v.half() if v.dtype == torch.float32 else v for k, v in sd_s.items()}, out / f"best_ema_ep{ep}.pt"); del sd_s
            print(f"[{name}] đã lưu EMA epoch {ep} -> {out / f'best_ema_ep{ep}.pt'}", flush=True)
        if t_stop: break
    if ac.get("EXTRA_VAL") and arch.startswith(("dinov3", "radio")) and not full:      # 10/10: val logits ở độ phân giải thấp hơn (RoPE, không train lại) để ước tính hệ ghép vừa ngân sách A10
        ema.load_state_dict({k: v.float() for k, v in torch.load(out / "best_ema.pt", map_location="cpu").items()})
        for (h2, w2) in ac["EXTRA_VAL"]:
            z2 = predict(ema, make_loader(val_frame, h2, w2, mean, std, False, CFG["BATCH"] * 2, wb=wb, crop=ac.get("CROP"), lbox=LB), q=bool(q2))
            if q2: np.save(out / f"val_logits_q_{h2}x{w2}.npy", z2[:, 40:]); z2 = z2[:, :40]
            np.save(out / f"val_logits_{h2}x{w2}.npy", z2)
            r2 = evaluate(val_frame, sigm(z2)); s2 = r2.loc[sel_ds, "Challenge_Avg"].mean() if sel_ds else r2.loc["MEAN", "Challenge_Avg"]
            print(f"[{name}] val ở {h2}x{w2} (không train lại): raw {s2:.4f} (256x128 tốt nhất {best:.4f})", flush=True)
    for h_ in ms_hooks: h_.remove()
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    json.dump({"name": name, "arch": arch, "parts": parts, "wb": wb, "qdec": qd, "cmid": list(cm), "chist": ch, "img_h": H, "img_w": W, "mean": list(mean), "std": list(std), **ac, "pmask": pm, "full": full,
               "select": "last" if (last_ep or full) else "best"},
              open(out / "meta.json", "w"), indent=1)
    tnets.clear()
    del net, model, ema, opt, dl_tr, dl_va; gc.collect()
    if dev.type == "cuda": torch.cuda.empty_cache()
    return out

def train_full_only(name, pin=None, n_mon=300):
    # 13/10 FULL_ONLY (notebook upar_full_*.ipynb, tail upar_full_tail.py): train 1 lần duy nhất trên train + val, KHÔNG có run train-only trước
    # (ngưỡng mượn từ config v17). pin: ghi đè công thức (EPOCHS / FREEZE / AUTO_FIT / SNAP_EP như run train-only của thành viên v17).
    # val_frame chỉ để theo dõi (n_mon ảnh / dataset, đã nằm trong train -> số in ra là trong-mẫu, không dùng để đánh giá).
    if pin: ARCH_CFG[name] = dict(ARCH_CFG[name], **pin)
    VA_MON = VA.groupby("ds").head(n_mon).reset_index(drop=True)
    TRVA = pd.concat([TR, VA], ignore_index=True)
    assert not TRVA.image.duplicated().any(), "FULL: ảnh trùng giữa train và val"
    print(f"[{name}] FULL_ONLY: train {len(TR)} + val {len(VA)} = {len(TRVA)} ảnh | theo dõi trên {len(VA_MON)} ảnh val (trong-mẫu) | "
          f"công thức {ARCH_CFG[name]}", flush=True)
    return train_candidate(name, TRVA, VA_MON, tag="full", full=True)

# %% [markdown]
# ## 7. Train các ứng viên + (tuỳ chọn) nạp bản nộp cũ

# %%
def find_baselines():
    out = []
    if not CFG["USE_BASELINE"] or not IS_KAGGLE:
        cands = [Path(p) for p in os.environ.get("UPAR_BASELINES", "").split(os.pathsep) if p]
    else:
        cands = sorted({d for d in input_dirs() if (d / "run.py").exists()})
        for zp in [z for d in input_dirs() for z in glob.glob(str(d / "*.zip"))]:
            try:
                names = zipfile.ZipFile(zp).namelist()
            except Exception:
                continue
            if "run.py" in names:
                d = WORK / "baselines" / Path(zp).stem; d.mkdir(parents=True, exist_ok=True)
                zipfile.ZipFile(zp).extractall(d); cands.append(d)
    for d in cands:
        for sub in [d, d / "assets"]:
            if (sub / "config.json").exists() and (sub / "weights.pt").exists():
                cfg = json.load(open(sub / "config.json"))
                if cfg.get("attributes", ATTR) == ATTR and "models" not in cfg:
                    out.append(dict(dir=d, cfg=cfg, weights=sub / "weights.pt"))
                break
    return out

@torch.no_grad()
def baseline_logits(b):
    c = b["cfg"]; arch, H, W = c["arch"], c["img_h"], c["img_w"]
    mean, std = (tuple(c["mean"]), tuple(c["std"])) if "mean" in c else norm_for(arch)
    m = PARNet(arch, pretrained=False, img_h=H, img_w=W)
    sd = torch.load(b["weights"], map_location="cpu")
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    z = predict(m.to(dev), make_loader(VA, H, W, mean, std, False, CFG["BATCH"] * 2))
    del m; gc.collect()
    return z, dict(arch=arch, img_h=H, img_w=W, mean=list(mean), std=list(std))

RESULTS = []
for name in CFG["CANDIDATES"]:
    if CFG.get("REUSE_ONLY") and not CFG["DEBUG"] and not reuse_runs(name, run_tag(), len(VA)):
        print(f"[{name}] không có run trong Input -> bỏ qua (REUSE_ONLY: notebook này không train)"); continue
    t0 = time.time()
    out = train_candidate(name, TR, VA, tag=run_tag())
    RESULTS.append(dict(name=name, kind="new", z=np.load(out / "val_logits.npy"), ens_w=[1.0], minutes=(time.time() - t0) / 60,
                        members=[dict(name=name, weights=out / "best_ema.pt", meta=json.load(open(out / "meta.json")))]))
for i, b in enumerate(find_baselines()):
    z, meta = baseline_logits(b)
    RESULTS.append(dict(name=f"baseline{i}:{meta['arch']}", kind="baseline", z=z, ens_w=[1.0], minutes=0, dir=b["dir"],
                        members=[dict(name=f"baseline{i}", weights=b["weights"], meta=meta)]))
    print(f"Đã nạp bản cũ {b['dir']} ({meta['arch']} {meta['img_h']}x{meta['img_w']})")
assert RESULTS, "Không có ứng viên nào: Add Input các output chứa runs_v2/<tên>_val_rs (hoặc _val) rồi chạy lại"

# %% [markdown]
# ## 8. Đo chi phí suy luận CPU (torch 2.4.1) và tìm ensemble trong ngân sách thời gian
# Chi phí tính tương đối so với cấu hình của bản 0.776 (DINOv2-B/14 252×126 int8 trên CPU) = 1.0. TTA flip (≈ 2.0) từng làm server quá giờ, nên mặc định chỉ nhận tổ hợp có chi phí fp32 ≤ `MAX_REL_COST`.

# %%
_T241 = {}
def _t241_setup():
    if os.environ.get("UPAR_T241_PY"): return os.environ["UPAR_T241_PY"], {}
    if not IS_KAGGLE: return sys.executable, {}
    if sys.version_info[:2] <= (3, 12):                    # Kaggle cũ: cài wheel torch 2.4.1 vào thư mục riêng
        env = WORK / "torch241"
        if not (env / "torch").exists():
            print("Cài torch 2.4.1 (CPU) vào", env, "...", flush=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", str(env),
                            "torch==2.4.1", "torchvision==0.19.1", "--index-url", "https://download.pytorch.org/whl/cpu",
                            "--extra-index-url", "https://pypi.org/simple"], check=True)
        return sys.executable, {"PYTHONPATH": str(env)}
    # Kaggle từ 10/2026 dùng Python 3.13: torch 2.4.1 không có wheel -> dựng Python 3.11 + torch 2.4.1 (như container server) bằng uv
    venv = WORK / "py311_t241"
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not py.exists():
        print(f"Python {sys.version.split()[0]} không có torch 2.4.1 -> dựng Python 3.11 + torch 2.4.1 CPU bằng uv vào {venv} ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
        subprocess.run([sys.executable, "-m", "uv", "venv", "-q", "--python", "3.11", str(venv)], check=True)
        subprocess.run([sys.executable, "-m", "uv", "pip", "install", "-q", "--python", str(py), "torch==2.4.1", "torchvision==0.19.1", "numpy", "pillow",
                        "--index-url", "https://download.pytorch.org/whl/cpu", "--extra-index-url", "https://pypi.org/simple",
                        "--index-strategy", "unsafe-best-match"], check=True)
    return str(py), {}

def t241_python():
    """Python có torch 2.4.1 CPU để chạy zip như server. Nếu dựng thất bại: dùng torch hiện có (đo tốc độ tương đối vẫn được),
    đặt _T241["ok"] = False và phải kiểm tra zip bằng torch 2.4.1 trên máy khác trước khi nộp."""
    if "py" not in _T241:
        try:
            py, extra = _t241_setup()
            v = subprocess.run([py, "-s", "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True,
                               env=dict(os.environ, **extra)).stdout.strip()
            ok = v.startswith("2.4.1")
            if not ok: print(f"!! Python {py} có torch {v or '(không có)'}, không phải 2.4.1", flush=True)
        except Exception as e:
            print(f"!! Không dựng được môi trường torch 2.4.1: {e}", flush=True); py, extra, ok = sys.executable, {}, False
        if not ok and IS_KAGGLE: py, extra = sys.executable, {}     # dùng torch của Kaggle cho phép đo tốc độ tương đối
        _T241.update(py=py, extra=extra, ok=ok)
        print(f"torch 2.4.1 cho kiểm tra zip: {'OK' if ok else 'KHÔNG CÓ'} ({py})", flush=True)
    return _T241["py"], _T241["extra"]

COST_PY = '''
import sys, os, time, json, torch
sys.path.insert(0, sys.argv[1]); import model as M
specs = json.loads(sys.argv[2]); torch.set_num_threads(max(1, os.cpu_count() or 1))
out = []
for sp in specs:
    m = M.PARNet(sp["arch"], pretrained=False, img_h=sp["img_h"], img_w=sp["img_w"], parts=sp.get("parts", False), qdec=sp.get("qdec", False), cmid=tuple(sp.get("cmid", ())), chist=bool(sp.get("chist", False))).eval()
    if sp.get("int8"):
        skip = tuple(sp.get("int8_skip") or ())
        spec = {n: torch.ao.quantization.default_dynamic_qconfig for n, mod in m.named_modules()
                if isinstance(mod, torch.nn.Linear) and not (skip and n.endswith(skip))}
        m = torch.ao.quantization.quantize_dynamic(m, spec, dtype=torch.qint8)
    x = torch.randn(64, 3, sp["img_h"], sp["img_w"])
    with torch.no_grad():
        m(x[:16]); best = 1e9
        for _ in range(2):
            t = time.time(); m(x); best = min(best, (time.time() - t) / 64 * 1000)
    out.append(best)
print("COST_JSON " + json.dumps(out))
'''

def measure_costs(specs):
    py, extra = t241_python()
    f = WORK / "cost_probe.py"; f.write_text(COST_PY, encoding="utf-8")
    r = subprocess.run([py, "-s", str(f), str(CODE), json.dumps(specs)], capture_output=True, text=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES="", **extra))
    m = re.search(r"COST_JSON (.*)", r.stdout)
    if not m: print(r.stdout[-2000:], r.stderr[-3000:]); raise RuntimeError("đo chi phí thất bại")
    return json.loads(m.group(1))

key_of = lambda meta: (meta["arch"], meta["img_h"], meta["img_w"], bool(meta.get("parts", False)), bool(meta.get("qdec", False)),
                       tuple(meta.get("cmid", ())), bool(meta.get("chist", False)))
REF = dict(arch="dinov2_vitb14_reg", img_h=252, img_w=126, int8=True)          # cấu hình bản 0.776
SEL_DS = CFG["SEL_DS"] or (["Market1501", "PA100k"] if os.environ.get("UPAR_ANN_PREFER", CFG.get("ANN_PREFER", "")).startswith("resplit") else list(DS_NAMES))
SEL_MASK = np.ones(len(VA), bool)
if CFG["SEL_DROP_OLDTRAIN_MARKET"]:
    smap = Path(ANN).parents[1] / "split_map.csv"          # .../resplit_0710/annotations/task1 -> .../resplit_0710/split_map.csv
    if smap.exists():
        mp_ = pd.read_csv(smap, header=None, comment="#", names=["image", "ds", "old", "new", "grp"])
        bad_ = set(mp_.image[(mp_.ds == "Market1501") & (mp_.old == "train")])
        SEL_MASK = ~VA.image.isin(bad_).values
        print(f"Chọn / calibrate bỏ {int((~SEL_MASK).sum())} ảnh Market của val mới lấy từ train chính thức")
    else:
        print(f"!! không thấy {smap} -> chọn trên toàn bộ val")
VS, FS = VA[SEL_MASK].reset_index(drop=True), FOLDS[SEL_MASK]
print("Chọn model theo:", " + ".join(SEL_DS), f"| {len(VS)} ảnh val")

def sel_score(z):
    k_, sc_, det_ = crossfit(VS, z[SEL_MASK], FS, CFG["SHRINK_GRID"])
    return float(np.mean([det_.loc[d, "Challenge_Avg"] for d in SEL_DS if d in det_.index])), k_, sc_, det_

if CFG["GPU_SUBMIT"]:
    def _params(arch):
        return (840e6 if "vith16p" in arch else 631e6 if arch.startswith("radio_v4h") else 304e6 if any(t_ in arch for t_ in ("vitl", "_l14", "_l16", "large")) else 400e6 if "so400m" in arch
                else 22e6 if "vits" in arch else 86e6)
    def gpu_ms(meta):
        p_ = patch_of(meta["arch"]); n_reg = 9 if meta["arch"].startswith("radio") else 4 if (meta["arch"].startswith("dinov3") or "_reg" in meta["arch"]) else 0
        tok = (meta["img_h"] // p_) * (meta["img_w"] // p_) + 1 + n_reg
        return CFG["A10_GPU_MS"] * _params(meta["arch"]) * tok / (304e6 * 133)
    def est_members(members, tta=False):
        # thời gian ước tính trên runner A10 cho 29 248 ảnh (số đo thật từ log G0): nạp + mỗi ảnh (cố định + resize theo kích thước + GPU theo model)
        sizes = {(mb["meta"]["img_h"], mb["meta"]["img_w"]) for mb in members}
        ms_ = (CFG["A10_FIXED_MS"] + sum(CFG["A10_PREP_MS"] * h * w / (256 * 128) for h, w in sizes)
               + sum(gpu_ms(mb["meta"]) for mb in members) * (2 if tta else 1))
        load_ = sum(max(0.0, _params(mb["meta"]["arch"]) - 304e6) / 1e8 for mb in members)    # model lớn hơn L nạp lâu hơn (~+5 s cho H+)
        return CFG["A10_LOAD_S"] + 2.0 * (len(members) - 1) + load_ + 29248 * ms_ / 1000

    @torch.no_grad()
    def flip_logits(mb):
        meta = mb["meta"]
        m = PARNet(meta["arch"], pretrained=False, img_h=meta["img_h"], img_w=meta["img_w"], parts=meta.get("parts", False), qdec=meta.get("qdec", False),
                   cmid=tuple(meta.get("cmid", ())), chist=bool(meta.get("chist", False)), pmask=meta.get("pmask"), qdec2=meta.get("QDEC2"))   # 14/10: TTA vẫn chỉ z
        m.load_state_dict({k_: v.float() for k_, v in torch.load(mb["weights"], map_location="cpu").items()}); m = m.to(dev).eval()
        out_ = []
        for x, *_ in make_loader(VA, meta["img_h"], meta["img_w"], tuple(meta["mean"]), tuple(meta["std"]), False, CFG["BATCH"] * 2, wb=meta.get("wb", False), crop=meta.get("CROP")):
            x = torch.flip(x.to(dev, non_blocking=True), dims=[3])
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                out_.append(m(x).float().cpu())
        del m; gc.collect()
        if dev.type == "cuda": torch.cuda.empty_cache()
        return torch.cat(out_).numpy()

    for r in RESULTS:
        r.update(est=est_members(r["members"]), tta=False)
    if CFG["TTA_EVAL"]:
        for r in [r_ for r_ in RESULTS if len(r_["members"]) == 1]:
            e_tta = est_members(r["members"], tta=True)
            if e_tta > CFG["MAX_EST_SEC"]:
                print(f"[tta] bỏ {r['name']}: ước tính {e_tta:.0f}s > {CFG['MAX_EST_SEC']:.0f}s"); continue
            print(f"[tta] logit val trên ảnh lật cho {r['name']} (ước tính {e_tta:.0f}s) ...", flush=True)
            RESULTS.append(dict(r, name=r["name"] + " +TTA", kind="tta", z=(r["z"] + flip_logits(r["members"][0])) / 2, tta=True, est=e_tta))
    for r in RESULTS:
        s_, k_, sc_, det_ = sel_score(r["z"])
        r.update(score=s_, k=k_, shrink_table=sc_, detail=det_, raw=evaluate(VA, sigm(r["z"])).loc["MEAN", "Challenge_Avg"], cost=r["est"])
        print(f"[{r['name']}] raw@0.5 {r['raw']:.4f} | cross-fit {' + '.join(SEL_DS)} {s_:.4f} | ước tính A10 {r['est']:.0f}s", flush=True)

    if CFG["ENSEMBLE"]:
        pool_ = [r for r in RESULTS if r["kind"] in ("new", "baseline")]
        raw_sel = lambda z: float(np.mean([v for d, v in evaluate(VS, sigm(z[SEL_MASK])).Challenge_Avg.items() if d in SEL_DS]))
        cur = max(pool_, key=lambda r: r["score"]) if pool_ else None
        for _step in range(CFG["ENS_MAX"] - 1):
            if cur is None: break
            best = None
            for c in pool_:
                if c["members"][0]["name"] in [mb["name"] for mb in cur["members"]]: continue
                e_ens = est_members(cur["members"] + c["members"])
                if e_ens > CFG["MAX_EST_SEC_ENS"]:
                    print(f"[ens] bỏ + {c['name']}: tổng ước tính {e_ens:.0f}s > {CFG['MAX_EST_SEC_ENS']:.0f}s"); continue
                a_ = max(CFG["ENS_ALPHAS"], key=lambda w_: raw_sel((1 - w_) * cur["z"] + w_ * c["z"]))
                z_ = (1 - a_) * cur["z"] + a_ * c["z"]; s_, k_, sc_, det_ = sel_score(z_)
                print(f"[ens] {cur['name']} + {c['name']} ×{a_:.2f}: {s_:.4f} (hiện {cur['score']:.4f})", flush=True)
                if best is None or s_ > best[0]: best = (s_, k_, sc_, det_, z_, c, a_)
            if best is None or best[0] < cur["score"] + CFG["ENS_MIN_GAIN"]: break
            s_, k_, sc_, det_, z_, c, a_ = best
            cur = dict(name=f"[{cur['name']}] ×{1 - a_:.2f} + {c['name']} ×{a_:.2f}", kind="ens", z=z_, minutes=0, members=cur["members"] + c["members"],
                       ens_w=[(1 - a_) * w_ for w_ in cur["ens_w"]] + [a_ * w_ for w_ in c["ens_w"]], est=est_members(cur["members"] + c["members"]), tta=False,
                       score=s_, k=k_, shrink_table=sc_, detail=det_, raw=evaluate(VA, sigm(z_)).loc["MEAN", "Challenge_Avg"], cost=est_members(cur["members"] + c["members"]))
            RESULTS.append(cur); print(f"[ens] chọn thêm {c['name']} -> {s_:.4f} | ước tính {cur['est']:.0f}s", flush=True)

    sel_col = "cross-fit " + "+".join(SEL_DS)
    board = pd.DataFrame([{"model": r["name"], "loại": r["kind"], "val raw@0.5": r["raw"], sel_col: r["score"],
                           **{d: r["detail"].loc[d, "Challenge_Avg"] for d in DS_NAMES if d in r["detail"].index},
                           "ước tính A10 (s)": r["est"], "phút train": r["minutes"]} for r in RESULTS]).sort_values(sel_col, ascending=False)
    print("\n=== BẢNG XẾP HẠNG (cross-fit trên val; chọn theo " + " + ".join(SEL_DS) + ") ===")
    print(board.round(4).to_string(index=False))
    best_single = max((r["score"] for r in RESULTS if r["kind"] in ("new", "baseline") and r["est"] <= CFG["MAX_EST_SEC"]), default=-1)
    lim_ = lambda r: CFG["MAX_EST_SEC_ENS"] if r["kind"] == "ens" else CFG["MAX_EST_SEC"]
    ELIG = [r for r in RESULTS if r["est"] <= lim_(r) and (r["kind"] in ("new", "baseline") or r["score"] >= best_single + CFG["ENS_MIN_GAIN"])]
    if not ELIG:
        print(f"!! Không tổ hợp nào ước tính ≤ {CFG['MAX_EST_SEC']:.0f}s -> lấy model nhanh nhất"); ELIG = [min(RESULTS, key=lambda r: r["est"])]
    WIN = max(ELIG, key=lambda r: r["score"])
    print(f"\n>>> CHỌN: {WIN['name']} | cross-fit {WIN['score']:.4f} | TTA {WIN.get('tta', False)} | ước tính A10 {WIN['est']:.0f}s (giới hạn 300 s)")
    board.to_csv(RUNS / f"leaderboard_{STAMP}.csv", index=False)
else:
    uniq = sorted({key_of(mb["meta"]) for r in RESULTS for mb in r["members"]})
    def pack_mode(arch):            # chế độ int8 thực sự dùng khi đóng gói model này
        if CFG["CPU_INT8"] == "smooth": return "smooth" if arch.startswith("dinov2") else "fp32"
        return "int8" if CFG["CPU_INT8"] else "fp32"
    specs = [REF] + [dict(arch=a, img_h=h, img_w=w, parts=pt, qdec=qd, cmid=list(cm), chist=ch, int8=pack_mode(a) != "fp32",
                          int8_skip=CFG["INT8_SKIP"] if pack_mode(a) == "smooth" else []) for a, h, w, pt, qd, cm, ch in uniq]
    ms = measure_costs(specs); REF_MS = ms[0]
    COST = {k: ms[1 + i] / REF_MS for i, k in enumerate(uniq)}
    print(f"Mốc (DINOv2-B int8, như bản 0.776): {REF_MS:.0f} ms/ảnh trên CPU này")
    print(pd.DataFrame({"chế độ": {k: pack_mode(k[0]) for k in uniq}, "chi phí (×mốc)": COST}).round(2).to_string())
    cost_of = lambda r: sum(COST[key_of(mb["meta"])] for mb in r["members"])

    if CFG["ENSEMBLE"]:
        singles = list(RESULTS)
        for a, b in itertools.combinations(singles, 2):
            c = cost_of(a) + cost_of(b)
            if c > CFG["MAX_REL_COST"]:
                print(f"[ens] bỏ {a['name']} + {b['name']}: chi phí {c:.2f} > {CFG['MAX_REL_COST']}"); continue
            best = None
            for w in CFG["ENS_WEIGHTS"]:
                z = w * a["z"] + (1 - w) * b["z"]
                s = evaluate(VA, sigm(z)).loc["MEAN", "Challenge_Avg"]
                if best is None or s > best[0]: best = (s, w, z)
            _, w, z = best
            RESULTS.append(dict(name=f"{a['name']} ×{w:.2f} + {b['name']} ×{1 - w:.2f}", kind="ens", z=z, minutes=0,
                                members=a["members"] + b["members"], ens_w=[w * x for x in a["ens_w"]] + [(1 - w) * x for x in b["ens_w"]]))
            print(f"[ens] {RESULTS[-1]['name']} | chi phí {c:.2f}", flush=True)

    for r in RESULTS:
        k, sc, det = crossfit(VA, r["z"], FOLDS, CFG["SHRINK_GRID"])
        r.update(k=k, score=sc[k], shrink_table=sc, detail=det, raw=evaluate(VA, sigm(r["z"])).loc["MEAN", "Challenge_Avg"], cost=cost_of(r))
        print(f"[{r['name']}] raw@0.5 {r['raw']:.4f} | cross-fit {r['score']:.4f} (shrink {k}) | chi phí {r['cost']:.2f}", flush=True)

    board = pd.DataFrame([{"model": r["name"], "loại": r["kind"], "val raw@0.5": r["raw"], "cross-fit Avg": r["score"], "shrink": r["k"],
                           **{d: r["detail"].loc[d, "Challenge_Avg"] for d in DS_NAMES if d in r["detail"].index},
                           "chi phí CPU (×bản cũ)": r["cost"], "phút train": r["minutes"]} for r in RESULTS]).sort_values("cross-fit Avg", ascending=False)
    print("\n=== BẢNG XẾP HẠNG (cross-fit = ngưỡng tìm trên nửa val này, chấm trên nửa kia) ===")
    print(board.round(4).to_string(index=False))
    best_single = max((r["score"] for r in RESULTS if r["kind"] != "ens" and r["cost"] <= CFG["MAX_REL_COST"]), default=-1)
    ELIG = [r for r in RESULTS if r["cost"] <= CFG["MAX_REL_COST"] and (r["kind"] != "ens" or r["score"] >= best_single + CFG["ENS_MIN_GAIN"])]
    ELIG = ELIG or [min(RESULTS, key=lambda r: r["cost"])]
    WIN = max(ELIG, key=lambda r: r["score"])
    print(f"\n>>> CHỌN: {WIN['name']} | cross-fit {WIN['score']:.4f} | shrink {WIN['k']} | chi phí {WIN['cost']:.2f}×")
    if WIN["kind"] == "baseline":
        print("!! Không ứng viên mới nào hơn bản cũ -> đóng gói lại bản cũ với ngưỡng cross-fit mới")
    board.to_csv(RUNS / f"leaderboard_{STAMP}.csv", index=False)

# %% [markdown]
# ## 9. Ngưỡng cuối cùng (calibrate trên toàn bộ val, áp shrink đã chọn) + (tuỳ chọn) train lại trên train+val

# %%
z_win = WIN["z"]
THR_CAL, s_in = calibrate(VS, sigm(z_win[SEL_MASK]))          # VS = toàn bộ val, trừ khi SEL_DROP_OLDTRAIN_MARKET
THR = sigm(WIN["k"] * logit(THR_CAL))
print(f"Calibrate in-sample (lạc quan): {s_in:.4f} | ước lượng trung thực (cross-fit): {WIN['score']:.4f} | shrink {WIN['k']}")
print(evaluate(VS, sigm(z_win[SEL_MASK] - logit(THR))).round(4).to_string())

FINAL_MEMBERS = [dict(mb) for mb in WIN["members"]]
THR_SOLO = None
if CFG["GPU_SUBMIT"] and len(FINAL_MEMBERS) > 1:
    # model phụ đánh dấu optional: máy chấm chậm thì run.py bỏ chúng, các ảnh còn lại dùng model chính với ngưỡng riêng của nó
    r_main = next(r for r in RESULTS if r["kind"] in ("new", "baseline") and r["members"][0]["name"] == FINAL_MEMBERS[0]["name"])
    THR_SOLO = sigm(WIN["k"] * logit(calibrate(VS, sigm(r_main["z"][SEL_MASK]))[0]))
    print(f"Ngưỡng riêng model chính ({FINAL_MEMBERS[0]['name']}) cho trường hợp bỏ model phụ: đã tính")
if CFG["FINAL_FULL"]:
    for mb in FINAL_MEMBERS:
        if mb["name"].startswith("baseline"): continue
        if ARCH_CFG.get(mb["name"], {}).get("AUTO_EP") and mb.get("meta", {}).get("EPOCHS"):   # H+: cùng số epoch với run train-only (ngưỡng mượn từ đó), AUTO_EP chỉ hạ nếu không kịp
            ARCH_CFG[mb["name"]] = dict(ARCH_CFG[mb["name"]], EPOCHS=int(mb["meta"]["EPOCHS"]))
            print(f"[{mb['name']}] FULL: {ARCH_CFG[mb['name']]['EPOCHS']} epoch như run train-only")
        out_full = train_candidate(mb["name"], pd.concat([TR, VA], ignore_index=True), VA, tag="full", full=True)
        mb["weights"] = out_full / "best_ema.pt"
        print("Dùng model FULL (train+val) với ngưỡng của run val:", mb["weights"])

# %% [markdown]
# ## 10. Đóng gói submission theo cấu trúc mẫu của BTC + tự test `run.py`

# %%
RUN_PY = r'''
# UPAR 2027 Track 1 submission (v2). Cấu trúc: run.py, metadata.yaml, model.py, assets/{config.json, weights_*.pt}
# Chốt chặn thời gian: run.py tự đo tốc độ sau mỗi batch. Nếu dự báo tổng thời gian vượt ngân sách, các ảnh còn lại chạy ở độ
# phân giải thấp hơn (cùng weights, nội suy pos_embed như research/make_lowres_zip.py; trên val 224x112 mất ~0.001, 196x98 ~0.005).
# Server đủ nhanh thì luôn ở mức 0 = đúng như bản không có chốt chặn. Log chỉ dùng ASCII để không lỗi mã hoá stderr.
# Model phụ ("optional": true trong config, cần "thresholds_solo"): nếu dự báo vượt ngân sách thì bỏ model phụ trước (các ảnh còn lại
# chỉ chạy model chính với ngưỡng riêng của nó, độ phân giải giữ nguyên); chỉ hạ độ phân giải khi đã bỏ model phụ và time_guard bật.
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
             hard_s=280.0, tail_n=3000)   # 11/10: model phụ đã bật ở tail_n ảnh cuối (gồm Synth = 50 % điểm) chỉ bị tắt lại khi dự báo vượt hard_s
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
        m = PARNet(sp["arch"], pretrained=False, img_h=sp["img_h"], img_w=sp["img_w"], parts=sp.get("parts", False), qdec=sp.get("qdec", False), cmid=tuple(sp.get("cmid", ())), chist=bool(sp.get("chist", False)))
        sd = torch.load(os.path.join(ASSETS, sp["file"]), map_location="cpu", weights_only=True)
        if any(k.endswith("::w8scale") for k in sd):              # weights int8 theo kênh (bản _w8): giải nén
            sd = {k: v.float() * sd[k + "::w8scale"].float() if k + "::w8scale" in sd else v for k, v in sd.items() if not k.endswith("::w8scale")}
        m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
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
        if sp["arch"].startswith(("dinov2", "eva02")):                 # DINOv2 đổi độ phân giải bằng nội suy pos_embed
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
    if cfg.get("time_guard") is False or os.environ.get("UPAR_GUARD_OFF") or not all("pe0" in sp for _, sp in models):
        g.update(scales=[1.0], costs=[1.0])
    g.update(k=0, n_done=0, last_t=None, last_n=0, last_k=0, calls=0, rates=[], n_tier=[0] * len(g["scales"]), groups={}, next_log=2880,
             has_opt=has_opt, opt_on=True, opt_rates=[], last_o=0.0, n_opt=0,
             shadow_on=True, sh_rates=[], last_s=0.0, t_io=0.0, t_prep=0.0, t_fwd=0.0, t_shadow=0.0, n_shadow=0)
    print(f"[run.py] torch {torch.__version__} | device {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else "")
          + f" | cpu threads {torch.get_num_threads()} | models "
          + ", ".join(f"{sp['arch']} {sp['img_h']}x{sp['img_w']} w=" + (f"{sp['w']:.2f}" if not isinstance(sp['w'], list) else f"vec{max(sp['w']):.2f}")
                      + (" crop" if sp.get("crop") else "") + (" optional" if sp.get("optional") else "") for _, sp in models)
          + f" | tta {cfg.get('tta_flip', False)} | int8 {dev.type == 'cpu' and cfg.get('cpu_int8', False)}"
          + f" | guard {g['budget_s']:.0f}s/{int(g['n_expected'])} img, {len(g['scales'])} tiers", file=sys.stderr, flush=True)
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

def _guard_step(samples):
    g = _S["guard"]; now = time.time(); n = len(samples)
    if g["last_t"] is None:
        s0 = samples[0]
        _log(f"first sample: keys {sorted(s0.keys())} | image {str(s0.get('image'))[:120]!r} | image_path {str(s0.get('image_path'))[:160]!r}")
    elif g["last_n"]:
        g["calls"] += 1
        if g["calls"] > 1:                                  # bỏ batch đầu (khởi động); tốc độ quy về chi phí mức 0, không gồm model phụ
            per = (now - g["last_t"]) / g["last_n"]
            g["rates"] = (g["rates"] + [max(per - g["last_o"] - g["last_s"], 1e-6) / g["costs"][g["last_k"]]])[-5:]
            if g["last_o"] > 0: g["opt_rates"] = (g["opt_rates"] + [g["last_o"]])[-5:]
            if g["last_s"] > 0: g["sh_rates"] = (g["sh_rates"] + [g["last_s"]])[-5:]
    if g["shadow_on"] and g["rates"] and g["sh_rates"]:      # lượt "bóng" (chỉ để đo) tắt trước mọi thứ khác nếu dự báo > 80 % ngân sách
        r_ = float(np.median(g["rates"])) * g["costs"][g["k"]] + (float(np.median(g["opt_rates"])) if (g["has_opt"] and g["opt_on"] and g["opt_rates"]) else 0.0)
        rs_, el_, left_ = float(np.median(g["sh_rates"])), now - T0, max(g["n_expected"] - g["n_done"], n)
        if el_ + (r_ + rs_) * left_ > 0.8 * g["budget_s"]:
            g["shadow_on"] = False
            _log(f"shadow OFF after {g['n_done']} images | main {1000 * r_:.2f} + shadow {1000 * rs_:.2f} ms/img | projected {el_ + (r_ + rs_) * left_:.0f}s")
    if g["has_opt"]:                                        # bỏ / bật lại model phụ (bật lại cần dư 2 %)
        # Khi tắt, mỗi ảnh chỉ chạy model chính làm dự báo "chạy cả 2 model cho phần còn lại" giảm dần -> model phụ được bật lại cho
        # đoạn cuối, dùng hết phần thời gian dư (07/10: hệ số 0.9 cũ quá chặt, server dư ~360 s mà model phụ không được bật lại).
        on = g["opt_on"]
        if os.environ.get("UPAR_GUARD_OFF"): on = True
        elif os.environ.get("UPAR_DROP_OPT"): on = False
        elif g["rates"] and g["opt_rates"]:
            r, ro = float(np.median(g["rates"])) * g["costs"][g["k"]], float(np.median(g["opt_rates"]))
            el, left = now - T0, max(g["n_expected"] - g["n_done"], n)
            lim = g.get("hard_s", g["budget_s"]) if (g["opt_on"] and left <= g.get("tail_n", 0)) else g["budget_s"] * (1.0 if g["opt_on"] else 0.98)
            on = el + (r + ro) * left <= lim
            if on != g["opt_on"]:
                _log(f"optional models {'ON' if on else 'OFF'} after {g['n_done']} images | main {1000 * r:.0f} + optional {1000 * ro:.0f} ms/img"
                     f" | projected {el + (r + ro) * left:.0f}s with, {el + r * left:.0f}s without")
        g["opt_on"] = on
    k = g["k"]
    if os.environ.get("UPAR_FORCE_TIER"): k = min(int(os.environ["UPAR_FORCE_TIER"]), len(g["scales"]) - 1)
    elif g["rates"] and len(g["scales"]) > 1 and not (g["has_opt"] and g["opt_on"]):
        r, el, left = float(np.median(g["rates"])), now - T0, max(g["n_expected"] - g["n_done"], n)
        k = len(g["scales"]) - 1
        for j in range(len(g["scales"])):                   # mức chính xác nhất còn kịp; quay lên mức tốt hơn cần dư 5 %
            if el + r * g["costs"][j] * left <= g["budget_s"] * (0.95 if j < g["k"] else 1.0): k = j; break
    if k != g["k"]:
        h, w = _size(_S["models"][0][1], k)
        _log(f"tier {g['k']} -> {k} ({h}x{w}) after {g['n_done']} images | "
             f"{1000 * float(np.median(g['rates'])) if g['rates'] else 0:.0f} ms/img at tier-0 cost")
        _set_tier(k); g["k"] = k
    g["last_t"], g["last_n"], g["last_k"] = now, n, k
    return k

def _guard_done(samples, k):
    g = _S["guard"]; c = g["groups"]
    for i, s in enumerate(samples):
        grp = _grp(s)
        if grp not in c:
            c[grp] = 0
            if len(c) <= 30: _log(f"new group {grp[:60]!r} at image #{g['n_done'] + i}")
        c[grp] += 1
    g["n_done"] += len(samples); g["n_tier"][k] += len(samples)
    if g["n_done"] >= g["next_log"] or g["n_done"] == int(g["n_expected"]):
        g["next_log"] = (g["n_done"] // 2880 + 1) * 2880
        el = time.time() - T0
        top = sorted(c.items(), key=lambda kv: -kv[1])[:8]
        n_ = max(g["n_done"], 1)
        tm = (f" | io {1000 * g['t_io'] / n_:.2f} prep {1000 * g['t_prep'] / n_:.2f} model {1000 * g['t_fwd'] / n_:.2f}"
              + (f" shadow {1000 * g['t_shadow'] / max(g['n_shadow'], 1):.2f} ({g['n_shadow']} img)" if g["n_shadow"] else "") + " ms/img")
        _log(f"{g['n_done']} images | {1000 * el / g['n_done']:.1f} ms/img avg{tm} | tier {k} | images per tier {g['n_tier']}"
             + (f" | with optional {g['n_opt']}" if g["has_opt"] else "") + f" | {len(c)} groups, top {top}")

@torch.no_grad()
def predict_batch(samples):
    if not _S: load_model()
    cfg, dev = _S["cfg"], _S["dev"]
    try: k = _guard_step(samples)
    except Exception as e: k = _S["guard"]["k"]; _log(f"guard error {e!r}")
    if "pool" not in _S:
        from concurrent.futures import ThreadPoolExecutor
        _S["pool"] = ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1))
    g = _S["guard"]; sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    use_opt = g["opt_on"] or not g["has_opt"]
    act = [(m, sp) for m, sp in _S["models"] if use_opt or not sp.get("optional")]
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
    z, wsum, t_opt, x0 = None, 0.0, 0.0, None
    # 12/10 v17r: cfg["route"] = {"tau", "tail_n"} -> model phụ chỉ chạy cho ảnh mà model chính kém chắc: u = Σ_a exp(-|z_chính - ngưỡng_solo|) >= tau
    # (tau chỉnh trên val, ~30 % ảnh). Ở tail_n ảnh cuối mọi ảnh chạy đủ như v17. Không có "route" -> y hệt bản cũ.
    route = cfg.get("route") if (use_opt and g["has_opt"]) else None
    if route is not None and (g["n_expected"] - g["n_done"]) <= int(route.get("tail_n", g.get("tail_n", 0))): route = None
    sel, zm, wm = None, None, 0.0
    for m, sp in act:
        t1 = time.time(); h, w = _size(sp, k)
        x = xs[(h, w, sp["_crop"])]
        if sp.get("wb", False): x = _grayworld_b(x)
        x = (x - sp["_mean"]) / sp["_std"]
        if bad: x[bad] = 0.0                                  # ảnh đọc lỗi: tensor 0 như _prep() cũ
        if route is not None and sp.get("optional"):
            if sel is None:
                u = torch.exp(-(zm / wm - _S["off_solo"]).abs()).sum(1); sel = u >= float(route["tau"])
            if not bool(sel.any()): continue
            x = x[sel]
        t2 = time.time()
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            zi = m(x).float()
            if cfg.get("tta_flip", False):
                zi = (zi + m(torch.flip(x, dims=[3])).float()) / 2
        sync(); g["t_fwd"] += time.time() - t2
        if x0 is None: x0 = x
        if route is not None and sp.get("optional"):
            zf = torch.zeros(len(samples), zi.shape[1], device=zi.device); zf[sel] = zi; zi = zf
        z = sp["_w"] * zi if z is None else z + sp["_w"] * zi; wsum = wsum + sp["_w"]
        if not sp.get("optional"): zm = z.clone(); wm = wsum
        if sp.get("optional"): t_opt += time.time() - t1
    g["last_o"] = t_opt / max(len(samples), 1)
    n_sh, t_sh = int(cfg.get("shadow_passes", 0) or 0), 0.0
    if n_sh and g["shadow_on"] and x0 is not None:          # chỉ để đo chi phí thêm 1 model: chạy lại model đầu trên ảnh lật, bỏ kết quả
        t3 = time.time()
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            for _ in range(n_sh): _S["models"][0][0](torch.flip(x0, dims=[3]))
        sync(); t_sh = time.time() - t3; g["t_shadow"] += t_sh; g["n_shadow"] += len(samples)
    g["last_s"] = t_sh / max(len(samples), 1)
    if route is not None:                                   # v17r: ảnh được chọn -> hệ ghép đủ + ngưỡng ghép; còn lại -> model chính + ngưỡng solo
        if sel is None: sel = torch.zeros(len(samples), dtype=torch.bool, device=zm.device)
        g["n_opt"] += int(sel.sum()); g["n_route"] = g.get("n_route", 0) + int(sel.sum())
        p = torch.where(sel[:, None], torch.sigmoid(z - _S["off"]), torch.sigmoid(zm / wm - _S["off_solo"]))
    elif use_opt:
        off = _S["off"]; g["n_opt"] += len(samples) if g["has_opt"] else 0
        p = torch.sigmoid(z - off)
    else:                                                   # chỉ model chính: logit chuẩn hoá lại theo tổng trọng số, ngưỡng riêng
        z, off = z / wsum, _S["off_solo"]
        p = torch.sigmoid(z - off)
    p = torch.nan_to_num(p, nan=0.0).clamp(0.0, 1.0).cpu().numpy()
    names = samples[0].get("attribute_names") or _S["names"]
    if list(names) != list(_S["names"]):                   # trả đúng thứ tự BTC yêu cầu
        p = p[:, [_S["names"].index(n) for n in names]]
    try: _guard_done(samples, k)
    except Exception as e: _log(f"guard error {e!r}")
    return p.tolist()

def predict_image(sample):
    return predict_batch([sample])[0]
'''
METADATA = '''# Code submissions are executed by the challenges ingestion program, which
# imports run.py and calls predict_image() / predict_batch(). Nothing in this
# file is used for inference, it exists only so the platform never treats the
# archive as a result submission. Leave it as-is.
command: python3 -c "print('Submission package ready. The ingestion program imports run.py')"
'''

@torch.no_grad()
def smoothquant_state(sd, meta, alpha):
    # SmoothQuant cho DINOv2: biến đổi tương đương trong fp32 (norm1->qkv, V->proj, norm2->fc1) để int8 động ít sai hơn.
    m = PARNet(meta["arch"], pretrained=False, img_h=meta["img_h"], img_w=meta["img_w"], parts=meta.get("parts", False), qdec=meta.get("qdec", False), cmid=tuple(meta.get("cmid", ())), chist=bool(meta.get("chist", False)), pmask=meta.get("pmask"))
    m.load_state_dict({k_: v.float() for k_, v in sd.items()}); m = m.to(dev).eval()
    st = {}
    def hook(name):
        def f(mod, inp, out):
            a = inp[0].detach().float().abs().reshape(-1, inp[0].shape[-1]).amax(0).cpu()
            st[name] = torch.maximum(st[name], a) if name in st else a
        return f
    hs = [mod.register_forward_hook(hook(n)) for n, mod in m.named_modules() if isinstance(mod, nn.Linear)]
    calib = TR.groupby("ds").head(100 if not CFG["DEBUG"] else 10).reset_index(drop=True)     # ảnh TRAIN, không dùng val/test
    for x, *_ in make_loader(calib, meta["img_h"], meta["img_w"], tuple(meta["mean"]), tuple(meta["std"]), False, 64, wb=meta.get("wb", False)):
        m(x.to(dev).float())
    for h in hs: h.remove()
    sd = {k_: v.float().clone() for k_, v in sd.items()}; C = sd["backbone.cls_token"].shape[-1]
    s_of = lambda a, w: (a.clamp(min=1e-5) ** alpha / w.abs().amax(0).clamp(min=1e-5) ** (1 - alpha)).clamp(min=1e-5)
    for b in range(len(m.backbone.blocks)):
        p = f"backbone.blocks.{b}."
        s = s_of(st[p + "attn.qkv"], sd[p + "attn.qkv.weight"])
        sd[p + "norm1.weight"] /= s; sd[p + "norm1.bias"] /= s; sd[p + "attn.qkv.weight"] *= s[None]
        s = s_of(st[p + "attn.proj"], sd[p + "attn.proj.weight"])
        sd[p + "attn.qkv.weight"][2 * C:] /= s[:, None]; sd[p + "attn.qkv.bias"][2 * C:] /= s; sd[p + "attn.proj.weight"] *= s[None]
        s = s_of(st[p + "mlp.fc1"], sd[p + "mlp.fc1.weight"])
        sd[p + "norm2.weight"] /= s; sd[p + "norm2.bias"] /= s; sd[p + "mlp.fc1.weight"] *= s[None]
    del m; gc.collect()
    return sd

def w8_state(sd):
    # int8 đối xứng theo kênh ra cho ma trận lớn (scale fp16), phần còn lại fp16; run.py giải nén khi nạp. Lệch ~0.1 % ô (kiểm 09/10 trên L3)
    q = {}
    for k_, v in sd.items():
        if v.is_floating_point() and v.ndim >= 2 and v.numel() >= 4096:
            w_ = v.float(); sc_ = w_.abs().amax(dim=tuple(range(1, w_.ndim)), keepdim=True).clamp_min(1e-8) / 127.0
            q[k_] = torch.round(w_ / sc_).clamp(-127, 127).to(torch.int8); q[k_ + "::w8scale"] = sc_.half()
        else:
            q[k_] = v.half() if v.dtype == torch.float32 else v
    return q

def build_submission(sub_dir, members, ens_w, thr, thr_cal, k, cpu_int8, tta=False, thr_solo=None, w8=False):
    shutil.rmtree(sub_dir, ignore_errors=True); (sub_dir / "assets").mkdir(parents=True)
    (sub_dir / "run.py").write_text(RUN_PY.lstrip(), encoding="utf-8")
    ex = [p for p in [REPO / "examples/task1/sample_code_submission/metadata.yaml",
                      Path(os.environ.get("UPAR_REPO", "_none_")) / "examples/task1/sample_code_submission/metadata.yaml"] if p.exists()]
    if ex: (sub_dir / "metadata.yaml").write_bytes(ex[0].read_bytes().replace(b"\r\n", b"\n"))
    else: (sub_dir / "metadata.yaml").write_text(METADATA, encoding="utf-8")
    shutil.copy(CODE / "model.py", sub_dir / "model.py")
    specs = []; any_smooth = False
    for i, (mb, w) in enumerate(zip(members, ens_w)):
        sd = {k_: v for k_, v in torch.load(mb["weights"], map_location="cpu").items() if not k_.startswith("qdec2.")}   # 14/10: zip chỉ z (run.py không dựng QDEC2)
        if cpu_int8 == "smooth" and mb["meta"]["arch"].startswith("dinov2"):
            sd = smoothquant_state(sd, mb["meta"], CFG["SQ_ALPHA"]); any_smooth = True
        torch.save(w8_state(sd) if w8 else {k_: v.half() if v.dtype == torch.float32 else v for k_, v in sd.items()}, sub_dir / f"assets/weights_{i}.pt")
        m = mb["meta"]
        specs.append(dict(arch=m["arch"], parts=bool(m.get("parts", False)), qdec=bool(m.get("qdec", False)), wb=bool(m.get("wb", False)),
                          cmid=list(m.get("cmid", [])), chist=bool(m.get("chist", False)), img_h=m["img_h"], img_w=m["img_w"], mean=list(m["mean"]), std=list(m["std"]),
                          file=f"weights_{i}.pt", w=float(w), source=str(mb["weights"]), **({"optional": True} if (thr_solo is not None and i > 0) else {}),
                          **({"crop": list(m["CROP"])} if m.get("CROP") else {})))
    json.dump({"models": specs, "tta_flip": bool(tta or CFG["TTA_FLIP"]), "attributes": ATTR, "thresholds": [float(t) for t in thr],
               "thresholds_calibrated": [float(t) for t in thr_cal], "thr_shrink": k, "cpu_int8": bool(cpu_int8) and (cpu_int8 != "smooth" or any_smooth),
               "int8_skip": CFG["INT8_SKIP"] if cpu_int8 == "smooth" else [], "smoothquant_alpha": CFG["SQ_ALPHA"] if any_smooth else None, "version": "v4",
               "time_guard": ({"budget_s": CFG["GUARD_BUDGET"], "n_expected": 29248} if CFG["GPU_SUBMIT"] else ({} if CFG.get("TIME_GUARD", False) else False)),
               **({"thresholds_solo": [float(t) for t in thr_solo]} if thr_solo is not None else {})},
              open(sub_dir / "assets/config.json", "w"), indent=1)
    zp = sub_dir.with_suffix(".zip")
    if zp.exists(): zp.unlink()
    shutil.make_archive(str(sub_dir), "zip", root_dir=sub_dir)          # nội dung sub_dir nằm ở gốc archive
    names = zipfile.ZipFile(zp).namelist()
    assert "run.py" in names and "metadata.yaml" in names and "assets/weights_0.pt" in names, names
    return zp

OUT = RUNS / f"final_{STAMP}"; OUT.mkdir(parents=True, exist_ok=True)
SUB = OUT / "submission"
ZIP = build_submission(SUB, FINAL_MEMBERS, WIN["ens_w"], THR, THR_CAL, WIN["k"], CFG["CPU_INT8"], tta=WIN.get("tta", False), thr_solo=THR_SOLO)
print("zip:", ZIP, f"{ZIP.stat().st_size / 2**20:.1f} MB |", zipfile.ZipFile(ZIP).namelist())

# test giống ingestion program: import run, load_model, predict_batch (trên máy notebook: GPU nếu có)
kk = min(256, len(VA)); os.environ["UPAR_N_EXPECTED"] = str(kk)      # chốt chặn thời gian tính theo số ảnh thử -> luôn ở mức 0
spec = importlib.util.spec_from_file_location("sub_run_v2", SUB / "run.py"); sub_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub_run); sub_run.load_model(); os.environ.pop("UPAR_N_EXPECTED", None)
samples = [{"index": i, "image": VA.image[i], "image_path": str(DATA / VA.image[i]), "attribute_names": ATTR} for i in range(kk)]
t0 = time.time(); outp = []
for j in range(0, kk, sub_run.BATCH_SIZE): outp += sub_run.predict_batch(samples[j:j + sub_run.BATCH_SIZE])
dt = (time.time() - t0) / kk; outp = np.array(outp)
assert outp.shape == (kk, 40) and np.isfinite(outp).all() and outp.min() >= 0 and outp.max() <= 1
if not CFG["FINAL_FULL"]:
    ref = sigm(z_win[:kk] - logit(THR))
    print(f"run.py OK | {dt * 1000:.1f} ms/ảnh ({sub_run._S['dev']}) | max |run.py - notebook| = {np.abs(outp - ref).max():.4f}"
          + ("  (CPU + int8 nên lệch; mục 11 sẽ đo ảnh hưởng tới điểm)" if sub_run._S["dev"].type == "cpu" and CFG["CPU_INT8"] else ""))
print("predict_image OK:", len(sub_run.predict_image(samples[0])), "giá trị")
if CFG["GPU_SUBMIT"]:
    print(f"Ước tính trên runner A10 (29 248 ảnh): ~{WIN['est']:.0f} s | giới hạn 300 s | chốt chặn run.py {CFG['GUARD_BUDGET']:.0f} s")
del sub_run; gc.collect()

# %% [markdown]
# ## 11. Chạy zip bằng torch 2.4.1 CPU (giống server) + tự quyết int8 + so tốc độ với bản cũ

# %%
TEST_PY = '''
import sys, os, time, json
import numpy as np, torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| torchvision", __import__("torchvision").__version__,
      "| numpy", np.__version__, "| cuda", torch.cuda.is_available(), flush=True)
assert torch.__version__.startswith("2.4.1"), "KHONG chay dung torch 2.4.1"
sys.path.insert(0, os.getcwd())
import run
t = time.time()
if hasattr(run, "load_model"): run.load_model()
print("load_model OK (%.1fs)" % (time.time() - t), flush=True)
paths, names = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
S = [{"index": i, "image": p, "image_path": p, "attribute_names": names} for i, p in enumerate(paths)]
bs = getattr(run, "BATCH_SIZE", 32); out = []; t = time.time()
if hasattr(run, "predict_batch"):
    for j in range(0, len(S), bs): out += [list(map(float, r)) for r in run.predict_batch(S[j:j + bs])]
else:
    out = [list(map(float, run.predict_image(s))) for s in S]
dt = (time.time() - t) / len(S)
P = np.array(out)
assert P.shape == (len(S), 40), P.shape
assert np.isfinite(P).all() and P.min() >= 0 and P.max() <= 1, "gia tri ngoai [0,1] hoac NaN"
assert len(run.predict_image(S[0])) == 40
np.save(sys.argv[3], P)
print("MS_PER_IMG %.1f" % (dt * 1000), flush=True)
print("SUBMISSION OK | %d anh | %.0f ms/anh tren CPU | ty le du doan duong %.3f" % (len(S), dt * 1000, (P > 0.5).mean()), flush=True)
'''

def check_submission_torch241(zip_path, image_paths, n=None):
    import tempfile
    py, extra = t241_python()
    base = Path(tempfile.mkdtemp()); sub = base / "submission"; sub.mkdir()
    if Path(zip_path).is_dir(): shutil.copytree(zip_path, sub, dirs_exist_ok=True)
    else: zipfile.ZipFile(zip_path).extractall(sub)
    assert (sub / "run.py").exists(), "run.py không nằm ở gốc zip"
    paths = [str(p) for p in list(image_paths)[:n]]
    json.dump(paths, open(base / "paths.json", "w")); json.dump(list(ATTR), open(base / "names.json", "w"))
    (base / "t241_test.py").write_text(TEST_PY, encoding="utf-8")
    envv = dict(os.environ, CUDA_VISIBLE_DEVICES="", UPAR_GUARD_OFF="1", **extra)     # đo tốc độ thật của mức 0, không để chốt chặn hạ độ phân giải
    r = subprocess.run([py, "-s", str(base / "t241_test.py"), str(base / "paths.json"), str(base / "names.json"), str(base / "preds.npy")],
                       cwd=sub, env=envv, capture_output=True, text=True)
    print(r.stdout[-2000:])
    if r.returncode: print(r.stderr[-6000:])
    else: print("\n".join(l for l in r.stderr.splitlines() if l.startswith("[run.py]")))
    P = np.load(base / "preds.npy") if (base / "preds.npy").exists() else None
    ms_ = float(re.search(r"MS_PER_IMG ([\d.]+)", r.stdout).group(1)) if "MS_PER_IMG" in r.stdout else None
    shutil.rmtree(base, ignore_errors=True)
    return r.returncode == 0, P, ms_

CHECK = {}
if CFG["CHECK_TORCH241"] and (IS_KAGGLE or os.environ.get("UPAR_T241_PY")): t241_python()
if CFG["CHECK_TORCH241"] and (IS_KAGGLE or os.environ.get("UPAR_T241_PY")) and not _T241.get("ok"):
    print("!!! CHƯA KIỂM TRA zip bằng torch 2.4.1 (không dựng được môi trường). Zip vẫn được xuất; PHẢI chạy "
          "research/check_zip_t241.py bằng torch 2.4.1 trên máy khác trước khi nộp.")
    CHECK.update(ok=None, note="chưa kiểm torch 2.4.1")
elif CFG["CHECK_TORCH241"] and (IS_KAGGLE or os.environ.get("UPAR_T241_PY")):
    n_per = (CFG["T241_N_PER"] if CFG["GPU_SUBMIT"] else 200) if not CFG["DEBUG"] else 20
    sub_idx = VA.groupby("ds").head(n_per).index.values; vs = VA.iloc[sub_idx].reset_index(drop=True)
    img_paths = [DATA / p for p in VA.image.values[sub_idx]]
    ok, P, ms_ = check_submission_torch241(ZIP, img_paths)
    assert ok, "Zip LỖI khi chạy bằng torch 2.4.1 - KHÔNG nộp, xem traceback phía trên"
    a_srv = evaluate(vs, P).loc["MEAN", "Challenge_Avg"]
    a_ref = evaluate(vs, sigm(z_win[sub_idx] - logit(THR))).loc["MEAN", "Challenge_Avg"] if not CFG["FINAL_FULL"] else a_srv
    print(f"Avg trên {len(vs)} ảnh val | như server (CPU, int8={CFG['CPU_INT8']}): {a_srv:.4f} | notebook: {a_ref:.4f} | chênh {a_srv - a_ref:+.4f}")
    if CFG["GPU_SUBMIT"]:
        ref_p = sigm(z_win[sub_idx] - logit(THR)) if not CFG["FINAL_FULL"] else P
        agree = float(((P > 0.5) == (ref_p > 0.5)).mean())
        CHECK.update(ok=bool(ok and agree >= 0.995), ms=ms_, int8=False, srv=a_srv, ref=a_ref, agree=agree, est_a10=WIN["est"])
        print(f"torch 2.4.1 CPU: {agree:.2%} ô dự đoán trùng notebook (GPU fp16) | ước tính A10 {WIN['est']:.0f}s / 300 s")
        print(">>> KẾT QUẢ:", "OK - có thể nộp" if CHECK["ok"] else "LỆCH / LỖI - KHÔNG NỘP, xem log phía trên")
        if ZIP.stat().st_size / 2**20 > CFG["W8_ABOVE_MB"]:
            ZIP8 = build_submission(OUT / "submission_w8", FINAL_MEMBERS, WIN["ens_w"], THR, THR_CAL, WIN["k"], False, tta=WIN.get("tta", False), thr_solo=THR_SOLO, w8=True)
            ok8, P8, _ = check_submission_torch241(ZIP8, img_paths)
            agree8 = float(((P8 > 0.5) == (ref_p > 0.5)).mean()) if (ok8 and P8 is not None) else 0.0
            CHECK.update(w8_zip=str(ZIP8), w8_mb=round(ZIP8.stat().st_size / 2**20, 1), w8_agree=agree8, w8_ok=bool(ok8 and agree8 >= 0.99))
            print(f"Bản dự phòng int8 theo kênh (dùng nếu Codabench từ chối zip {ZIP.stat().st_size / 2**20:.0f} MB): {CHECK['w8_mb']} MB "
                  f"| trùng notebook {agree8:.2%} | {'OK' if CHECK['w8_ok'] else 'KHÔNG DÙNG'}")
    else:
        CHECK.update(ok=ok, ms=ms_, int8=CFG["CPU_INT8"], srv=a_srv, ref=a_ref)
        if CFG["CPU_INT8"] and a_srv < a_ref - CFG["INT8_MAX_DROP"]:
            print(f"!! int8 làm giảm {a_ref - a_srv:.4f} > {CFG['INT8_MAX_DROP']} -> đóng gói lại với cpu_int8 = False (fp32: CHẬM, kiểm tra tốc độ bên dưới)")
            ZIP = build_submission(SUB, FINAL_MEMBERS, WIN["ens_w"], THR, THR_CAL, WIN["k"], False)
            ok, P, ms_ = check_submission_torch241(ZIP, img_paths); assert ok
            CHECK.update(ok=ok, ms=ms_, int8=False, srv=evaluate(vs, P).loc["MEAN", "Challenge_Avg"])
        # mốc so sánh công bằng: zip cấu hình bản 0.776 (DINOv2-B 252x126 int8, weights ngẫu nhiên) chạy cùng bộ test, cùng ảnh
        ref_w = WORK / "ref_weights.pt"
        torch.save({k_: v.half() for k_, v in PARNet("dinov2_vitb14_reg", img_h=252, img_w=126).state_dict().items()}, ref_w)
        ref_meta = dict(arch="dinov2_vitb14_reg", img_h=252, img_w=126, mean=list(norm_for("dinov2_vitb14_reg")[0]), std=list(norm_for("dinov2_vitb14_reg")[1]))
        REF_ZIP = build_submission(RUNS / "ref_submission", [dict(weights=ref_w, meta=ref_meta)], [1.0], np.full(40, 0.5), np.full(40, 0.5), 1.0, True)
        ok_r, _, ms_ref = check_submission_torch241(REF_ZIP, img_paths)
        ok, P2, ms_ = check_submission_torch241(ZIP, img_paths)          # đo lại zip thật ngay sau mốc, cùng điều kiện máy
        CHECK["ms"] = ms_
        rel = ms_ / ms_ref if (ok_r and ms_ref and ms_) else float("nan")
        CHECK.update(ms_ref=ms_ref, rel_cost_measured=rel)
        print(f"Tốc độ CPU (gồm đọc ảnh): zip nộp {ms_:.0f} ms/ảnh | zip mốc cấu hình 0.776 {ms_ref:.0f} ms/ảnh | tỷ lệ {rel:.2f}×")
        if CFG["CPU_INT8"] == "smooth" and CHECK.get("int8") is False and rel > CFG["MAX_REL_COST"]:
            print(f"!! Bản fp32 chậm {rel:.2f}× > {CFG['MAX_REL_COST']} -> sẽ quá 1 giờ. Quay lại bản SmoothQuant (chính xác hơn int8 thường, nhanh hơn fp32)")
            ZIP = build_submission(SUB, FINAL_MEMBERS, WIN["ens_w"], THR, THR_CAL, WIN["k"], "smooth")
            ok, P3, ms_ = check_submission_torch241(ZIP, img_paths); assert ok
            rel = ms_ / ms_ref; CHECK.update(ms=ms_, int8="smooth", srv=evaluate(vs, P3).loc["MEAN", "Challenge_Avg"], rel_cost_measured=rel)
            print(f"SmoothQuant: Avg {CHECK['srv']:.4f} | {ms_:.0f} ms/ảnh = {rel:.2f}× bản cũ")
        shutil.rmtree(RUNS / "ref_submission", ignore_errors=True); Path(str(RUNS / "ref_submission") + ".zip").unlink(missing_ok=True); ref_w.unlink(missing_ok=True)
        if rel > CFG["MAX_REL_COST"]:
            print(f"!!! KHÔNG NỘP: zip chậm {rel:.2f}× bản cũ > {CFG['MAX_REL_COST']} -> sẽ quá 1 giờ trên server (fp32 252x126 = 1.29× đã quá giờ)")
        print(">>> KẾT QUẢ:", "OK - có thể nộp" if CHECK.get("ok") else "LỖI")
else:
    print("Bỏ qua kiểm tra torch 2.4.1 (không ở Kaggle và không có UPAR_T241_PY)")

# %% [markdown]
# ## 12. Ghi log, xuất file nộp, dọn Output

# %%
tag = re.sub(r"[^A-Za-z0-9]+", "-", "+".join(mb["meta"]["arch"] for mb in FINAL_MEMBERS))[:60]
final_name = ("DEBUG_KHONG_NOP_" if CFG["DEBUG"] else "SUBMIT_") + (f"gpu_{tag}{'_tta' if WIN.get('tta') else ''}_{STAMP}.zip" if CFG["GPU_SUBMIT"] else f"v2_{tag}_{STAMP}.zip")
dst = WORK / final_name; shutil.copy(ZIP, dst)
if CHECK.get("w8_ok"):
    dst8 = WORK / final_name.replace(".zip", "_w8.zip"); shutil.copy(CHECK["w8_zip"], dst8)
    print(">>> BẢN DỰ PHÒNG NHỎ (int8 theo kênh):", dst8, f"{dst8.stat().st_size / 2**20:.0f} MB")
summary = {"stamp": STAMP, "winner": WIN["name"], "crossfit": WIN["score"], "shrink": WIN["k"], "cost_est": WIN["cost"],
           "final_full": CFG["FINAL_FULL"], "zip": final_name, "zip_mb": round(dst.stat().st_size / 2**20, 1),
           **{f"check_{k}": v for k, v in CHECK.items()}, **{f"cf_{r['name']}": r["score"] for r in RESULTS}}
log = RUNS / "experiments_v2.csv"
pd.DataFrame([summary]).to_csv(log, mode="a", header=not log.exists(), index=False)
json.dump({**summary, "CFG": CFG, "ARCH_CFG": ARCH_CFG}, open(OUT / "summary.json", "w"), indent=1, default=str)

if CFG["CLEAN_OUTPUT"] and IS_KAGGLE:
    for p in [WORK / "data", WORK / "upar", WORK / "torch241", WORK / "py311_t241", WORK / "baselines", CODE]:
        if p.is_symlink(): p.unlink()
        elif p.exists(): shutil.rmtree(p, ignore_errors=True)
    for r in RUNS.iterdir():
        if r.is_dir():
            if "_debug" in r.name and not os.environ.get("UPAR_KEEP_DEBUG"): shutil.rmtree(r, ignore_errors=True); continue   # 14/10: smoke Colab giữ lại
            shutil.rmtree(r / "submission", ignore_errors=True); shutil.rmtree(r / "submission_w8", ignore_errors=True)
    if not CFG["DEBUG"]:
        for z in WORK.glob("DEBUG_KHONG_NOP_*.zip"): z.unlink()

print("\n" + "=" * 70)
print(board.round(4).to_string(index=False))
if CFG["GPU_SUBMIT"]:
    print(f"\nMODEL NỘP: {WIN['name']} | cross-fit ({' + '.join(SEL_DS)}) {WIN['score']:.4f} | TTA {WIN.get('tta', False)} "
          f"| ước tính A10 {WIN['est']:.0f}s | torch 2.4.1 trùng {CHECK.get('agree', float('nan')):.2%}")
else:
    print(f"\nMODEL NỘP: {WIN['name']} | cross-fit {WIN['score']:.4f} | shrink {WIN['k']} | int8 {CHECK.get('int8', CFG['CPU_INT8'])} "
          f"| chi phí ước tính {WIN['cost']:.2f}× | đo được {CHECK.get('rel_cost_measured', float('nan')):.2f}× bản cũ")
print(">>> FILE ĐỂ NỘP:", dst)
