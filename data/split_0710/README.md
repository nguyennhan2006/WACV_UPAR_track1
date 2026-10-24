# Split "0710"

Our train/validation re-split of the UPAR 2027 Track 1 training data, used for every member, fusion weight and threshold of both entries.

- `train_images.txt`: 111,296 images. `val_images.txt`: 19,780 images.
- One image path per line, written exactly as in the first column of the official `gt.csv`.
- Labels are not included; take them from the official annotations.

How it was made (`training/x2f_cmask/make_resplit.py` rebuilds it):

1. Pool the official train and validation sets.
2. For each source dataset, draw 15% as validation, in groups so that one identity never falls on both sides: Market-1501 by person ID, PA-100K by blocks of 20 consecutive indices, PETA by blocks of 5.
3. Groups are assigned by iterative stratification over the 40 attributes, seed 0.
