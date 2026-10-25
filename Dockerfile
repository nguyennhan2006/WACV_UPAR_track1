FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace/upar-t1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Inference (README Section 9): the container runs inference/predict.py, for example
#   docker run --rm --gpus all -v /path/to/test/images:/data:ro \
#     -v "$PWD/packages:/workspace/upar-t1/packages" -v "$PWD/out:/out" \
#     upar-t1 --package x2f_cmask --images /data --out /out/predictions.csv
# Training (README Sections 7-8): docker run --rm -it --gpus all --entrypoint bash upar-t1
ENTRYPOINT ["python", "-u", "inference/predict.py"]
CMD ["--help"]
