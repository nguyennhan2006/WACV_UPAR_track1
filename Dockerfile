FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace/upar-t1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Example: jupyter nbconvert --to notebook --execute notebooks/upar_e2e_08017.ipynb
# (set UPAR_STAGES and mount the data first, see README Section 7)
CMD ["bash"]
