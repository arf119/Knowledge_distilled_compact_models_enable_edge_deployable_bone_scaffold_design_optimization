FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
WORKDIR /workspace
COPY requirements.txt pyproject.toml ./
COPY compact_scaffold ./compact_scaffold
RUN pip install --no-cache-dir .
ENTRYPOINT ["python", "-m", "compact_scaffold.commands.train"]

