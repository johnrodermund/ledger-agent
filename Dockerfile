FROM python:3.12.16-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src

# Install the pinned dependencies only; source is mounted or copied below. constraints.txt
# pins everything they pull in, so a rebuild gets the same packages.
COPY pyproject.toml constraints.txt ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt -c constraints.txt

COPY src/ src/
COPY tests/ tests/
COPY evals/ evals/
