FROM docker:29.7.2-cli@sha256:3f4743208d2338c934d7b8bcfbe1bb54c0b2355c510ad5e0f31c0c4a54bd704e AS docker_cli
FROM python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/chimera-eval
COPY --from=docker_cli /usr/local/bin/docker /usr/local/bin/docker
COPY requirements.lock .
RUN pip install --no-cache-dir --no-deps -r requirements.lock
# EvalPlus' code-generation dependencies (torch, cloud clients) are intentionally not used.
RUN python -c "from ifbench import instructions_registry; import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
COPY eval_stack ./eval_stack
COPY source_revisions.json ./source_revisions.json
COPY eval_entrypoint.sh ./eval_entrypoint.sh
ENTRYPOINT ["bash", "/opt/chimera-eval/eval_entrypoint.sh"]
