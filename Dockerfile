FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /tmp/nsr

RUN pip install /tmp/nsr && rm -rf /tmp/nsr

ENTRYPOINT ["sh", "-c"]
CMD ["nsr-run"]
