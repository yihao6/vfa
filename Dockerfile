FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /tmp/nsr

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y # install cv2 dependencies

RUN pip install /tmp/nsr && rm -rf /tmp/nsr

ENTRYPOINT ["sh", "-c"]
CMD ["nsr-run"]
