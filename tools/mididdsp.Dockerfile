# MIDI-DDSP's released runtime is x86_64 TensorFlow 2.7.  Keep it outside the
# Quill runtime; this image is only an offline teacher-control exporter.
FROM --platform=linux/amd64 tensorflow/tensorflow:2.7.4

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir "midi-ddsp==0.2.6"
RUN midi_ddsp_download_model_weights

WORKDIR /work

