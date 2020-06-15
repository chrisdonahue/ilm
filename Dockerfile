FROM nvidia/cuda:10.0-cudnn7-runtime
RUN apt update && \
    apt install -y bash \
                   build-essential \
                   curl \
                   wget \
                   python3 \
                   python3-pip \
                   ca-certificates \
                   software-properties-common && \
    rm -rf /var/lib/apt/lists
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
    nltk==3.4.5 \
    numpy==1.18.2 \
    regex==2020.5.14 \
    torch==1.4.0 \
    tokenizers==0.5.2 \
    transformers==2.7.0 \
    tqdm==4.46.0
RUN python -c "import nltk; nltk.download('punkt')"
