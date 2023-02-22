FROM nvidia/cuda:11.6.0-devel-ubuntu20.04
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

ENV TZ=Asia/Seoul
ENV DEBIAN_FRONTEND noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.6"

RUN rm -rf /workspace
WORKDIR /workspace

RUN apt-get update
# RUN apt-get install software-properties-common
# RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get install -y libgl1-mesa-glx wget bzip2 ca-certificates  build-essential libssl-dev openssl libglib2.0-0 libxext6 libsm6 libxrender1 mercurial subversion libffi-dev && apt-get clean 
RUN apt-get install -y fontconfig libfreetype6 xfonts-75dpi xfonts-base
RUN apt-get install python3-dev
# RUN apt-get install -y python-setuptools
# RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
ADD requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt
RUN pip install cython
RUN pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
WORKDIR /workspace

# ADD models /workspace/models
# ADD inference /workspace/inference
# ADD utils /workspace/utils
# ADD runs /workspace/runs
# ADD detect_web.py /workspace/detect_web.py
ENTRYPOINT ["python3","detect.py"]





