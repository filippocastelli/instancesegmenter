FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu18.04

RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && apt-get -y update \
    && add-apt-repository universe

RUN rm -r /var/lib/apt/lists/* && apt update
RUN apt-get -y update
RUN apt-get -y install build-essential
RUN apt-get -y install python3.8 python3.8-distutils python3.8-dev python3.8-venv curl git && update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py
RUN apt-get update && apt-get -y install --no-install-recommends ocl-icd-libopencl1 clinfo && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
RUN python -m venv instancesegmenter_env && . instancesegmenter_env/bin/activate

ARG CACHEBUSTER=1
# installing my fork of ZetaStitcher
RUN git clone https://github.com/filippocastelli/ZetaStitcher.git && cd ZetaStitcher && pip install -r requirements.txt && pip install -e . && cd ../

# installing latest skimage
RUN git clone https://github.com/scikit-image/scikit-image.git && cd scikit-image && git checkout b1c485ef3a483cc06788537bb3807652dbb61961 && pip install -r requirements.txt && pip install -e . && cd ../

#COPY requirements /requirements
#RUN pip install -r /requirements && rm /requirements && rm -rf /root/.cache/pip/

COPY . /instance_segmenter/
RUN cd /instance_segmenter/ && pip install . && cd ../

CMD ["python"]