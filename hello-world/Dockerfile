# example on how to build docker:
# DOCKER_BUILDKIT=1 docker build -t benjamintanweihao/hello-world hello-world -f Dockerfile --build-arg user_id=`id -u`

# example on tagging and pushing to a (local) repository
# docker tag benjamintanweihao/hello-world:latest  localhost:32000/benjamintanweihao/hello-world:latest
# docker push localhost:32000/benjamintanweihao/hello-world:latest

# example on how to run:
# docker run -it benjamintanweihao/hello-world /bin/bash

FROM nvcr.io/nvidia/tensorflow:19.11-tf1-py3
LABEL MAINTAINER "Benjamin Tan <benjamintanweihao AT gmail.com>"
SHELL ["/bin/bash", "-c"]

# Set the locale

RUN  echo 'Acquire {http::Pipeline-Depth "0";};' >> /etc/apt/apt.conf

# Set the locale
RUN DEBIAN_FRONTEND="noninteractive"
RUN apt-get update  && apt-get -y install --no-install-recommends locales && locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN apt-get install -y --no-install-recommends \
    wget \
    git \
    libgl1-mesa-glx \
    openssh-client && \
    rm -rf /var/lib/apt/lists/*

# install conda
WORKDIR /tmp

# install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# cleanup
RUN rm -rf /workspace/{nvidia,docker}-examples && rm -rf /usr/local/nvidia-examples && \
    rm /tmp/requirements.txt

# make /bin/sh symlink to bash instead of dash:
RUN echo "dash dash/sh boolean false" | debconf-set-selections && \
    DEBIAN_FRONTEND=noninteractive dpkg-reconfigure dash

RUN apt-get update && apt-get -y install sudo

# add a user so as not to run as root
ARG user_id
RUN useradd -ms /bin/bash -u ${user_id} -g root -G sudo -p $(openssl passwd -1 password) dev
RUN chown dev /home/dev

WORKDIR /workspace/hello-world
ENV PYTHONPATH /workspace/hello-world:$PYTHONPATH
COPY mnist.py .

# switch to default user
USER dev

