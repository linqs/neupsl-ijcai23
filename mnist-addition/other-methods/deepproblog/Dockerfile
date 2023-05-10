# A good statring point for non-PSL images.

FROM ubuntu:20.04

ARG USER=linqs
ARG PASSWORD=linqs
ARG UID=1000
ARG GID=1000

RUN groupadd -f --non-unique --gid ${GID} ${USER}
RUN useradd --create-home --uid ${UID} --gid ${GID} --password ${PASSWORD} ${USER}
RUN echo "${USER}:${PASSWORD}" | chpasswd

WORKDIR /home/${USER}

RUN mkdir temp

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

# Root-level operations.

RUN ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime
RUN apt-get -y install tzdata locales

RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen
RUN echo "LANG=en_US.UTF-8" > /etc/locale.conf
RUN locale-gen

RUN apt-get install -y sudo
RUN usermod -a -G sudo ${USER}

RUN apt-get install -y git maven vim sudo time tmux wget curl
RUN apt-get install -y python3-pip python3-setuptools

# Experiment-specific.
RUN apt-get -y install software-properties-common swi-prolog libjpeg8-dev  zlib1g-dev
RUN apt-add-repository -y ppa:swi-prolog/stable
RUN apt-get update
RUN apt-get -y install swi-prolog

# Cleanup permissions.
RUN chown -R ${UID}:${GID} .

# User-level operations.

# Set default user.
USER ${UID}:${GID}

RUN git clone https://github.com/eriq-augustine/skeletons.git
RUN cp skeletons/_.bashrc .bashrc
RUN cp skeletons/_.vimrc .vimrc
RUN cp skeletons/_.bash_aliases .bash_aliases
RUN cp skeletons/_.bash_profile .bash_profile
RUN cp skeletons/_.gitconfig .gitconfig

# Experiment operations.

RUN echo 'export PATH=$PATH:$HOME/.local/bin' >> $HOME/.bashrc

RUN pip3 install --user Cython
RUN pip3 install --user \
    'problog==2.2.2' \
    'git+https://github.com/ML-KULeuven/pyswip@2b1e14f068f474afe10c81eff074ed4165cfe7a6#egg=pyswip' \
    'pysdd==0.2.10' \
    'torch==1.10.0' \
    'torchvision==0.11.1'

RUN git clone https://github.com/cfpryor/deepproblog.git
RUN pip3 install ~/deepproblog
