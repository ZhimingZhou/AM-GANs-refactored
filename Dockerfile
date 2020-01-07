FROM  tensorflow/tensorflow:1.5.0-devel-gpu

RUN apt update
RUN apt install --yes software-properties-common
RUN apt install --yes python3-pip sudo git locales wget libssl-dev openssl
RUN locale-gen "en_US.UTF-8"
RUN update-locale LC_ALL="en_US.UTF-8"

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip3 install tensorflow-gpu==1.5.0 Pillow scipy==1.1.0 scikit-learn matplotlib==2.1.2

ENV USER developer

RUN echo "${USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/${USER}
RUN chmod u+s /usr/sbin/useradd \
   && chmod u+s /usr/sbin/groupadd
ENV HOME /home/${USER}
ENV SHELL /bin/bash
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV TERM xterm-256color

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]

WORKDIR /workspace
