sudo -H apt-get update
sudo -H apt-get install -y python3-pip
sudo -H pip3 install --upgrade pip
sudo -H pip3 install --upgrade setuptools

sudo -H pip3 install /newNAS/Share/GPU_Server/tensorflow_gpu-1.5.0rc1-cp35-cp35m-manylinux1_x86_64.whl
sudo -H pip3 install Pillow scipy scikit-learn matplotlib==2.1.2

sudo sh /newNAS/Share/GPU_Server/cuda_9.0.176_384.81_linux.run --silent --toolkit
echo 'export PATH=$PATH:/usr/local/cuda/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' >> ~/.bashrc
. ~/.bashrc

tar xf /newNAS/Share/GPU_Server/cudnn-9.0-linux-x64-v7.tgz -C /tmp
sudo cp -r /tmp/cuda/* /usr/local/cuda/