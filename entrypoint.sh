#!/bin/bash -e

USER_ID=$(id -u)
GROUP_ID=$(id -g)

if [ x"$GROUP_ID" != x"0" ]; then
    groupadd -g $GROUP_ID $USER
fi

if [ x"$USER_ID" != x"0" ]; then
    useradd -d /home/$USER -m -s /bin/bash -u $USER_ID -g $GROUP_ID $USER
fi

sudo chmod u-s /usr/sbin/useradd
sudo chmod u-s /usr/sbin/groupadd

sudo chown -R ${USER}:${GROUP_ID} /home/${USER}
sudo chown -R ${USER}:${GROUP_ID} /workspace
echo "${USER}:USER" | sudo chpasswd

exec $@