import math
import os
import pprint

import matplotlib
import numpy as np
import scipy.misc
import tensorflow as tf
import warnings
import random

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json
# import msgpack
# import pickle
from shutil import *
from tensorflow.python.client import device_lib

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])


def allocate_gpu(GPU=-1):

    if GPU == -1:
        try:
            import common.GPUtil as GPU
            GPU_ID = GPU.getFirstAvailable(order='memory', maxLoad=0.1, maxMemory=0.5)[0]
        except:
            GPU_ID = 0
    else:
        GPU_ID = GPU
        
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    # os.system('sudo -H sh /newNAS/Workspaces/CVGroup/zmzhou/upgrade_sys.sh')
    # os.system('sudo -H pip3 install --upgrade pip')
    # os.system('sudo -H pip3 install --upgrade setuptools')
    # os.system('sudo -H pip3 install Pillow scipy matplotlib==2.1.2 scikit-learn')

    return GPU_ID

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def ini_model(sess):
    sess.run(tf.global_variables_initializer())


def save_model(saver, sess, checkpoint_dir, step=None):
    makedirs(checkpoint_dir)
    model_name = "model"
    saver.save(sess, checkpoint_dir + model_name, global_step=step)


def load_model(saver, sess, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        return True
    else:
        return False


from functools import reduce
import operator

def prod(iterable):
    return reduce(operator.mul, iterable, 1)



def mean(x):
    try:
        return np.mean(x).__float__()
    except:
        return 0.

def std(x):
    try:
        return np.std(x).__float__()
    except:
        return 0.

def dotK(x, y):
    xn = x / np.sqrt(np.sum(x * x, 1, keepdims=True))
    yn = y / np.sqrt(np.sum(y * y, 1, keepdims=True))
    return np.sum(xn * yn, 1)


def get_cross_entropy(prob, ref=None):
    n = prob.shape[-1]
    if ref is None:
        ref = np.ones_like(prob) / n
    return -np.sum(ref * np.log(prob * (1.0 - n * 1e-8) + 1e-8), len(prob.shape) - 1)


def copydir(src, dst):
    if os.path.exists(dst):
        removedirs(dst)
    copytree(src, dst)


def remove(path):
    if os.path.exists(path):
        os.remove(path)


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def removedirs(path):
    if os.path.exists(path):
        rmtree(path)


def str_flags(flags):
    p = ''
    for key in np.sort(list(flags.keys())):
        p += str(key) + ':' + str(flags.get(key)._value) + '\n'
    return p


def get_image(image_path, iCenterCropSize, is_crop=True, resize_w=64, is_grayscale=False):
    return transform(imread(image_path, is_grayscale), iCenterCropSize, is_crop, resize_w)


def save_images(images, size, path):
    if images.shape[3] == 1:
        images = np.concatenate([images, images, images], 3)
    return scipy.misc.toimage(merge(images, size), cmin=-1, cmax=1).save(path)


def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def imresize(image, resize=1):
    h, w = image.shape[0], image.shape[1]
    img = np.zeros((h * resize, w * resize, image.shape[2]))
    for i in range(h * resize):
        for j in range(w * resize):
            img[i, j] = image[i // resize, j // resize]
    return img


def merge(images, size, resize=3):
    h, w = images.shape[1] * resize, images.shape[2] * resize
    img = np.zeros((h * size[0], w * size[1], images.shape[3]))
    assert size[0] * size[1] == images.shape[0]
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = imresize(image, resize)
    return img


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    h, w = x.shape[:2]

    if crop_w is None:
        crop_w = crop_h
    if crop_h == 0:
        crop_h = crop_w = min(h, w)

    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(x[j:j + crop_h, i:i + crop_w],
                               [resize_w, resize_w])


def batch_resize(images, newHeight, newWidth):
    images_resized = np.zeros([images.shape[0], newHeight, newWidth, 3])
    for idx, image in enumerate(images):
        if (images.shape[3] == 1):
            image_3c = []
            image_3c.append(image)
            image_3c.append(image)
            image_3c.append(image)
            image = np.concatenate(image_3c, 2)
        images_resized[idx] = scipy.misc.imresize(image, [newHeight, newWidth], 'bilinear')
    return images_resized


def clip_truncated_normal(mean, stddev, shape, minval=None, maxval=None):
    if minval == None:
        minval = mean - 2 * stddev
    if maxval == None:
        maxval = mean + 2 * stddev
    return np.clip(np.random.normal(mean, stddev, shape), minval, maxval)


def transform(image, npx=64, is_crop=True, resize_w=64):
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return (np.array(cropped_image) - 127.5) / 128


def inverse_transform(images):
    return (images + 1.) / 2.


def plot(data, legend, save_path):
    if len(data) <= 3:
        plt_nx = 1
        plt_ny = len(data)
    elif len(data) == 4:
        plt_ny = 2
        plt_nx = 2
    else:
        plt_ny = 3
        plt_nx = int(np.ceil(len(data) / 3.0))

    f = plt.figure()
    size = f.get_size_inches()
    f.set_size_inches(size[0] * plt_ny, size[1] * plt_nx)

    for i in range(len(data)):
        plt.subplot(plt_nx, plt_ny, i + 1)
        plt.plot(data[i])
        plt.ylabel(legend[i])
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


class Tweet(object):
    def __init__(self, text=None, userId=None, timestamp=None, location=None):
        self.text = text
        self.userId = userId
        self.timestamp = timestamp
        self.location = location

    def toJSON(self):
        return json.dumps(self.__dict__)

    @classmethod
    def fromJSON(cls, data):
        return cls(**json.loads(data))

    ##def toMessagePack(self):
    #    return msgpack.packb(self.__dict__)

    # @classmethod
    # def fromMessagePack(cls, data):
    #    return cls(**msgpack.unpackb(data))


class qlogger(object):

    def __init__(self):
        self.minq = 1e20
        self.maxq = -1e20
        self.cur_minq = 1e20
        self.cur_maxq = 1e20


def displayQvalue(qvalues):
    output = ''
    for i in range(len(qvalues)):
        output += '%7.3f  ' % (qvalues[i])
    print(output)


def collect(X, x, len):
    if isinstance(x, np.ndarray):
        if x.shape.__len__() == 1:
            x = x.reshape((1,) + x.shape)
        return x if X is None else np.concatenate([X, x], 0)[-len:]
    else:
        return [x] if X is None else (X + [x])[-len:]


NO_OPS = 'NO_OPS'


def _l2normalize(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_normed_weight(W, u=None, num_iters=1, update_collection=None, with_sigma=False):
    # Usually num_iters = 1 will be enough
    W_shape = W.shape.as_list()
    W_reshaped = tf.reshape(W, [-1, W_shape[-1]])
    if u is None:
        u = tf.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    def power_iteration(i, u_i, v_i):
        v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
        u_ip1 = _l2normalize(tf.matmul(v_ip1, W_reshaped))
        return i + 1, u_ip1, v_ip1

    _, u_final, v_final = tf.while_loop(
        cond=lambda i, _1, _2: i < num_iters,
        body=power_iteration,
        loop_vars=(tf.constant(0, dtype=tf.int32),
                   u, tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]]))
    )
    if update_collection is None:
        warnings.warn('Setting update_collection to None will make u being updated every W execution. This maybe undesirable'
                      '. Please consider using a update collection instead.')
        sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
        # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
        W_bar = W_reshaped / sigma
        with tf.control_dependencies([u.assign(u_final)]):
            W_bar = tf.reshape(W_bar, W_shape)
    else:
        sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
        # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
        W_bar = W_reshaped / sigma
        W_bar = tf.reshape(W_bar, W_shape)
        # Put NO_OPS to not update any collection. This is useful for the second call of discriminator if the update_op
        # has already been collected on the first call.
        if update_collection != NO_OPS:
            tf.add_to_collection(update_collection, u.assign(u_final))

    if with_sigma:
        return W_bar, sigma
    else:
        return W_bar


def get_name(layer_name, cts):

    if not layer_name in cts:
        cts[layer_name] = 0

    name = layer_name + '_' + str(cts[layer_name])
    cts[layer_name] += 1

    return name


def shuffle_images(images):
    rand_indexes = np.random.permutation(images.shape[0])
    shuffled_images = images[rand_indexes]
    return shuffled_images


def shuffle_images_and_labels(images, labels):
    rand_indexes = np.random.permutation(images.shape[0])
    shuffled_images = images[rand_indexes]
    shuffled_labels = labels[rand_indexes]
    return shuffled_images, shuffled_labels


def data_gen_random(data, num_sample):
    while True:
        num_data = len(data)
        data_index = np.random.choice(num_data, num_sample, replace=True, p=num_data * [1 / num_data])
        yield data[data_index]


def data_gen_epoch(datas, batch_size, func=None, epoch=None):
    cur_epoch = 0
    while True:
        np.random.shuffle(datas)
        for i in range(len(datas) // batch_size):
            if func is None:
                yield datas[i * batch_size:(i + 1) * batch_size]
            else:
                yield func(datas[i * batch_size:(i + 1) * batch_size])
        cur_epoch += 1
        if epoch is not None:
            if cur_epoch >= epoch:
                break


def labeled_data_gen_random(data, labels, num_sample):
    while True:
        num_data = len(data)
        index = np.random.choice(num_data, num_sample, replace=True, p=num_data * [1 / num_data])
        yield data[index], labels[index]


def labeled_data_gen_epoch(datas, labels, batch_size, func=None, epoch=None):
    cur_epoch = 0
    while True:
        rng_state = np.random.get_state()
        np.random.shuffle(datas)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)
        for i in range(len(datas) // batch_size):
            if func is None:
                yield (datas[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size])
            else:
                yield (func(datas[i * batch_size:(i + 1) * batch_size]), labels[i * batch_size:(i + 1) * batch_size])
        cur_epoch += 1
        if epoch is not None:
            if cur_epoch >= epoch:
                break


def random_augment_image(image, pad=4, data_format="NCHW"):

    if data_format=="NHWC":
        image = np.transpose(image, [2,0,1])

    init_shape = image.shape
    new_shape = [init_shape[0],
                 init_shape[1] + pad * 2,
                 init_shape[2] + pad * 2]
    zeros_padded = np.zeros(new_shape)
    zeros_padded[:, pad:init_shape[1] + pad, pad:init_shape[2] + pad] = image

    init_x = np.random.randint(0, pad * 2)
    init_y = np.random.randint(0, pad * 2)
    cropped = zeros_padded[:,
        init_x: init_x + init_shape[1],
        init_y: init_y + init_shape[2]]

    flip = random.getrandbits(1)
    if flip:
        cropped = cropped[:, :, ::-1]

    if data_format=="NHWC":
        cropped = np.transpose(cropped, [1,2,0])

    return cropped


def random_augment_image2(image, pad=4, data_format="NHWC"):

    if data_format=="NCHW":
        image = np.transpose(image, [1,2,0])

    init_shape = image.shape
    new_shape = [init_shape[0] + pad * 2,
                 init_shape[1] + pad * 2,
                 init_shape[2]]
    zeros_padded = np.zeros(new_shape)
    zeros_padded[pad:init_shape[0] + pad, pad:init_shape[1] + pad, :] = image

    init_x = np.random.randint(0, pad * 2)
    init_y = np.random.randint(0, pad * 2)
    cropped = zeros_padded[
        init_x: init_x + init_shape[0],
        init_y: init_y + init_shape[1],
        :]

    flip = random.getrandbits(1)
    if flip:
        cropped = cropped[:, ::-1, :]

    if data_format=="NCHW":
        cropped = np.transpose(cropped, [2,0,1])

    return cropped


def random_augment_all_images(initial_images, pad=4, data_format="NCHW"):
    new_images = np.zeros(initial_images.shape)
    for i in range(initial_images.shape[0]):
        new_images[i] = random_augment_image(initial_images[i], pad=pad, data_format=data_format)
    return new_images