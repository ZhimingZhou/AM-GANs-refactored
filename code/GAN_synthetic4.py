import os, sys, locale
from os import path
sys.path.append(path.dirname(path.abspath(__file__)))
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.system('sudo -H sh ../upgrade_sys.sh')
os.system('sudo -H apt-get remove python3-matplotlib -y')
os.system('sudo -H pip3 install --upgrade pip')
os.system('sudo -H pip3 install --upgrade setuptools')
os.system('sudo -H pip3 install Pillow scipy matplotlib==2.1.2 scikit-learn')

import time
from common.data_loader import *
import sklearn.datasets
from functools import partial
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams, colorbar, colors

cfg = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("sDataSet", "toy", "cifar10, mnist, toy")
tf.app.flags.DEFINE_string("sResultTag", "test4", "your tag for each test case")

tf.app.flags.DEFINE_string("sResultDir", "../result/", "where to save the checkpoint and sample")
tf.app.flags.DEFINE_string("sSourceDir", "../code/", "")

tf.app.flags.DEFINE_integer("n", 2, "")
tf.app.flags.DEFINE_string("sTitle", "", "")

tf.app.flags.DEFINE_string("generator", 'generator_dcgan', "")
tf.app.flags.DEFINE_string("discriminator", 'discriminator_mlp_dense', "")

tf.app.flags.DEFINE_boolean("bWGAN", False, "")
tf.app.flags.DEFINE_boolean("bWGANs", False, "")
tf.app.flags.DEFINE_boolean("bLSGAN", False, "")

tf.app.flags.DEFINE_boolean("bLip", False, "")
tf.app.flags.DEFINE_boolean("bMaxGP", True, "")

tf.app.flags.DEFINE_float("fWeightGP", 1.0, "")
tf.app.flags.DEFINE_float("fWeightZero", 0.0, "")

tf.app.flags.DEFINE_boolean("bUseSN", False, "")
tf.app.flags.DEFINE_boolean("bUseSNK", False, "")

################################################# Learning Process ###########################################################################################

tf.app.flags.DEFINE_integer("iIterCheckpoint", 10000, "")
tf.app.flags.DEFINE_integer("iSamplesEvaluate", 50000, "")
tf.app.flags.DEFINE_boolean("bLoadCheckpoint", False, "bLoadCheckpoint")
tf.app.flags.DEFINE_boolean("bLoadForEvaluation", False, "bLoadForEvaluation")

tf.app.flags.DEFINE_integer("iMaxIter", 300000, "")
tf.app.flags.DEFINE_integer("iBatchSize", 128, "")

tf.app.flags.DEFINE_float("fLrDecay", 0.1, "")
tf.app.flags.DEFINE_integer("iLrStep", 100000, "")
tf.app.flags.DEFINE_boolean("bLrStair", False, "")

tf.app.flags.DEFINE_float("fLrIniD", 0.0001, "")
tf.app.flags.DEFINE_string("oOptD", 'adam', "adam, sgd, mom")

tf.app.flags.DEFINE_float("fBeta1D", 0.0, "")
tf.app.flags.DEFINE_float("fBeta2D", 0.999, "")
tf.app.flags.DEFINE_float("fEpsilonD", 1e-8, "")

tf.app.flags.DEFINE_float("fL2Weight", 0.0, "")

##################################################### Network Structure #######################################################################################

tf.app.flags.DEFINE_integer("iDimsC", 3, "")
tf.app.flags.DEFINE_integer("iMinSizeD", 4, "")
tf.app.flags.DEFINE_integer("iGrowthRateD", 256, "")
tf.app.flags.DEFINE_integer("iLayerPerBlockD", 8, "")

tf.app.flags.DEFINE_float("fDimIncreaseRate", 2.0, "")
tf.app.flags.DEFINE_boolean("bc", False, "")

tf.app.flags.DEFINE_float("fLayerNoiseD",   0.00, "")
tf.app.flags.DEFINE_float("fLayerDropoutD", 0.00, "")

tf.app.flags.DEFINE_float("fIniScale", 1.0, "")
tf.app.flags.DEFINE_string("oActD", 'relu', "relu, lrelu, elu")
tf.app.flags.DEFINE_string("oBnD", 'none', "bn, ln, none")
tf.app.flags.DEFINE_string("oDownsize", 'avgpool', "avgpool, maxpool")

tf.app.flags.DEFINE_boolean("bAugment", False, "")
tf.app.flags.DEFINE_boolean("bUseWN", False, "")

cfg(sys.argv)

############################################################################################################################################

if cfg.oActD == 'relu':
    cfg.fIniScale = 2.0

from common.ops import *
set_data_format('NCHW')
set_enable_wn(cfg.bUseWN)
set_enable_sn(cfg.bUseSN)
set_enable_snk(cfg.bUseSNK)
set_ini_scale(cfg.fIniScale)
set_enable_bias(cfg.oBnD == 'none')


def transform(data, mean=(0, 0), size=1.0, rot=0.0, hflip=False, vflip=False):

    data *= size

    rotMatrix = np.array([[np.cos(rot), -np.sin(rot)],
                          [np.sin(rot), np.cos(rot)]])
    data = np.matmul(data, rotMatrix)

    if hflip:
        data[:, 0] *= -1

    if vflip:
        data[:, 1] *= -1

    data += mean

    return data


def sqaure_generator(num_sample, noise, transform):
    while True:
        x = np.random.rand(num_sample) - 0.5
        y = np.random.rand(num_sample) - 0.5
        data = np.asarray([x, y]).transpose().astype('float32') * 2.0
        data += noise * np.random.randn(num_sample, 2)
        data = transform(data)
        yield data


def discrete_generator(num_sample, noise, transform, num_meta=2):
    meta_data = np.asarray([[i, j] for i in range(-num_meta, num_meta+1,1) for j in range(-num_meta, num_meta+1, 1)]) / float(num_meta)
    while True:
        idx = np.random.random_integers(0, len(meta_data)-1, num_sample)
        data = meta_data[idx].astype('float32')
        data += noise * np.random.randn(num_sample, 2)
        data = transform(data)
        yield data


def boundary_generator(num_sample, noise, transform, num_meta=2):
    meta_data = []
    for i in range(-num_meta, num_meta+1, 1):
        meta_data.append([i, -num_meta])
        meta_data.append([i, +num_meta])
    for i in range(-num_meta+1, num_meta, 1):
        meta_data.append([-num_meta, i])
        meta_data.append([+num_meta, i])
    meta_data = np.asarray(meta_data) / float(num_meta)

    while True:
        idx = np.random.random_integers(0, len(meta_data)-1, num_sample)
        data = meta_data[idx].astype('float32')
        data += noise * np.random.randn(num_sample, 2)
        data = transform(data)
        yield data


def circle_generator(num_sample, noise, transform):
    while True:
        linspace = np.random.rand(num_sample)
        x = np.cos(linspace * 2 * np.pi)
        y = np.sin(linspace * 2 * np.pi)
        data = np.asarray([x, y]).transpose().astype('float32')
        data += noise * np.random.randn(num_sample, 2)
        data = transform(data)
        yield data


def scurve_generator(num_sample, noise, transform):
    while True:
        data = sklearn.datasets.make_s_curve(
            n_samples=num_sample,
            noise=noise
        )[0]
        data = data.astype('float32')[:, [0, 2]]
        data /= 2.0
        data = transform(data)
        yield data


def swiss_generator(num_sample, noise, transform):
    while True:
        data = sklearn.datasets.make_swiss_roll(
            n_samples=num_sample,
            noise=noise
        )[0]
        data = data.astype('float32')[:, [0, 2]]
        data /= 14.13717
        data = transform(data)
        yield data


def gaussian_generator(num_sample, noise, transform):
    while True:
        data = np.random.multivariate_normal([0.0, 0.0], noise * np.eye(2), num_sample)
        data = transform(data)
        yield data


def mix_generator(num_sample, generators, weights):
    while True:
        data = np.concatenate([generators[i].__next__() for i in range(len(generators))], 0)
        data_index = np.random.choice(len(weights), num_sample, replace=True, p=weights)
        data2 = np.concatenate([data[num_sample * i:num_sample * i + np.sum(data_index == i)] for i in range(len(weights))], 0)
        np.random.shuffle(data2)
        yield data2


cfg.iDimsC = 2

# fake_gen = circle_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5))
# real_gen = circle_generator(cfg.iBatchSize, 0.0, partial(transform, size=1.5))

# fake_gen = scurve_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(-1.0, 0), rot=np.pi / 2, hflip=True))
# real_gen = scurve_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(+1.0, 0), rot=np.pi / 2))

# fake_gen = scurve_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.8, mean=(0, 0), hflip=False, rot=np.pi / 4))
# real_gen = scurve_generator(cfg.iBatchSize, 0.0, partial(transform, size=1.5, mean=(0, 0)))

fake_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(-1.0, 0.0)))
real_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(+1.0, 0.0)))

# fake_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(-1.0, 0.0)))
# real_gen = mix_generator(cfg.iBatchSize, [gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(+1.0, 0.0))),
#                           gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(-1.5, 0.0)))], [0.9, 0.1])

# fake_gen = sqaure_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(-1.0, 0.0)))
# real_gen = mix_generator(cfg.iBatchSize, [gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(+1.0, 0.0))),
#                           gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(-1.0, 0.0)))], [0.9, 0.1])

# fake_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=1.0, mean=(+0.0, 0)))
# real_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=1.0, mean=(+0.0, 0)))

cfg.sTitle = cfg.sTitle.replace(' ', '_')

if cfg.sTitle == 'Case_1.1':

    fake_gen = sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(-1.0, -0.0)))
    real_gen = sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(+1.0, +0.0)))

elif cfg.sTitle == 'Case_1':

    fake_gen = mix_generator(cfg.iBatchSize, [boundary_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.50, mean=(-1.0, -0.0))),
                                              sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.25, mean=(-1.0, -0.0)))],
                             [0.20, 0.80])
    real_gen = sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(+1.0, +0.0)))

elif cfg.sTitle == 'Case_2':

    fake_gen = mix_generator(cfg.iBatchSize, [sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(-1.0, -0.0))),
                                              sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(+1.0, +0.0)))],
                             [0.80, 0.20])

    real_gen = mix_generator(cfg.iBatchSize, [sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(-1.0, -0.0))),
                                              sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(+1.0, +0.0)))],
                             [0.20, 0.80])

elif cfg.sTitle == 'Case_3.1':

    fake_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.15, mean=(-1.0, -0.0)))
    real_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.15, mean=(+1.0, +0.0)))

elif cfg.sTitle == 'Case_3':

    fake_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.15, mean=(-0.75, 0.0)))
    real_gen = mix_generator(cfg.iBatchSize, [gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.15, mean=(+1.0, 0.0))), gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.15, mean=(-1.25, 0.0)))], [0.5, 0.5])

elif 'Case_4' in cfg.sTitle:

    n = cfg.n
    np.random.seed(123456789)

    if n == 0:
        n = 2
        r0 = [[+0.8, +0.5], [-0.8, -0.5]]
        f0 = [[-0.8, +0.5], [+0.8, -0.5]]
    else:
        r0 = (np.random.rand(n, 2) - 0.5) * 2
        f0 = (np.random.rand(n, 2) - 0.5) * 2

    def get_mix_gen(centers):
        mix_gen = []
        for i in range(n):
            mix_gen.append(gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.0, mean=(centers[i][0], centers[i][1]))))
        return mix_gen

    fake_gen = mix_generator(cfg.iBatchSize, get_mix_gen(f0), [1 / n] * n)
    real_gen = mix_generator(cfg.iBatchSize, get_mix_gen(r0), [1 / n] * n)


elif 'Case_6' in cfg.sTitle:

    n = cfg.n
    std = 0

    if n < 0:
        n= -n
        std = 1e-2

    def get_mix_gen(centers, std):
        mix_gen = []
        for i in range(len(centers)):
            mix_gen.append(gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=std, mean=(centers[i][0], centers[i][1]))))
        return mix_gen

    if n == 2:

        f0 = [[-0.5, +0.5], [+0.5, -0.5]]
        r0 = [[+0.5, +0.5], [-0.5, -0.5]]

        fake_gen = mix_generator(cfg.iBatchSize, get_mix_gen(f0, std), [1 / n] * n)
        real_gen = mix_generator(cfg.iBatchSize, get_mix_gen(r0, 0), [1 / n] * n)

    elif n == 4:

        np.random.seed(123456789)
        f0 = (np.random.rand(2, 2) - 0.5) * 2
        r0 = (np.random.rand(4, 2) - 0.5) * 2

        fake_gen = mix_generator(cfg.iBatchSize, get_mix_gen(f0, std), [1 / 2] * 2)
        real_gen = mix_generator(cfg.iBatchSize, get_mix_gen(r0, 0), [1 / 4] * 4)


def discriminator_mlp(input, num_logits, name=None):

    layers = []
    iGrowthRateD = cfg.iGrowthRateD

    with tf.name_scope('' if name is None else name):

        with tf.variable_scope('discriminator', tf.AUTO_REUSE):

            h0 = input
            layers.append(h0)

            for i in range(cfg.iLayerPerBlockD):

                with tf.variable_scope('layer' + str(i)):

                    h0 = linear(h0, iGrowthRateD)
                    layers.append(h0)

                    h0 = activate(h0, oAct=cfg.oActD, oBn=cfg.oBnD)
                    layers.append(h0)

            h0 = linear(h0, num_logits, name='final_linear')
            layers.append(h0)

        return h0, layers


def discriminator_mlp_dense(input, num_logits, name=None):

    layers = []
    iGrowthRateD = cfg.iGrowthRateD

    with tf.name_scope('' if name is None else name):

        with tf.variable_scope('discriminator', tf.AUTO_REUSE):

            h0 = input
            layers.append(h0)

            for i in range(cfg.iLayerPerBlockD):

                with tf.variable_scope('layer' + str(i)):

                    h1 = h0

                    with tf.variable_scope('composite'):

                        h1 = linear(h1, iGrowthRateD)
                        layers.append(h1)

                        h1 = activate(h1, oAct=cfg.oActD, oBn=cfg.oBnD)
                        layers.append(h1)

                    h0 = tf.concat(values=[h0, h1], axis=1)

            h0 = linear(h0, num_logits, name='final_linear')
            layers.append(h0)

        return h0, layers


############################################################################################################################################

sTestName = (cfg.sResultTag + '_' if len(cfg.sResultTag) else "") + cfg.sDataSet

sTestCaseDir = cfg.sResultDir + sTestName + '/'
sSampleDir = sTestCaseDir + 'samples/'
sCheckpointDir = sTestCaseDir + 'checkpoint/'
sTFSummaryDir = sTestCaseDir + 'tflog/'

makedirs(sCheckpointDir)
makedirs(sTFSummaryDir)
makedirs(sSampleDir)
makedirs(sTestCaseDir + 'code/')

from common.logger import Logger

logger = Logger()
logger.set_dir(sTestCaseDir)
logger.set_casename(sTestName)

logger.log(sTestCaseDir)

commandline = ''
for arg in ['CUDA_VISIBLE_DEVICES="0" python3'] + sys.argv:
    commandline += arg + ' '
logger.log(commandline)

logger.log(str_flags(cfg.__flags))

copydir(cfg.sSourceDir, sTestCaseDir + 'code')

############################################################################################################################################

tf.logging.set_verbosity(tf.logging.ERROR)

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# generator = globals()[cfg.generator]
discriminator = globals()[cfg.discriminator]

real_data = tf.placeholder(tf.float32, [None, cfg.iDimsC], name='real_data')
fake_data = tf.placeholder(tf.float32, [None, cfg.iDimsC], name='fake_data')
iter_data = tf.placeholder(tf.float32, [None, cfg.iDimsC], name='iter_data')

real_logits, dis_real_layers = discriminator(real_data, 1, 'real')
fake_logits, dis_fake_layers = discriminator(fake_data, 1, 'fake')

real_logits = tf.reshape(real_logits, [-1])
fake_logits = tf.reshape(fake_logits, [-1])

if cfg.bWGAN:

    dis_real_loss = -real_logits
    dis_fake_loss = fake_logits

elif cfg.bLSGAN:

    dis_real_loss = tf.square(real_logits - 1.0)
    dis_fake_loss = tf.square(fake_logits + 1.0)

elif cfg.bWGANs:

    dis_real_loss = -tf.log_sigmoid(real_logits)-real_logits
    dis_fake_loss = -tf.log_sigmoid(-fake_logits)+fake_logits

else:

    dis_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits))
    dis_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits))

varphi_gradients = tf.gradients(dis_real_loss, real_logits)[0]
phi_gradients = tf.gradients(dis_fake_loss, fake_logits)[0]

dis_gan_loss = tf.reduce_mean(dis_fake_loss) + tf.reduce_mean(dis_real_loss)
dis_zero_loss = cfg.fWeightZero * tf.square(tf.reduce_mean(fake_logits) + tf.reduce_mean(real_logits))

dis_total_loss = dis_gan_loss + dis_zero_loss

layers = dis_real_layers + dis_fake_layers

dotk = tf.constant(0.0)
slopes = tf.constant(0.0)
gradients = tf.constant(0.0)
gp_loss = tf.constant(0.0)
interpolates = tf.constant(0.0)

if cfg.bLip:

    alpha = tf.random_uniform(shape=[tf.shape(fake_data)[0], 1], minval=0., maxval=1.)

    differences = fake_data - real_data
    interpolates = real_data + alpha * differences

    if cfg.bMaxGP:
        interpolates = tf.concat([interpolates[:-tf.shape(iter_data)[0]], iter_data], 0)

    interpolates_logits, interpolates_layers = discriminator(interpolates, 1, 'inter')
    gradients = tf.gradients(interpolates_logits, interpolates)[0]

    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))  #tf.norm()

    if cfg.bUseSN:
        dotk = tf.reduce_prod(SPECTRAL_NORM_K_LIST)
        # gp_loss = cfg.fWeightGP * (dotk ** 2)
        gp_loss = cfg.fWeightGP * (tf.maximum(0.0, dotk - 1.0) ** 2)
        # gp_loss = cfg.fWeightGP * tf.reduce_sum(tf.maximum(0.0, SPECTRAL_NORM_K_LIST - tf.constant(1.0)) ** 2)
    else:
        if cfg.bMaxGP:
            gp_loss = cfg.fWeightGP * tf.reduce_max(tf.square(slopes))
        else:
            # gp_loss = cfg.fWeightGP * tf.reduce_mean(tf.square(tf.maximum(0.0, slopes - 1.0)))
            # gp_loss = cfg.fWeightGP * tf.reduce_mean(tf.square(slopes-1))
            gp_loss = cfg.fWeightGP * tf.reduce_mean(tf.square(slopes))
            # gp_loss = cfg.fWeightGP * tf.reduce_mean(tf.exp(slopes))

    dis_total_loss += gp_loss
    layers += interpolates_layers

    # slope_gradients = tf.gradients(gp_loss, interpolates_logits)[0]

real_gradients = tf.gradients(real_logits, real_data)[0]

layer_gradients = tf.gradients(dis_total_loss, layers)
layer_gradients = [tf.constant(0.0) if layer_gradient is None else layer_gradient for layer_gradient in layer_gradients]

tot_vars = tf.trainable_variables()
dis_vars = [var for var in tot_vars if 'discriminator' in var.name]

dis_l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in dis_vars])
dis_total_loss += dis_l2_loss * cfg.fL2Weight


d_global_step = tf.Variable(0, trainable=False, name='d_global_step')
dis_lr = tf.train.exponential_decay(cfg.fLrIniD, d_global_step, cfg.iLrStep, cfg.fLrDecay, cfg.bLrStair)
dis_optimizer = None
if cfg.oOptD == 'sgd':
    dis_optimizer = tf.train.GradientDescentOptimizer(learning_rate=dis_lr)
elif cfg.oOptD == 'mom':
    dis_optimizer = tf.train.MomentumOptimizer(learning_rate=dis_lr, momentum=0.9, use_nesterov=True)
elif cfg.oOptD == 'adam':
    dis_optimizer = tf.train.AdamOptimizer(learning_rate=dis_lr, beta1=cfg.fBeta1D, beta2=cfg.fBeta2D, epsilon=cfg.fEpsilonD)
dis_gradient_values = dis_optimizer.compute_gradients(dis_total_loss, var_list=dis_vars)
dis_optimize_ops = dis_optimizer.apply_gradients(dis_gradient_values, global_step=d_global_step)

dis_gradient_values = [(tf.constant(0.0),dis_gradient_value[1]) if dis_gradient_value[0] is None else dis_gradient_value for dis_gradient_value in dis_gradient_values]

saver = tf.train.Saver(max_to_keep=1000)
writer = tf.summary.FileWriter(sTFSummaryDir, sess.graph)


def plot(names, x_map_size, y_map_size, x_value_range, y_value_range, mode='fake'):

    def get_current_logits_map():

        logits_map = np.zeros([y_map_size, x_map_size])
        gradients_map = np.zeros([y_map_size, x_map_size, 2])

        for i in range(y_map_size):  # the i-th row and j-th column
            locations = []
            for j in range(x_map_size):
                y = y_value_range[1] - (y_value_range[1] - y_value_range[0]) / y_map_size * (i + 0.5)
                x = x_value_range[0] + (x_value_range[1] - x_value_range[0]) / x_map_size * (j + 0.5)
                locations.append([x, y])
            locations = np.asarray(locations).reshape([x_map_size, 2])
            logits_map[i], gradients_map[i] = sess.run([real_logits, real_gradients], feed_dict={real_data: locations})

        return logits_map, gradients_map

    def boundary_data(num_meta, mean, size):

        meta_data = []
        for i in range(-num_meta, num_meta + 1, 1):
            meta_data.append([i, -num_meta])
            meta_data.append([i, +num_meta])
        for i in range(-num_meta + 1, num_meta, 1):
            meta_data.append([-num_meta, i])
            meta_data.append([+num_meta, i])
        meta_data = np.asarray(meta_data) / float(num_meta)

        meta_data *= size
        meta_data += mean

        return meta_data

    def get_data_and_gradient(gen, num, pre_sample=None):
        data = []
        logit = []
        gradient = []
        if pre_sample is not None:
            _logit, _gradient = sess.run([real_logits, real_gradients], feed_dict={real_data: pre_sample})
            data.append(pre_sample)
            logit.append(_logit)
            gradient.append(_gradient)
        for i in range(num // cfg.iBatchSize + 1):
            _data = gen.__next__()
            _logit, _gradient = sess.run([real_logits, real_gradients], feed_dict={real_data: _data})
            data.append(_data)
            logit.append(_logit)
            gradient.append(_gradient)
        data = np.concatenate(data, axis=0)
        logit = np.concatenate(logit, axis=0)
        gradient = np.concatenate(gradient, axis=0)
        return data[:num], logit[:num], gradient[:num]

    pre_sample = None
    if cfg.sTitle == 'Case_1':
        pre_sample = boundary_data(5, [-1.0, 0.0], 0.25)
    elif cfg.sTitle == 'Case_2':
        pre_sample = np.concatenate([boundary_data(5, [-1.0, 0.0], 0.5), boundary_data(5, [1.0, 0.0], 0.5)], 0)

    _real_data, _real_logit, _real_gradients = get_data_and_gradient(real_gen, 1024)
    _fake_data, _fake_logit, _fake_gradients = get_data_and_gradient(fake_gen, 1024, pre_sample)

    if cfg.bLSGAN:
        cmin = -2.0
        cmax = +2.0
    else:
        cmin = np.min(np.concatenate([_fake_logit, _real_logit], 0))
        cmax = np.max(np.concatenate([_fake_logit, _real_logit], 0))

    rcParams['font.family'] = 'monospace'
    fig, ax = plt.subplots(dpi=1000)

    logits_map, gradients_map = get_current_logits_map()
    im = ax.imshow(logits_map, extent=[x_value_range[0], x_value_range[1], y_value_range[0], y_value_range[1]], vmin=cmin, vmax=cmax, cmap='viridis')
    pickle.dump([_real_data, _real_logit, _real_gradients, _fake_data, _fake_logit, _fake_gradients, logits_map, gradients_map], open(sTestCaseDir + names[0] + '.pck', 'wb'))

    plt.scatter(_real_data[:, 0], _real_data[:, 1], marker='+', s=1.5, label='real samples', color='navy')#purple')#indigo')#navy')balck
    plt.scatter(_fake_data[:, 0], _fake_data[:, 1], marker='*', s=1.5, label='fake samples', color='ivory')#ivory') #'silver')white

    plt.xlim(x_value_range[0], x_value_range[1])
    plt.ylim(y_value_range[0], y_value_range[1])

    if mode == 'fake':
        xx, yy, uu, vv = _fake_data[:, 0], _fake_data[:, 1], _fake_gradients[:, 0], _fake_gradients[:, 1]
    else:
        num_arrow = 20
        skip = (slice(y_map_size // num_arrow // 2, None, y_map_size // num_arrow), slice(x_map_size // num_arrow // 2, None, x_map_size // num_arrow))
        y, x = np.mgrid[y_value_range[1]:y_value_range[0]:y_map_size * 1j, x_value_range[0]:x_value_range[1]:x_map_size * 1j]
        xx, yy, uu, vv = x[skip], y[skip], gradients_map[skip][:, :, 0], gradients_map[skip][:, :, 1]

    ref_scale = np.max(np.linalg.norm(_fake_gradients, axis=1)) / 2 if np.max(np.linalg.norm(_fake_gradients, axis=1)) / np.mean(np.linalg.norm(_fake_gradients, axis=1)) < 2 else np.mean(np.linalg.norm(_fake_gradients, axis=1))

    if cfg.sTitle == 'Case_5':
        ref_scale = np.mean(np.linalg.norm(_fake_gradients, axis=1)) / 5

    len = np.hypot(uu, vv)
    uu = uu / (len+1e-8) * np.minimum(len, 2 * ref_scale)
    vv = vv / (len+1e-8) * np.minimum(len, 2 * ref_scale)

    q = ax.quiver(xx, yy, uu, vv, color='red', angles='xy', minlength=0.8, minshaft=3, scale=ref_scale * 30) # violet, fuchsia
    plt.quiverkey(q, 0.67, 0.917, ref_scale, r'$\vert\frac{\partial{f(x)}}{\partial{x}}\vert$=%.2E' % (float(ref_scale)), labelpos='E', coordinates='figure')

    title = cfg.sTitle.replace('_', ' ')
    if 'Non-Differentiable' in cfg.sTitle:
        title = 'Non-Differentiable'

    ax.set(aspect=1, title=title)
    plt.legend(loc='upper left', prop={'size': 10})

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, extend='both', format='%.1f')

    plt.tight_layout()

    for name in names:
        plt.savefig(sTestCaseDir + name)

    plt.close()


iter = 0
if cfg.bLoadCheckpoint:
    try:
        if load_model(saver, sess, sCheckpointDir):
            logger.log(" [*] Load SUCCESS")
            iter = sess.run(d_global_step)
            logger.load()
            logger.flush(plot=True)
        else:
            assert False
    except:
        logger.log(" [*] Load FAILED")
        ini_model(sess)
else:
    ini_model(sess)


def param_count(gradient_value):
    total_param_count = 0
    for g, v in gradient_value:
        shape = v.get_shape()
        param_count = 1
        for dim in shape:
            param_count *= int(dim)
        total_param_count += param_count
    return total_param_count


logger.log("Discriminator Total Parameter Count: {}\n\n".format(locale.format("%d", param_count(dis_gradient_values), grouping=True)))

_real_data = real_gen.__next__()
_fake_data = fake_gen.__next__()
_iter_data = ((_real_data + _fake_data) / 2)[:cfg.iBatchSize // 8]


def log_netstate():

    global _iter_data

    _real_data = real_gen.__next__()
    _fake_data = fake_gen.__next__()

    logger.log('\n\n')
    gradient_value = dis_gradient_values
    _gradient_value = sess.run(gradient_value, feed_dict={real_data: _real_data, fake_data: _fake_data, iter_data: _iter_data})
    for i in range(len(_gradient_value)):
        logger.log('weight values: %8.5f %8.5f, its gradients: %13.10f, %13.10f   ' % (np.mean(_gradient_value[i][1]), np.std(_gradient_value[i][1]), np.mean(_gradient_value[i][0]), np.std(_gradient_value[i][0])) + gradient_value[i][1].name + ' shape: ' + str(_gradient_value[i][0].shape))
    logger.log('\n\n')
    _layers, _layer_gradients = sess.run([layers, layer_gradients], feed_dict={real_data: _real_data, fake_data: _fake_data, iter_data: _iter_data})
    for i in range(len(_layers)):
        logger.log('layer values: %8.5f %8.5f, its gradients: %13.10f, %13.10f   ' % (np.mean(_layers[i]), np.std(_layers[i]), np.mean(_layer_gradients[i]), np.std(_layer_gradients[i])) + layers[i].name + ' shape: ' + str(_layers[i].shape))
    logger.log('\n\n')


def run(iter, max_iter):

    global _iter_data

    while iter < max_iter:

        iter += 1
        start_time = time.time()

        _real_data = real_gen.__next__()
        _fake_data = fake_gen.__next__()

        sess.run(SPECTRAL_NORM_UV_UPDATE_OPS_LIST)

        _, _dis_total_loss, _dis_l2_loss, _dis_gan_loss, _gp_loss, _interpolates, _dphi, _dvarphi, _slopes, _dotk, _dis_zero_loss, _dis_lr, _real_logits, _fake_logits = sess.run(
            [dis_optimize_ops, dis_total_loss, dis_l2_loss, dis_gan_loss, gp_loss, interpolates, phi_gradients, varphi_gradients, slopes, dotk, dis_zero_loss, dis_lr, real_logits, fake_logits], feed_dict={real_data: _real_data, fake_data: _fake_data, iter_data: _iter_data})

        logger.tick(iter)
        logger.info('klr', _dis_lr * 1000)
        logger.info('time', time.time() - start_time)

        logger.info('logit_real', np.mean(_real_logits))
        logger.info('logit_fake', np.mean(_fake_logits))

        logger.info('loss_gp', _gp_loss)
        logger.info('loss_gan', _dis_gan_loss)
        logger.info('loss_tot', _dis_total_loss)
        logger.info('loss_zero', _dis_zero_loss)

        logger.info('d_phi', np.mean(_dphi))
        logger.info('d_varphi', np.mean(_dvarphi))

        logger.info('dotk', _dotk)
        logger.info('slopes', np.max(_slopes))

        if cfg.bLip:
            logger.info('slopes_var', np.max((sess.run(slopes, feed_dict = {real_data: _real_data, fake_data: _fake_data, iter_data: _iter_data})-_slopes)[-cfg.iBatchSize//2:]))

        if cfg.bLip and cfg.bMaxGP:
            _iter_data = _interpolates[np.argsort(-np.asarray(_slopes))[:len(_iter_data)]]

        if np.any(np.isnan(_real_logits)) or np.any(np.isnan(_fake_logits)):
            log_netstate()
            logger.flush(plot=True)
            exit(0)

        if iter % 10000 == 0:
            log_netstate()

        if iter % 100 == 0:
            logger.flush(plot=iter % 10000 == 0)

        if iter % 10000 == 0:
            save_model(saver, sess, sCheckpointDir, step=iter)
            logger.save()

    return iter


inteval = 1000
iter_plot_list = list(np.arange(inteval, cfg.iMaxIter+1, inteval))

if cfg.bUseSN:
    sess.run(SPECTRAL_NORM_UV_UPDATE_OPS_LIST)

if cfg.bUseSNK:
    sess.run(SPECTRAL_NORM_K_INIT_OPS_LIST)

log_netstate()

for iter_plot in iter_plot_list:

    if iter > iter_plot:
        continue

    iter = run(iter, iter_plot)

    plot(['map_fake_%d.png' % iter, 'map_fake_%d.pdf' % iter], 900, 600, [-2.0, 2.0], [-1.5, 1.5], 'fake')
    # plot(['map_%d.pdf' % iter,'map_%d.png' % iter], 600, 400, [-2.0, 2.0], [-1.5, 1.5], 'map')