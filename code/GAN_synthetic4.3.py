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
from scipy import ndimage
from common.data_loader import *
import sklearn.datasets
from functools import partial
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams, colorbar, colors

cfg = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("sDataSet", "toy", "cifar10, mnist, toy")
tf.app.flags.DEFINE_string("sResultTag", "test4.3", "your tag for each test case")

tf.app.flags.DEFINE_string("sResultDir", "../result/", "where to save the checkpoint and sample")
tf.app.flags.DEFINE_string("sSourceDir", "../code/", "")

tf.app.flags.DEFINE_integer("n", 2, "")
tf.app.flags.DEFINE_string("sTitle", "", "")

tf.app.flags.DEFINE_string("generator", 'generator_dcgan', "")
tf.app.flags.DEFINE_string("discriminator", 'discriminator_mlp_dense', "")

tf.app.flags.DEFINE_boolean("bWGAN", False, "")
tf.app.flags.DEFINE_boolean("bWGANs", True, "")
tf.app.flags.DEFINE_boolean("bLSGAN", False, "")

tf.app.flags.DEFINE_boolean("bLip", True, "")
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

tf.app.flags.DEFINE_integer("iDimsZ", 100, "")
tf.app.flags.DEFINE_integer("iMinSizeG", 4, "")
tf.app.flags.DEFINE_integer("iGrowthRateG", 256, "")
tf.app.flags.DEFINE_integer("iLayerPerBlockG", 8, "")

tf.app.flags.DEFINE_float("fDimIncreaseRate", 2.0, "")
tf.app.flags.DEFINE_boolean("bc", False, "")

tf.app.flags.DEFINE_float("fLayerNoiseD",   0.00, "")
tf.app.flags.DEFINE_float("fLayerDropoutD", 0.00, "")

tf.app.flags.DEFINE_float("fIniScale", 1.0, "")
tf.app.flags.DEFINE_string("oActD", 'relu', "relu, lrelu, elu")
tf.app.flags.DEFINE_string("oBnD", 'none', "bn, ln, none")
tf.app.flags.DEFINE_string("oDownsize", 'avgpool', "avgpool, maxpool")

tf.app.flags.DEFINE_string("oActG", 'relu', "relu, lrelu, elu")
tf.app.flags.DEFINE_string("oBnG", 'none', "bn, ln, none")

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
h_axis, w_axis, c_axis = [2, 3, 1]


def discriminator_dcgan(input, num_logits, name):

    layers = []
    iGrowthRateD = cfg.iGrowthRateD

    with tf.name_scope('' if name is None else name):

        with tf.variable_scope('discriminator', tf.AUTO_REUSE):

            h0 = input
            layers.append(h0)

            while True:

                with tf.variable_scope('size' + str(h0.get_shape().as_list()[w_axis])):

                    h0 = conv2d(h0, iGrowthRateD, ksize=4, stride=2)
                    layers.append(h0)

                    h0 = activate(h0, oAct=cfg.oActD, oBn=cfg.oBnD)
                    layers.append(h0)

                    if h0.get_shape().as_list()[w_axis] / 2 >= cfg.iMinSizeD:
                        iGrowthRateD = int(iGrowthRateD * cfg.fDimIncreaseRate)
                    else:
                        break

            h0 = tf.contrib.layers.flatten(h0)
            layers.append(h0)

            h0 = lnoise(h0, fNoise=cfg.fLayerNoiseD, fDrop=cfg.fLayerDropoutD)
            layers.append(h0)

            h0 = linear(h0, num_logits)
            layers.append(h0)

    return h0, layers


def discriminator_dcgan_mod(input, num_logits, name):

    layers = []
    iGrowthRateD = cfg.iGrowthRateD

    with tf.name_scope('' if name is None else name):

        with tf.variable_scope('discriminator', tf.AUTO_REUSE):

            h0 = input
            layers.append(h0)

            while True:

                with tf.variable_scope('size' + str(h0.get_shape().as_list()[w_axis])):

                    h0 = conv2d(h0, iGrowthRateD, ksize=3, stride=1)
                    layers.append(h0)

                    h0 = activate(h0, oAct=cfg.oActD, oBn=cfg.oBnD)
                    layers.append(h0)

                    if h0.get_shape().as_list()[w_axis] / 2 >= cfg.iMinSizeD:

                        with tf.variable_scope('downsize'):

                            h0 = downsize(h0, cfg.oDownsize, 2)
                            layers.append(h0)

                        iGrowthRateD = int(iGrowthRateD * cfg.fDimIncreaseRate)

                    else:
                        break

            h0 = downsize(h0, cfg.oDownsize, h0.get_shape().as_list()[w_axis])
            h0 = tf.contrib.layers.flatten(h0)
            layers.append(h0)

            h0 = lnoise(h0, fNoise=cfg.fLayerNoiseD, fDrop=cfg.fLayerDropoutD)
            layers.append(h0)

            h0 = linear(h0, num_logits)
            layers.append(h0)

    return h0, layers


def discriminator_dense(input, num_logits, name):

    layers = []
    iGrowthRateD = cfg.iGrowthRateD

    with tf.name_scope('' if name is None else name):

        with tf.variable_scope('discriminator', tf.AUTO_REUSE):

            h0 = input
            layers.append(h0)

            while True:

                with tf.variable_scope('size' + str(h0.get_shape().as_list()[w_axis])):

                    for i in range(cfg.iLayerPerBlockD):

                        with tf.variable_scope('layer' + str(i)):

                            h1 = h0

                            if cfg.bc and h0.get_shape().as_list()[c_axis] > iGrowthRateD * 8:
                                with tf.variable_scope('bottleneck'):

                                    h1 = conv2d(h1, iGrowthRateD * 4, ksize=1, stride=1)
                                    layers.append(h1)

                                    h1 = activate(h1, oAct=cfg.oActD, oBn=cfg.oBnD)
                                    layers.append(h1)

                            with tf.variable_scope('composite'):

                                h1 = conv2d(h1, iGrowthRateD, ksize=3, stride=1)
                                layers.append(h1)

                                h1 = activate(h1, oAct=cfg.oActD, oBn=cfg.oBnD)
                                layers.append(h1)

                            h0 = tf.concat(values=[h0, h1], axis=c_axis)

                    if h0.get_shape().as_list()[w_axis] / 2 >= cfg.iMinSizeD:

                        with tf.variable_scope('downsize'):

                            h0 = downsize(h0, cfg.oDownsize, 2)
                            layers.append(h0)

                        iGrowthRateD = int(iGrowthRateD * cfg.fDimIncreaseRate)

                    else:
                        break

            h0 = downsize(h0, cfg.oDownsize, h0.get_shape().as_list()[w_axis])
            h0 = tf.contrib.layers.flatten(h0)
            layers.append(h0)

            h0 = lnoise(h0, fNoise=cfg.fLayerNoiseD, fDrop=cfg.fLayerDropoutD)
            layers.append(h0)

            h0 = linear(h0, num_logits)
            layers.append(h0)

    return h0, layers


def discriminator_mlp_dense(input, num_logits, name):

    layers = []
    iGrowthRateD = cfg.iGrowthRateD

    with tf.name_scope('' if name is None else name):

        with tf.variable_scope('discriminator', tf.AUTO_REUSE):

            h0 = input
            layers.append(h0)

            h0 = tf.contrib.layers.flatten(h0)
            layers.append(h0)

            for i in range(cfg.iLayerPerBlockD):

                with tf.variable_scope('layer' + str(i)):

                    h1 = h0

                    with tf.variable_scope('composite'):

                        h1 = linear(h1, iGrowthRateD)
                        layers.append(h1)

                        h1 = activate(h1, oAct=cfg.oActD, oBn=cfg.oBnD)
                        layers.append(h1)

                    h0 = tf.concat(values=[h0, h1], axis=c_axis)

            h0 = tf.contrib.layers.flatten(h0)
            layers.append(h0)

            h0 = lnoise(h0, fNoise=cfg.fLayerNoiseD, fDrop=cfg.fLayerDropoutD)
            layers.append(h0)

            h0 = linear(h0, num_logits)
            layers.append(h0)

    return h0, layers


def generator_dcgan(z=None, name=None):

    layers = []

    with tf.name_scope('' if name is None else name):

        with tf.variable_scope('generator', tf.AUTO_REUSE):

            h0 = z
            layers.append(h0)

            size = 32
            iGrowthRateG = cfg.iGrowthRateD
            while size > cfg.iMinSizeG:
                iGrowthRateG = int(iGrowthRateG * cfg.fDimIncreaseRate)
                size = size // 2

            with tf.variable_scope('latent'):
                h0 = linear(h0, iGrowthRateG * cfg.iMinSizeG * cfg.iMinSizeG)
                layers.append(h0)
                h0 = tf.reshape(h0, [-1, iGrowthRateG, cfg.iMinSizeG, cfg.iMinSizeG])
                layers.append(h0)

            while h0.get_shape().as_list()[w_axis] < 32:

                with tf.variable_scope('size' + str(h0.get_shape().as_list()[w_axis])):

                    iGrowthRateG = int(iGrowthRateG / cfg.fDimIncreaseRate)
                    h0 = deconv2d(h0, iGrowthRateG, ksize=3, stride=2)
                    layers.append(h0)
                    h0 = activate(h0, oAct=cfg.oActG, oBn=cfg.oBnG)
                    layers.append(h0)

            h0 = conv2d(h0, cfg.iDimsC, ksize=3, stride=1)
            layers.append(h0)

            h0 = tf.nn.tanh(h0)
            layers.append(h0)

    return h0, layers


def generator_dense(z=None, name=None):

    layers = []

    with tf.name_scope('' if name is None else name):

        with tf.variable_scope('generator', tf.AUTO_REUSE):

            h0 = z
            layers.append(h0)

            size = 32
            iGrowthRateG = cfg.iGrowthRateG
            while size > cfg.iMinSizeG:
                iGrowthRateG = int(iGrowthRateG * cfg.fDimIncreaseRate)
                size = size // 2

            with tf.variable_scope('latent'):
                h0 = linear(h0, iGrowthRateG * cfg.iMinSizeG * cfg.iMinSizeG)
                layers.append(h0)
                h0 = tf.reshape(h0, [-1, iGrowthRateG, cfg.iMinSizeG, cfg.iMinSizeG])
                layers.append(h0)

            while True:

                with tf.variable_scope('size' + str(h0.get_shape().as_list()[w_axis])):

                    for i in range(cfg.iLayerPerBlockG):

                        with tf.variable_scope('layer' + str(i)):

                            h1 = h0

                            if cfg.bc and h0.get_shape().as_list()[c_axis] > iGrowthRateG * 8:
                                with tf.variable_scope('bottleneck'):

                                    h1 = conv2d(h1, iGrowthRateG * 4, ksize=1, stride=1)
                                    layers.append(h1)

                                    h1 = activate(h1, oAct=cfg.oActD, oBn=cfg.oBnD)
                                    layers.append(h1)

                            with tf.variable_scope('composite'):

                                h1 = conv2d(h1, iGrowthRateG, ksize=3, stride=1)
                                layers.append(h1)

                                h1 = activate(h1, oAct=cfg.oActD, oBn=cfg.oBnD)
                                layers.append(h1)

                            h0 = tf.concat(values=[h0, h1], axis=c_axis)

                    if h0.get_shape().as_list()[w_axis] < 32:

                        with tf.variable_scope('downsize'):

                            h0 = upsize(h0, cfg.oUpsize, 2)
                            layers.append(h0)

                        iGrowthRateG = int(iGrowthRateG / cfg.fDimIncreaseRate)

                    else:
                        break

            h0 = conv2d(h0, cfg.iDimsC, ksize=3, stride=1)
            layers.append(h0)

            h0 = tf.nn.tanh(h0)
            layers.append(h0)

    return h0, layers


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

            # h0 = lnoise(h0, fNoise=cfg.fLayerNoiseD, fDrop=cfg.fLayerDropoutD)
            # layers.append(h0)

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

generator = globals()[cfg.generator]
discriminator = globals()[cfg.discriminator]


def load_dataset(dataset_name):
    if dataset_name == 'cifar10':
        return load_cifar10()
    if dataset_name == 'mnist':
        return load_mnist()


n = cfg.n
cifarX, cifarY = load_cifar10()[:2]
cifarK = [1400, 1621, 808, 809, 3687, 6815, 2100, 4037, 9154, 8532]

mnistX, mnistY = load_mnist(useC3=True)[:2]
mnistK = [1916, 1936, 1923, 1994, 1992, 1996, 1642, 1946, 1790, 1949]

# #1973 1989 1970 1930
# makedirs(sTestCaseDir+'real_sample')
# for i in range(10000):
#     data = dataX[i]
#     if data.shape[0] == 1:
#         data = np.concatenate([data, data, data], 0)
#     scipy.misc.toimage(data.transpose(1, 2, 0), cmin=-1, cmax=1).save(sTestCaseDir + 'real_sample/%d_%d.png' % (dataY[i], i))

cfg.sTitle = cfg.sTitle.replace(' ', '_')

if 'Case_5' in cfg.sTitle:

    dataX, dataY, testX, testY = load_dataset(cfg.sDataSet)
    cfg.iDimsC = np.shape(dataX)[1]
    k = mnistK if cfg.sDataSet == 'mnist' else cifarK

    r0 = dataX[k[:n]]


    def get_gen_sample(n):
        z0 = np.random.randn(n, cfg.iDimsZ)
        data = sess.run(gen, feed_dict={z: z0})
        return data


    def gen_with_generator(n):
        while True:
            data = get_gen_sample(n)
            yield data


    z = tf.placeholder(tf.float32, [cfg.iBatchSize, cfg.iDimsZ], name='z')
    gen, _ = generator(z, name='fake_data')
    sess.run(tf.global_variables_initializer())

    # f0 = get_gen_sample(n)
    f0 = np.random.uniform(size=np.shape(r0), low=-1., high=1.)
    # f0 = ndimage.gaussian_filter(r0, sigma=(0.0, 5.0, 5.0, 5.0))
    # f0 = 0.9 * np.random.uniform(size=np.shape(r0), low=-1., high=1.) + 0.1 * f0

elif 'Case_7' in cfg.sTitle:

    cfg.iDimsC = 3
    r0 = cifarX[cifarK[:n]]
    f0 = mnistX[mnistK[:n]]


def data_gen(data, num_sample):
    while True:
        num_data = len(data)
        data_index = np.random.choice(num_data, num_sample, replace=True, p=num_data * [1/num_data])
        yield data[data_index]

np.random.shuffle(r0)
np.random.shuffle(f0)

def fake_uniform_gen():
    while True:
        data = np.random.uniform(size=(cfg.iBatchSize,)+np.shape(r0)[1:], low=-1., high=1.)
        yield data


real_gen = data_gen(r0, cfg.iBatchSize)
fake_gen = data_gen(f0, cfg.iBatchSize)

if 'Case_7' in cfg.sTitle:
    fake_gen = fake_uniform_gen()

real_data = tf.placeholder(tf.float32, (None,) + np.shape(r0)[1:], name='real_data')
fake_data = tf.placeholder(tf.float32, (None,) + np.shape(r0)[1:], name='fake_data')
iter_data = tf.placeholder(tf.float32, (None,) + np.shape(r0)[1:], name='iter_data')

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

if cfg.bLip:

    alpha = tf.random_uniform(shape=[tf.shape(fake_data)[0], 1, 1, 1], minval=0., maxval=1.)

    differences = fake_data - real_data
    interpolates = real_data + alpha * differences

    if cfg.bMaxGP:
        interpolates = tf.concat([interpolates[:-tf.shape(iter_data)[0]], iter_data], 0)

    interpolates_logits, interpolates_layers = discriminator(interpolates, 1, 'inter')
    gradients = tf.gradients(interpolates_logits, interpolates)[0]

    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))  # tf.norm()

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
    # layers += interpolates_layers

real_gradients = tf.gradients(real_logits, real_data)[0]
fake_gradients = tf.gradients(fake_logits, fake_data)[0]

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

dis_gradient_values = [(tf.constant(0.0), dis_gradient_value[1]) if dis_gradient_value[0] is None else dis_gradient_value for dis_gradient_value in dis_gradient_values]

saver = tf.train.Saver(max_to_keep=1000)
writer = tf.summary.FileWriter(sTFSummaryDir, sess.graph)

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
            logger.info('slopes_var', np.max((sess.run(slopes, feed_dict={real_data: _real_data, fake_data: _fake_data, iter_data: _iter_data}) - _slopes)[-cfg.iBatchSize // 2:]))

        if cfg.bMaxGP:
            _iter_data = _interpolates[np.argsort(-np.asarray(_slopes))[:len(_iter_data)]]

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


def path(r, f, g, n):

    images = []

    rr = []
    for i in range(len(f)):
        error = np.zeros(len(r))
        for j in range(len(r)):
            # s = np.mean(r[j] - f[i]) / np.mean(g[i])
            # error[j] = np.linalg.norm(r[j] - f[i] - s * g[i])
            g_dir = np.reshape(g[i] / np.linalg.norm(g[i]), [-1])
            rf_dir = np.reshape((r[j]-f[i]) / np.linalg.norm(r[j]-f[i]), [-1])
            error[j] = -g_dir.dot(rf_dir)
        ir = np.argmin(error)
        rr.append(r[ir])
    rr = np.asarray(rr)

    s = np.median((rr-f) / g, axis=(1, 2, 3), keepdims=True)
    # s = np.mean(rr - f, axis=(1, 2, 3), keepdims=True) / np.mean(g, axis=(1, 2, 3), keepdims=True)

    images.append(f)
    images.append(g / np.max(np.abs(g), axis=(1, 2, 3), keepdims=True))

    for i in range(n):
        nn = int(n // 3)
        ff = f + (i+1)/(n-nn) * g * s
        images.append(ff)
        # ff = ff / np.max(np.abs(ff), axis=(1, 2, 3), keepdims=True)
        # images.append(ff)

    images.append(rr)

    return np.stack(images, 1)


for iter_plot in iter_plot_list:

    if iter > iter_plot:
        continue

    iter = run(iter, iter_plot)

    if 'Case_7' not in cfg.sTitle:

        m = 9
        grad_path = path(r0, f0, sess.run(real_gradients, feed_dict={real_data: f0}), m)
        save_images(grad_path.reshape(-1, cfg.iDimsC, 32, 32).transpose([0, 2, 3, 1]), [np.shape(grad_path)[0], np.shape(grad_path)[1]], sTestCaseDir + 'grad_path9_%d.png' % iter)
        save_images(grad_path.reshape(-1, cfg.iDimsC, 32, 32).transpose([0, 2, 3, 1]), [np.shape(grad_path)[0], np.shape(grad_path)[1]], sTestCaseDir + 'grad_path9_%d.pdf' % iter)

        m = 14
        grad_path = path(r0, f0, sess.run(real_gradients, feed_dict={real_data: f0}), m)
        save_images(grad_path.reshape(-1, cfg.iDimsC, 32, 32).transpose([0, 2, 3, 1]), [np.shape(grad_path)[0], np.shape(grad_path)[1]], sTestCaseDir + 'grad_path14_%d.png' % iter)
        save_images(grad_path.reshape(-1, cfg.iDimsC, 32, 32).transpose([0, 2, 3, 1]), [np.shape(grad_path)[0], np.shape(grad_path)[1]], sTestCaseDir + 'grad_path14_%d.pdf' % iter)

    else:

        tmp = fake_gen.__next__()

        f0 = tmp[:10]

        m = 14
        grad_path = path(r0, f0, sess.run(real_gradients, feed_dict={real_data: f0}), m)
        save_images(grad_path.reshape(-1, cfg.iDimsC, 32, 32).transpose([0, 2, 3, 1]), [np.shape(grad_path)[0], np.shape(grad_path)[1]], sTestCaseDir + 'grad_path25x14_%d.png' % iter)
        save_images(grad_path.reshape(-1, cfg.iDimsC, 32, 32).transpose([0, 2, 3, 1]), [np.shape(grad_path)[0], np.shape(grad_path)[1]], sTestCaseDir + 'grad_path25x14_%d.pdf' % iter)

        f0 = tmp[:25]

        m = 14
        grad_path = path(r0, f0, sess.run(real_gradients, feed_dict={real_data: f0}), m)
        save_images(grad_path.reshape(-1, cfg.iDimsC, 32, 32).transpose([0, 2, 3, 1]), [np.shape(grad_path)[0], np.shape(grad_path)[1]], sTestCaseDir + 'grad_path10x14_%d.png' % iter)
        save_images(grad_path.reshape(-1, cfg.iDimsC, 32, 32).transpose([0, 2, 3, 1]), [np.shape(grad_path)[0], np.shape(grad_path)[1]], sTestCaseDir + 'grad_path10x14_%d.pdf' % iter)

        f0 = tmp[:128]

        g0 = sess.run(real_gradients, feed_dict={real_data: f0})
        g0 = g0 / np.max(np.abs(g0), axis=(1, 2, 3), keepdims=True)
        grad_image = np.stack([f0, g0], 1)
        save_images(grad_image.reshape(-1, cfg.iDimsC, 32, 32).transpose([0, 2, 3, 1]), [16, 16], sTestCaseDir + 'grad_image16x16_%d.png' % iter)
        save_images(grad_image.reshape(-1, cfg.iDimsC, 32, 32).transpose([0, 2, 3, 1]), [16, 16], sTestCaseDir + 'grad_image16x16_%d.pdf' % iter)