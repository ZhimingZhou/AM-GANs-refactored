import sys, locale, os
from os import path

locale.setlocale(locale.LC_ALL, '')
sys.path.append(path.dirname(path.abspath(__file__)))
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
from common.data_loader import *

cfg = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("sDataSet", "mnist", "cifar10, mnist, toy")
tf.app.flags.DEFINE_string("sResultTag", "test_v0", "your tag for each test case")

tf.app.flags.DEFINE_string("sResultDir", "/newNAS/Workspaces/CVGroup/liheng/result/",
                           "where to save the checkpoint and sample")
tf.app.flags.DEFINE_string("sSourceDir", "/newNAS/Workspaces/CVGroup/liheng/code/", "")

tf.app.flags.DEFINE_boolean("bWGAN", True, "")
tf.app.flags.DEFINE_boolean("bLSGAN", False, "")
tf.app.flags.DEFINE_float("fWeightGP", 1.0, "")

################################################# Learning Process ###########################################################################################

tf.app.flags.DEFINE_boolean("bLoadCheckpoint", False, "bLoadCheckpoint")

tf.app.flags.DEFINE_integer("iMaxIter", 1000000, "")
tf.app.flags.DEFINE_integer("iBatchSize", 64, "")

tf.app.flags.DEFINE_integer("iTrainG", 1, "")
tf.app.flags.DEFINE_integer("iTrainD", 1, "")

tf.app.flags.DEFINE_float("fLrIni", 0.0004, "")
tf.app.flags.DEFINE_float("fBeta1", 0.5, "")
tf.app.flags.DEFINE_float("fBeta2", 0.999, "")
tf.app.flags.DEFINE_float("fEpsilon", 1e-8, "")
tf.app.flags.DEFINE_string("oOpt", 'adam', "adam, sgd, mom")

##################################################### Network Structure #######################################################################################

tf.app.flags.DEFINE_integer("iDimsC", 3, "")
tf.app.flags.DEFINE_integer("iFilterDimsD", 16, "")

tf.app.flags.DEFINE_integer("iDimsZ", 100, "")
tf.app.flags.DEFINE_integer("iFilterDimsG", 16, "")

tf.app.flags.DEFINE_boolean("bUseWN", False, "")
tf.app.flags.DEFINE_float("fIniScale", 2.0, "")

cfg(sys.argv)

############################################################################################################################################

from common.ops import *

set_enable_bias(True)
set_data_format('NCHW')
set_enable_wn(cfg.bUseWN)
set_ini_scale(cfg.fIniScale)


def load_dataset(dataset_name):

    if dataset_name == 'cifar10':
        cfg.iDimsC = 3
        return load_cifar10()

    if dataset_name == 'mnist':
        cfg.iDimsC = 1
        return load_mnist()

    if dataset_name == 'toy':
        cfg.iDimsC = 2
        return load_toy_data()


dataX, dataY, testX, testY = load_dataset(cfg.sDataSet)


def discriminator_dcgan(input, num_logits):

    iFilterDimsD = cfg.iFilterDimsD

    with tf.variable_scope('discriminator', tf.AUTO_REUSE):
        h0 = input

        h0 = conv2d(h0, iFilterDimsD * 1, ksize=3, stride=1, name='conv32')  # 32x32
        # h0 = batch_norm(h0, name='bn32')
        h0 = tf.nn.relu(h0)

        h0 = conv2d(h0, iFilterDimsD * 2, ksize=3, stride=2, name='conv32_16')  # 32x32 --> 16x16
        h0 = batch_norm(h0, name='bn16')
        h0 = tf.nn.relu(h0)

        h0 = conv2d(h0, iFilterDimsD * 4, ksize=3, stride=2, name='conv16_8')  # 16x16 --> 8x8
        h0 = batch_norm(h0, name='bn8')
        h0 = tf.nn.relu(h0)

        h0 = conv2d(h0, iFilterDimsD * 8, ksize=3, stride=2, name='conv8_4')  # 8x8 --> 4x4
        h0 = batch_norm(h0, name='bn4')
        h0 = tf.nn.relu(h0)

        h0 = avgpool(h0, 4, 4)
        h0 = tf.contrib.layers.flatten(h0)

        h0 = linear(h0, num_logits)

        return h0


def generator_dcgan(num_sample, z=None):

    iFilterDimsG = cfg.iFilterDimsG

    with tf.variable_scope('generator', tf.AUTO_REUSE):

        if z is None:
            h0 = tf.random_normal(shape=[num_sample, cfg.iDimsZ])

        else:
            h0 = z

        h0 = linear(h0, 4 * 4 * (iFilterDimsG * 8))  # linear 4x4
        h0 = batch_norm(h0, name='bn4')
        h0 = tf.nn.relu(h0)

        h0 = tf.reshape(h0, [-1, iFilterDimsG * 8, 4, 4])  # reshape 4x4

        h0 = deconv2d(h0, iFilterDimsG * 4, ksize=3, stride=2, name='deconv4_8')  # 4x4 --> 8x8
        h0 = batch_norm(h0, name='bn8')
        h0 = tf.nn.relu(h0)

        h0 = deconv2d(h0, iFilterDimsG * 2, ksize=3, stride=2, name='deconv8_16')  # 8x8 --> 16x16
        h0 = batch_norm(h0, name='bn16')
        h0 = tf.nn.relu(h0)

        h0 = deconv2d(h0, iFilterDimsG * 1, ksize=3, stride=2, name='deconv16_32')  # 16x16 --> 32x32
        h0 = batch_norm(h0, name='bn32')
        h0 = tf.nn.relu(h0)

        h0 = deconv2d(h0, cfg.iDimsC, ksize=3, stride=1, name='deconv32')  # 32x32
        h0 = tf.nn.tanh(h0)

        return h0


def discriminator_mlp(input, num_logits):

    iNumLayer = 5
    iFilterDimsD = cfg.iFilterDimsD

    with tf.variable_scope('discriminator', tf.AUTO_REUSE):

        h0 = input

        for i in range(iNumLayer):
            h0 = linear(h0, iFilterDimsD, name='linear%d' % i)
            if i > 0:
                h0 = batch_norm(h0, name='bn%d' % i)
            h0 = tf.nn.relu(h0)

        h0 = linear(h0, num_logits)

        return h0


def generator_mlp(num_sample, z=None):

    iNumLayer = 5
    iFilterDimsG = cfg.iFilterDimsG

    with tf.variable_scope('generator', tf.AUTO_REUSE):

        if z is None:
            h0 = tf.random_normal(shape=[num_sample, cfg.iDimsZ])
        else:
            h0 = z

        for i in range(iNumLayer):
            h0 = linear(h0, iFilterDimsG, name='linear%d' % i)
            h0 = batch_norm(h0, name='bn%d' % i)
            h0 = tf.nn.relu(h0)

        h0 = linear(h0, cfg.iDimsC)

        return h0


############################################################################################################################################

sTestName = (cfg.sResultTag + '_' if len(cfg.sResultTag) else "") + cfg.sDataSet

sTestCaseDir = cfg.sResultDir + sTestName + '/'
sSampleDir = sTestCaseDir + '/samples/'
sCheckpointDir = sTestCaseDir + '/checkpoint/'

makedirs(sCheckpointDir)
makedirs(sSampleDir)
makedirs(sTestCaseDir + '/code/')

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

copydir(cfg.sSourceDir, sTestCaseDir + '/code')

############################################################################################################################################

tf.logging.set_verbosity(tf.logging.ERROR)

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

if cfg.sDataSet == 'cifar10' or cfg.sDataSet == 'mnist':

    generator = generator_dcgan
    discriminator = discriminator_dcgan

    real_datas = tf.placeholder(tf.float32, [cfg.iBatchSize, cfg.iDimsC, 32, 32], name='real_datas')
    fake_datas = generator(cfg.iBatchSize)

else:

    generator = generator_mlp
    discriminator = discriminator_mlp

    real_datas = tf.placeholder(tf.float32, [cfg.iBatchSize, cfg.iDimsC], name='real_datas')
    fake_datas = generator(cfg.iBatchSize)

real_logits = discriminator(real_datas, 1)
fake_logits = discriminator(fake_datas, 1)

real_logits = tf.reshape(real_logits, [-1])
fake_logits = tf.reshape(fake_logits, [-1])

if cfg.bWGAN:

    dis_real_loss = -tf.reduce_mean(real_logits)
    dis_fake_loss = tf.reduce_mean(fake_logits)

    dis_gan_loss = dis_fake_loss + dis_real_loss
    gen_gan_loss = -tf.reduce_mean(fake_logits)

elif cfg.bLSGAN:

    dis_real_loss = tf.reduce_mean(tf.square(real_logits - 1.0))
    dis_fake_loss = tf.reduce_mean(tf.square(fake_logits - 0.0))

    dis_gan_loss = dis_fake_loss + dis_real_loss
    gen_gan_loss = tf.reduce_mean(tf.square(fake_logits - 1.0))

else:

    dis_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)))
    dis_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))

    dis_gan_loss = dis_fake_loss + dis_real_loss
    gen_gan_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.constant(1.0, shape=real_logits.get_shape())))

gen_total_loss = gen_gan_loss
dis_total_loss = dis_gan_loss

slopes = tf.constant(0.0)
gp_loss = tf.constant(0.0)

if cfg.bWGAN:

    alpha = tf.random_uniform(
        shape=[cfg.iBatchSize, 1] if cfg.sDataSet == 'toy' else [cfg.iBatchSize, 1, 1, 1],
        minval=0.,
        maxval=1.
    )

    differences = fake_datas - real_datas
    interpolates = real_datas + alpha * differences
    gradients = tf.gradients(discriminator(interpolates, 1)[0], [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1] if cfg.sDataSet == 'toy' else [1, 2, 3]))
    gp_loss = cfg.fWeightGP * tf.reduce_mean(tf.maximum(0.0, slopes - 1.) ** 2)
    dis_total_loss += gp_loss

############################################################################################################################################

tot_vars = tf.trainable_variables()
gen_vars = [var for var in tot_vars if 'generator' in var.name]
dis_vars = [var for var in tot_vars if 'discriminator' in var.name]

global_step = tf.Variable(0, trainable=False, name='global_step')
lr = cfg.fLrIni * tf.maximum(0., 1. - (tf.cast(global_step, tf.float32) / cfg.iMaxIter))

gen_optimizer = None

if cfg.oOpt == 'sgd':
    gen_optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
elif cfg.oOpt == 'mom':
    gen_optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=cfg.fBeta1, use_nesterov=True)
elif cfg.oOpt == 'adam':
    gen_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=cfg.fBeta1, beta2=cfg.fBeta2, epsilon=cfg.fEpsilon)

gen_gradient_value = gen_optimizer.compute_gradients(gen_total_loss, var_list=gen_vars)
gen_optimize_ops = gen_optimizer.apply_gradients(gen_gradient_value, global_step=global_step)

dis_optimizer = None

if cfg.oOpt == 'sgd':
    dis_optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
elif cfg.oOpt == 'mom':
    dis_optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=cfg.fBeta1, use_nesterov=True)
elif cfg.oOpt == 'adam':
    dis_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=cfg.fBeta1, beta2=cfg.fBeta2, epsilon=cfg.fEpsilon)

dis_gradient_value = dis_optimizer.compute_gradients(dis_total_loss, var_list=dis_vars)
dis_optimize_ops = dis_optimizer.apply_gradients(dis_gradient_value)


def param_count(gradient_value):
    total_param_count = 0
    for g, v in gradient_value:
        shape = v.get_shape()
        param_count = 1
        for dim in shape:
            param_count *= int(dim)
        total_param_count += param_count
    return total_param_count


logger.log("Generator Total Parameter Count: {}".format(locale.format("%d", param_count(gen_gradient_value), grouping=True)))
logger.log("Discriminator Total Parameter Count: {}".format(locale.format("%d", param_count(dis_gradient_value), grouping=True)))

saver = tf.train.Saver(max_to_keep=1000)

iter = 0
last_save_time = last_log_time = last_plot_time = time.time()

if cfg.bLoadCheckpoint:
    try:
        if load_model(saver, sess, sCheckpointDir):
            logger.log(" [*] Load SUCCESS")
            iter = sess.run(global_step)
            logger.load()
            logger.tick(iter)
            logger.log('\n\n')
            logger.flush()
            logger.log('\n\n')
        else:
            assert False
    except:
        logger.clear()
        logger.log(" [*] Load FAILED")
        ini_model(sess)
else:
    ini_model(sess)

fixed_noise = tf.constant(np.random.normal(size=(100, cfg.iDimsZ)).astype('float32'))
fixed_noise_gen = generator(100, fixed_noise)

data_gen = labeled_data_gen_epoch(dataX, dataY, cfg.iBatchSize)


def plot_generated_toy_data(X_gen, X_real, gen_iter, dir):

    plt.figure()
    plt.scatter(X_gen[:5000, 0],  X_gen[:5000, 1], s=0.5, color="red", marker="o", label='fake_data')
    plt.scatter(X_real[:5000, 0], X_real[:5000, 1], s=0.5, color="blue", marker="o", label='real_data')
    plt.legend()
    plt.savefig(dir + "/train_iter%s.png" % gen_iter)
    plt.close()


while iter <= cfg.iMaxIter:

    iter += 1
    start_time = time.time()

    for id in range(cfg.iTrainD):
        _datas, _labels = data_gen.__next__()
        _, _dis_total_loss, _dis_gan_loss, _gp_loss, _slopes, _lr, _real_logits, _fake_logits = sess.run(
            [dis_optimize_ops, dis_total_loss, dis_gan_loss, gp_loss, slopes, lr, real_logits, fake_logits],
            feed_dict={real_datas: _datas})

    for ig in range(cfg.iTrainG):
        _, _gen_total_loss, _gen_gan_loss, _lr, _fake_logits = sess.run(
            [gen_optimize_ops, gen_total_loss, gen_gan_loss, lr, fake_logits])

    logger.tick(iter)
    logger.info('time', time.time() - start_time)

    logger.info('loss_dis_total', _dis_total_loss)
    logger.info('loss_dis_gan', _dis_gan_loss)

    logger.info('loss_gen_total', _gen_total_loss)
    logger.info('loss_gen_gan', _gen_gan_loss)

    logger.info('logit_real', np.mean(_real_logits))
    logger.info('logit_fake', np.mean(_fake_logits))

    logger.info('loss_dis_gp', _gp_loss)
    logger.info('slope', np.mean(_slopes))

    logger.info('klr_dis', _lr * 1000)

    if time.time() - last_save_time > 60*30:
        logger.save()
        save_model(saver, sess, sCheckpointDir, step=iter)
        last_save_time = time.time()

    if time.time() - last_log_time > 60*1:
        logger.flush()
        last_log_time = time.time()

    if time.time() - last_plot_time > 60*10:
        logger.plot()

        _fixed_noise_gen = sess.run(fixed_noise_gen)
        if cfg.sDataSet == 'toy':
            plot_generated_toy_data(_fixed_noise_gen, _datas, iter, sSampleDir)
        else:
            save_images(_fixed_noise_gen, [10, 10], '{}/train_{:02d}_{:04d}.png'.format(sSampleDir, iter // 10000, iter % 10000))

        last_plot_time = time.time()


# from tensorflow.contrib.distributions import Mixture, Normal, Categorical
#
# pi = tf.get_variable('Generator.pi', dtype=tf.float32, initializer=cfg.iDimsZ * [cfg.iMixturesZ * [1.0 / cfg.iMixturesZ]], trainable=False)
# mu = tf.get_variable('Generator.mu', [cfg.iDimsZ, cfg.iMixturesZ], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-1, maxval=1), trainable=cfg.bMuTrainable)
# sigma = tf.get_variable('Generator.sigma', [cfg.iDimsZ, cfg.iMixturesZ], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=1.0 / cfg.iMixturesZ), trainable=cfg.bSigmaTrainable)
#
#
# def Generator(n_samples, noise=None):
#
#     with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
#
#         if noise is None:
#             if cfg.bUseMixture:
#                 with tf.device('/cpu:0'):
#                     mixtures = [Mixture(Categorical(pi[z]), [Normal(mu[z][i], sigma[z][i]) for i in range(cfg.iMixturesZ)]) for z in range(cfg.iDimsZ)]
#                     noise = tf.stack([mixture.sample(n_samples) for mixture in mixtures], 1)
#                     print('num mixture:', cfg.iMixturesZ, 'mu trainable:', cfg.bMuTrainable, 'sigma trainable:', cfg.bSigmaTrainable)
#             else:
#                 noise = tf.random_normal([n_samples, cfg.iDimsZ])
