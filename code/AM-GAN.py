import sys, locale
from os import path

locale.setlocale(locale.LC_ALL, '')
sys.path.append(path.dirname(path.abspath(__file__)))
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
SOURCE_DIR = path.dirname(path.dirname(path.abspath(__file__))) + '/'

import time
from common.ops import *
from common.score import *
from common.data_loader import *
from common.logger import Logger

############################################################################################################################################

cfg = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("sDataSet", "cifar10", "cifar10, mnist, toy")
tf.app.flags.DEFINE_string("sResultTag", "test_v0", "your tag for each test case")

tf.app.flags.DEFINE_boolean("bLoadCheckpoint", False, "bLoadCheckpoint")
tf.app.flags.DEFINE_string("sResultDir", SOURCE_DIR + "result/", "where to save the checkpoint and sample")

tf.app.flags.DEFINE_boolean("bAMGAN", True, "")

tf.app.flags.DEFINE_integer("iMaxIter", 1000000, "")
tf.app.flags.DEFINE_integer("iBatchSize", 100, "")

tf.app.flags.DEFINE_integer("iTrainG", 1, "")
tf.app.flags.DEFINE_integer("iTrainD", 1, "")

tf.app.flags.DEFINE_float("fLrIni", 0.0004, "")
tf.app.flags.DEFINE_float("fBeta1", 0.5, "")
tf.app.flags.DEFINE_float("fBeta2", 0.999, "")
tf.app.flags.DEFINE_float("fEpsilon", 1e-8, "")

tf.app.flags.DEFINE_string("oDecay", 'linear', "exp, linear")
tf.app.flags.DEFINE_string("oOpt", 'adam', "adam, sgd, mom")
tf.app.flags.DEFINE_string("oAct", 'lrelu', "relu, lrelu, selu")

tf.app.flags.DEFINE_integer("iDimsC", 3, "")
tf.app.flags.DEFINE_integer("iDimsZ", 100, "")

tf.app.flags.DEFINE_integer("iFilterDimsG", 96, "")
tf.app.flags.DEFINE_integer("iFilterDimsD", 64, "")

tf.app.flags.DEFINE_float("fDropRate", 0.3, "")

cfg(sys.argv)

allocate_gpu()

############################################################################################################################################

def discriminator_dcgan(input, num_logits):

    iFilterDimsD = cfg.iFilterDimsD

    with tf.variable_scope('discriminator', tf.AUTO_REUSE):

        h0 = input
        h0 = noise(h0, 0.1, bAdd=True)

        h0 = conv2d(h0, iFilterDimsD * 1, ksize=3, stride=1, name='conv32')  # 32x32
        # h0 = batch_norm(h0, name='bn32')
        h0 = activate(h0, cfg.oAct)
        h0 = dropout(h0, cfg.fDropRate)

        h0 = conv2d(h0, iFilterDimsD * 2, ksize=3, stride=2, name='conv32_16')  # 32x32 --> 16x16
        h0 = batch_norm(h0, name='bn16')
        h0 = activate(h0, cfg.oAct)
        h0 = dropout(h0, cfg.fDropRate)

        h0 = conv2d(h0, iFilterDimsD * 4, ksize=3, stride=2, name='conv16_8')  # 16x16 --> 8x8
        h0 = batch_norm(h0, name='bn8')
        h0 = activate(h0, cfg.oAct)
        h0 = dropout(h0, cfg.fDropRate)

        h0 = conv2d(h0, iFilterDimsD * 8, ksize=3, stride=2, name='conv8_4')  # 8x8 --> 4x4
        h0 = batch_norm(h0, name='bn4')
        h0 = activate(h0, cfg.oAct)
        h0 = dropout(h0, cfg.fDropRate)

        h0 = avgpool(h0, h0.get_shape().as_list()[2], h0.get_shape().as_list()[3])
        h0 = tf.contrib.layers.flatten(h0)
        h0 = dropout(h0, cfg.fDropRate)

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
        h0 = tf.reshape(h0, [-1, iFilterDimsG * 8, 4, 4])  # reshape 4x4
        h0 = batch_norm(h0, name='bn4')
        h0 = activate(h0, cfg.oAct)

        h0 = deconv2d(h0, iFilterDimsG * 4, ksize=3, stride=2, name='deconv4_8')  # 4x4 --> 8x8
        h0 = batch_norm(h0, name='bn8')
        h0 = activate(h0, cfg.oAct)

        h0 = deconv2d(h0, iFilterDimsG * 2, ksize=3, stride=2, name='deconv8_16')  # 8x8 --> 16x16
        h0 = batch_norm(h0, name='bn16')
        h0 = activate(h0, cfg.oAct)

        h0 = deconv2d(h0, iFilterDimsG * 1, ksize=3, stride=2, name='deconv16_32')  # 16x16 --> 32x32
        h0 = batch_norm(h0, name='bn32')
        h0 = activate(h0, cfg.oAct)

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
            h0 = activate(h0, cfg.oAct)

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
            h0 = activate(h0, cfg.oAct)

        h0 = linear(h0, cfg.iDimsC)

        return h0

############################################################################################################################################

def load_dataset(dataset_name):

    if dataset_name == 'cifar10':
        cfg.iDimsC = 3
        return load_cifar10()

    if dataset_name == 'mnist':
        cfg.iDimsC = 1
        return load_mnist()

    if dataset_name == 'toy':
        cfg.iDimsC = 2
        return load_toy_data(n_mixture=10)


def param_count(gradient_value):
    total_param_count = 0
    for g, v in gradient_value:
        shape = v.get_shape()
        param_count = 1
        for dim in shape:
            param_count *= int(dim)
        total_param_count += param_count
    return total_param_count


def plot_generated_toy_data(X_gen, X_real, gen_iter, dir):

    plt.figure()
    plt.scatter(X_gen[:5000, 0],  X_gen[:5000, 1], s=0.5, color="red", marker="o", label='fake_data')
    plt.scatter(X_real[:5000, 0], X_real[:5000, 1], s=0.5, color="blue", marker="o", label='real_data')
    plt.legend()
    plt.savefig(dir + "/train_iter%s.png" % gen_iter)
    plt.close()


def gen_n_images(n):
    images = []
    for i in range(n // cfg.iBatchSize + 1):
        images.append(sess.run(fake_datas))
    images = np.concatenate(images, 0)
    return images[:n]


ref_am_preds, ref_am_activations = None, None
am_model = PreTrainedDenseNet()

ref_icp_preds, ref_icp_activations = None, None
icp_model = PreTrainedInception()


def get_score(samples):

    global ref_icp_preds, ref_icp_activations, ref_am_preds, ref_am_activations

    if ref_icp_activations is None:
        logger.log('Evaluating Reference Statistic: icp_model')
        ref_icp_preds, ref_icp_activations = icp_model.get_preds(dataX.transpose(0, 2, 3, 1))
        logger.log('\nref_icp_score: %.3f\n' % InceptionScore.inception_score_H(ref_icp_preds)[0])

    if ref_am_preds is None:
        logger.log('Evaluating Reference Statistic: am_model')
        ref_am_preds, ref_am_activations = am_model.get_preds(dataX.transpose(0, 2, 3, 1))

    logger.log('Evaluating Generator Statistic')
    icp_preds, icp_activcations = icp_model.get_preds(samples.transpose(0, 2, 3, 1))
    am_preds, am_activations = am_model.get_preds(samples.transpose(0, 2, 3, 1), dataX.transpose(0, 2, 3, 1))

    icp_score = InceptionScore.inception_score_KL(icp_preds)
    am_score = AMScore.am_score(am_preds, ref_am_preds)[0]
    fid = FID.get_FID_with_activations(icp_activcations, ref_icp_activations)

    return icp_score, am_score, fid

############################################################################################################################################

dataX, dataY, testX, testY = load_dataset(cfg.sDataSet)
data_gen = labeled_data_gen_epoch(dataX, dataY, cfg.iBatchSize)

sTestName = (cfg.sResultTag + '_' if len(cfg.sResultTag) else "") + cfg.sDataSet

sTestCaseDir = cfg.sResultDir + sTestName + '/'
sSampleDir = sTestCaseDir + '/samples/'
sCheckpointDir = sTestCaseDir + '/checkpoint/'

makedirs(cfg.sResultDir)
makedirs(sTestCaseDir)
makedirs(sSampleDir)
makedirs(sCheckpointDir)
makedirs(sTestCaseDir + '/code/')

logger = Logger()
logger.set_dir(sTestCaseDir)
logger.set_casename(sTestName)

logger.log(sTestCaseDir)

commandline = ''
for arg in ['CUDA_VISIBLE_DEVICES="0" python3'] + sys.argv:
    commandline += arg + ' '
logger.log(commandline)

logger.log(str_flags(cfg.__flags))

copydir(SOURCE_DIR + "code/", sTestCaseDir + '/source/code/')
copydir(SOURCE_DIR + "common/", sTestCaseDir + '/source/common/')

tf.logging.set_verbosity(tf.logging.ERROR)

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

############################################################################################################################################

if cfg.sDataSet == 'cifar10' or cfg.sDataSet == 'mnist':
    generator = generator_dcgan
    discriminator = discriminator_dcgan

    real_datas = tf.placeholder(tf.float32, [None, cfg.iDimsC, 32, 32], name='real_datas')
    fake_datas = generator(cfg.iBatchSize)
else:
    generator = generator_mlp
    discriminator = discriminator_mlp

    real_datas = tf.placeholder(tf.float32, [None, cfg.iDimsC], name='real_datas')
    fake_datas = generator(cfg.iBatchSize)

fake_labels = tf.placeholder(tf.int32, shape=[None])
real_labels = tf.placeholder(tf.int32, shape=[None])

num_logits = 11 if cfg.bAMGAN else 1
real_logits = discriminator(real_datas, num_logits)
fake_logits = discriminator(fake_datas, num_logits)

if cfg.bAMGAN:

    dis_real_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=real_logits, labels=real_labels))
    dis_fake_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(real_labels)*10))

    dis_gan_loss = dis_fake_loss + dis_real_loss
    gen_gan_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fake_logits, labels=tf.stop_gradient(tf.to_int32(tf.arg_max(fake_logits[:, :10], 1)))))

else:

    dis_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)))
    dis_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))

    dis_gan_loss = dis_fake_loss + dis_real_loss
    gen_gan_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits)))

gen_total_loss = gen_gan_loss
dis_total_loss = dis_gan_loss

tot_vars = tf.trainable_variables()
gen_vars = [var for var in tot_vars if 'generator' in var.name]
dis_vars = [var for var in tot_vars if 'discriminator' in var.name]

global_step = tf.Variable(0, trainable=False, name='global_step')

if cfg.oDecay == 'linear':
    lr = cfg.fLrIni * tf.maximum(0., 1. - (tf.cast(global_step, tf.float32) / cfg.iMaxIter))
elif cfg.oDecay == 'exp':
    lr = tf.train.exponential_decay(cfg.fLrIni, global_step, cfg.iMaxIter // 10, 0.5, True)
else:
    lr = cfg.fLrIni

gen_optimizer = None

if cfg.oOpt == 'sgd':
    gen_optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
elif cfg.oOpt == 'mom':
    gen_optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=cfg.fBeta1, use_nesterov=True)
elif cfg.oOpt == 'adam':
    gen_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=cfg.fBeta1, beta2=cfg.fBeta2, epsilon=cfg.fEpsilon)

gen_gradient_values = gen_optimizer.compute_gradients(gen_total_loss, var_list=gen_vars)
gen_optimize_ops = gen_optimizer.apply_gradients(gen_gradient_values, global_step=global_step)

dis_optimizer = None

if cfg.oOpt == 'sgd':
    dis_optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
elif cfg.oOpt == 'mom':
    dis_optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=cfg.fBeta1, use_nesterov=True)
elif cfg.oOpt == 'adam':
    dis_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=cfg.fBeta1, beta2=cfg.fBeta2, epsilon=cfg.fEpsilon)

dis_gradient_values = dis_optimizer.compute_gradients(dis_total_loss, var_list=dis_vars)
dis_optimize_ops = dis_optimizer.apply_gradients(dis_gradient_values)

saver = tf.train.Saver(max_to_keep=1000)

############################################################################################################################################

iter = 0
last_save_time = last_log_time = last_plot_time = last_score_time = time.time()

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

logger.log("Generator Total Parameter Count: {}".format(locale.format("%d", param_count(gen_gradient_values), grouping=True)))
logger.log("Discriminator Total Parameter Count: {}".format(locale.format("%d", param_count(dis_gradient_values), grouping=True)))
logger.log("\n\n")

while iter <= cfg.iMaxIter:

    iter += 1
    start_time = time.time()

    for id in range(cfg.iTrainD):
        _datas, _labels = data_gen.__next__()
        _, _dis_total_loss, _dis_gan_loss, _lr, _real_logits, _fake_logits = sess.run(
            [dis_optimize_ops, dis_total_loss, dis_gan_loss, lr, real_logits, fake_logits],
            feed_dict={real_datas: _datas, real_labels: _labels})

    for ig in range(cfg.iTrainG):
        _, _gen_total_loss, _gen_gan_loss, _lr, _fake_logits = sess.run(
            [gen_optimize_ops, gen_total_loss, gen_gan_loss, lr, fake_logits])

    logger.tick(iter)
    logger.info('klr', _lr * 1000)
    logger.info('time', time.time() - start_time)

    logger.info('loss_dis_total', _dis_total_loss)
    logger.info('loss_dis_gan', _dis_gan_loss)

    logger.info('loss_gen_total', _gen_total_loss)
    logger.info('loss_gen_gan', _gen_gan_loss)

    if cfg.bAMGAN:
        logger.info('logit_real_rsum', np.mean(np.sum(softmax(_real_logits)[:, :10], 1)))
        logger.info('logit_fake_rsum', np.mean(np.sum(softmax(_fake_logits)[:, :10], 1)))
        logger.info('logit_real_rmax', np.mean(np.max(softmax(_real_logits)[:, :10], 1)))
        logger.info('logit_fake_rmax', np.mean(np.max(softmax(_fake_logits)[:, :10], 1)))
    else:
        logger.info('logit_real_r2', np.mean(_real_logits))
        logger.info('logit_fake_r2', np.mean(_fake_logits))

    if time.time() - last_score_time > 60*60 and cfg.sDataSet == 'cifar10':
        icp_score, am_score, fid = get_score(gen_n_images(50000))
        logger.info('score_fid', fid)
        logger.info('score_am', am_score)
        logger.info('score_icp', icp_score)
        last_score_time = time.time()

    if time.time() - last_save_time > 60*30:
        logger.save()
        save_model(saver, sess, sCheckpointDir, step=iter)
        last_save_time = time.time()
        logger.log('Model Saved\n\n')

    if time.time() - last_log_time > 60*1:
        logger.flush()
        last_log_time = time.time()

    if time.time() - last_plot_time > 60*10:
        logger.plot()
        _fixed_noise_gen = sess.run(fixed_noise_gen)
        if cfg.sDataSet == 'toy':
            plot_generated_toy_data(_fixed_noise_gen, _datas, iter, sSampleDir)
        else:
            save_images(_fixed_noise_gen.transpose(0, 2, 3, 1), [10, 10], '{}/train_{:02d}_{:04d}.png'.format(sSampleDir, iter // 10000, iter % 10000))
        last_plot_time = time.time()