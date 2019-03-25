"""
Implementation of example defense.
This defense loads inception v1 checkpoint and classifies all images using loaded checkpoint.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception, vgg, resnet_v1
from scipy.misc import imread
from scipy.misc import imresize
from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.attacks import Model
from cleverhans.compat import softmax_cross_entropy_with_logits
from PIL import Image
import keras.backend as K
from cleverhans.utils_tf import clip_eta
from cleverhans import utils_tf

slim = tf.contrib.slim
tf.flags.DEFINE_string(
    'checkpoint_path', './inception_v1/inception_v1.ckpt', 'Path to checkpoint for inception network.')
tf.flags.DEFINE_string(
    'input_dir', './dev_data/', 'Input directory with images.')
tf.flags.DEFINE_string(
    'output_dir', './results/', 'Output directory with images.')
tf.flags.DEFINE_integer(
    'image_width', 224, 'Width of each input images.')
tf.flags.DEFINE_integer(
    'image_height', 224, 'Height of each input images.')
tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')
tf.flags.DEFINE_integer(
    'num_classes', 110, 'Number of Classes')
tf.flags.DEFINE_float(
    'eps', 1, 'eps')
tf.flags.DEFINE_float(
    'lamda', 3, 'lamda')
tf.flags.DEFINE_float(
    'noise', 1.0, 'noise')
tf.flags.DEFINE_integer(
    'attack_iter', 10000, 'attack iterations')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    labels = np.zeros(batch_shape[0], dtype=np.int32)
    filenames = []

    target_images = np.zeros(batch_shape)

    idx = 0
    batch_size = batch_shape[0]
    fstr = []
    label = []
    with open(os.path.join(input_dir, 'dev.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fstr.append(row['filename'])
            label.append(row['trueLabel'])
    with open(os.path.join(input_dir, 'dev.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = os.path.join(input_dir, row['filename'])

            for i in range(len(label)):
                if label[i] == row['targetedLabel']:
                    filepath2 = fstr[i]

            with open(filepath, 'rb') as f:
                raw_image = imread(f, mode='RGB').astype(np.float)
                image = imresize(raw_image, [FLAGS.image_height, FLAGS.image_width]) / 255.0

            filepath2 = os.path.join(input_dir, filepath2)
            with open(filepath2, 'rb') as f:
                raw_image = imread(f, mode='RGB').astype(np.float)
                image2 = imresize(raw_image, [FLAGS.image_height, FLAGS.image_width]) / 255.0

            # Images for inception classifier are normalized to be in [-1, 1] interval.
            images[idx, :, :, :] = image * 2.0 - 1.0
            target_images[idx, :, :, :] = image2 * 2.0 - 1.0

            labels[idx] = int(row['targetedLabel'])
            filenames.append(os.path.basename(filepath))

            idx += 1
            if idx == batch_size:
                yield filenames, images, labels, target_images
                filenames = []
                images = np.zeros(batch_shape)
                labels = np.zeros(batch_shape[0], dtype=np.int32)

                target_images = np.zeros(batch_shape)

                idx = 0
        if idx > 0:
            yield filenames, images, labels, target_images


def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with open(os.path.join(output_dir, filename), 'wb') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            # resize back to [299, 299]
            r_img = imresize(img, [299, 299])
            Image.fromarray(r_img).save(f, format='PNG')


class InceptionModel(Model):
    """Model class for CleverHans library."""
    def __init__(self, nb_classes):
        super(InceptionModel, self).__init__(nb_classes=nb_classes,
                                             needs_dummy_fprop=True)
        self.built = False

    def __call__(self, x_input, return_logits=False, return_layer=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            _, end_points = inception.inception_v1(
                x_input, num_classes=self.nb_classes, is_training=False,
                reuse=reuse)
        self.built = True
        self.logits = end_points['Logits']
        # Strip off the extra reshape op at the output
        self.probs = end_points['Predictions'].op.inputs[0]
        if return_logits:
            return self.logits
        elif return_layer:
            return end_points['Conv2d_2c_3x3']
        else:
            return self.probs

    def get_logits(self, x_input):
        return self(x_input, return_logits=True)

    def get_probs(self, x_input):
        return self(x_input)

    def get_layer(self, x_input):
        return self(x_input, return_layer=True)


def generate_input(model,
                   x_input,
                   adv_x,
                   target_input,
                   target_input_class,
                   clip_min=-1.0,
                   clip_max=1.0,
                   eps=1.0):
    bench_output = model.get_layer(target_input)
    output = model.get_layer(adv_x)

    loss1 = tf.reduce_mean(tf.square(output - bench_output))
    loss2 = tf.reduce_mean(tf.square(adv_x - x_input))
    loss = loss1 + eps * loss2
    grads = tf.gradients(loss, adv_x, colocate_gradients_with_ops=True)[0]

    adv_x = adv_x - FLAGS.lamda * grads

    adv_x = utils_tf.clip_by_value(adv_x, clip_min, clip_max)

    return loss, grads


def generate(model,
             x_input,
             target_input,
             adv_x,
             target_class_input,
             max_iter=FLAGS.attack_iter,
             clip_max=1.0,
             clip_min=-1.0,
             eps=FLAGS.eps):

    asserts = []
    asserts.append(utils_tf.assert_greater_equal(x_input,
                                                 tf.cast(clip_min, x_input.dtype)))
    asserts.append(utils_tf.assert_less_equal(x_input,
                                              tf.cast(clip_max, x_input.dtype)))

    def cond(i, ax):
        # pred = model.get_predicted_class(ax)
        # for j in range(FLAGS.batch_size):
        #     result = tf.cond(tf.equal(tf.gather(pred, j), tf.gather(target_class_input, j)),
        #                      lambda: tf.constant(False, dtype=tf.bool),
        #                      lambda: tf.constant(True, dtype=tf.bool))
        # return tf.less(i, max_iter) & result
        return tf.less(i, max_iter)

    def body(i, ax):
        bench_output = model.get_layer(target_input)
        output = model.get_layer(ax)

        loss1 = tf.reduce_mean(tf.square(output - bench_output))
        loss2 = tf.reduce_mean(tf.square(ax - x_input))
        loss = loss1 + eps * loss2
        grads = tf.gradients(loss, ax, colocate_gradients_with_ops=True)[0]
        ax = ax - FLAGS.lamda * grads
        ax = utils_tf.clip_by_value(ax, clip_min, clip_max)
        return i+1, ax

    _, adv_x = tf.while_loop(
        cond, body, [tf.zeros([]), adv_x], back_prop=True)

    with tf.control_dependencies(asserts):
        adv_x = tf.identity(adv_x)

    return adv_x


def main(_):
    """Run the sample attack"""
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    nb_classes = FLAGS.num_classes
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        target_input = tf.placeholder(tf.float32, shape=batch_shape)
        adv_x = tf.placeholder(tf.float32, shape=batch_shape)
        target_class_input = tf.placeholder(tf.int64, shape=[FLAGS.batch_size])
        model = InceptionModel(nb_classes)
        # Run computation
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            x_adv = generate(model, x_input, target_input, adv_x, target_class_input)
            saver = tf.train.Saver(slim.get_model_variables())
            saver.restore(sess, FLAGS.checkpoint_path)

            for filenames, images, tlabels, target_images in load_images(FLAGS.input_dir, batch_shape):
                adv_images = images + FLAGS.noise * np.random.normal(size=(FLAGS.batch_size, 224, 224, 3))
                adv_res = sess.run(x_adv,
                                   feed_dict={
                                       x_input: images,
                                       target_input: target_images,
                                       adv_x: adv_images,
                                       target_class_input: tlabels
                                   })

                save_images(adv_res, filenames, FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()
