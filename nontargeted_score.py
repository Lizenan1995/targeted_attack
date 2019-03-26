import os
from scipy.misc import imread
from scipy.misc import imresize
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
from cleverhans.attacks import Model
import csv
slim = tf.contrib.slim


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


dir1 = './nontargeted_results/'
dir2 = './dev_data/'

adv_files = []
trueLabel = []
score = []

with open('./dev_data/dev.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        adv_files.append(row['filename'])
        trueLabel.append(row['trueLabel'])

model = InceptionModel(110)
x = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])
pred = model.get_predicted_class(x)
i = 0
sess = tf.Session()
saver = tf.train.Saver(slim.get_model_variables())
saver.restore(sess, './inception_v1/inception_v1.ckpt')
success = 0
for adv_f in adv_files:
    with open(dir1 + adv_f, 'rb') as f:
        raw_image = imread(f, mode='RGB').astype(np.float)
        adv_image = imresize(raw_image, [224, 224]) / 255.0
        adv = adv_image * 2.0 - 1.0

    label = sess.run(pred, feed_dict={x: adv.reshape(1, 224, 224, 3)})
    if label[0] != int(trueLabel[i]):
        with open(dir2 + adv_f, 'rb') as f:
            raw_image = imread(f, mode='RGB').astype(np.float)
            image = imresize(raw_image, [224, 224]) / 255.0
            img = image * 2.0 - 1.0
        score.append(np.linalg.norm(adv-img))
        success += 1
    else:
        score.append(128)
    i += 1

score = np.array(score)
print(np.sum(score) / 110)
print(success)
