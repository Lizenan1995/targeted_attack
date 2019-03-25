import numpy as np
import keras.backend as K
import tensorflow as tf
from collections import defaultdict
import matplotlib.pyplot as plt

def generate_input(model, orig, target, eps=1):
    # orig is the original input
    # target is the target input
    # we generate adv closed to target from orig
    orig = np.expand_dims(orig, axis=0)
    target = np.expand_dims(target, axis=0)
    target_label = np.argmax(model.predict(target), axis=1)

    input_tensor = model.input
    output = model.layers[-4].output
    fun = K.function([input_tensor], [output])

    bench_output = fun([target])[0]
    loss1 = tf.reduce_mean(tf.square(output - bench_output))
    loss2 = tf.reduce_mean(tf.square(input_tensor - orig))
    loss  = loss1 + eps * loss2
    grads = K.gradients(loss, input_tensor)[0]
    iterate = K.function([input_tensor],[loss, grads])

    adv_x = orig.copy() + 0.1 * np.random.normal(size=orig.shape)
    max_iter = 10000

    for i in range(max_iter):
        loss, grads = iterate([adv_x])
        if i % 100 == 0:
        	print("iterate {}, loss is {}".format(i, loss))
        adv_x = np.clip(adv_x - 0.01 * grads, 0, 1)
        pred = np.argmax(model.predict(adv_x), axis=1)
        if(pred == target_label):
            print("attack successfully")
            break
    return adv_x

def fgsm(model, orig, target, eps = 0.3):
    orig = np.expand_dims(orig, axis=0)
    target = np.expand_dims(target, axis=0)
    target_label = np.argmax(model.predict(target), axis=1)
    from keras.utils.np_utils import to_categorical
    target_label = to_categorical(target_label, 10)
    input_tensor = model.input
    output = model.layers[-1].output 
    loss  = K.categorical_crossentropy(target_label, output, from_logits=False)
    grads = K.gradients(loss, input_tensor)[0]
    iterate = K.function([input_tensor],[loss, grads])
    adv_x = orig.copy()
    for i in range(1000):
        loss, grads = iterate([adv_x])
        if i % 100 == 0:
            print("iterate {}, loss is {}".format(i, loss))
        adv_x = np.clip(adv_x - 0.01 * grads, 0, 1)
        pred = np.argmax(model.predict(adv_x), axis=1)
        if(pred == np.argmax(target_label,axis=1)):
            print("attack successfully")
            break
    return adv_x


if __name__ == '__main__':
    # preprocess the data set
    from keras.datasets import mnist
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_train /= 255
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    from keras.models import load_model
    model = load_model('../PycharmProjects/regress/models/lenet5.h5')

    adv_x = generate_input(model, x_test[1], x_test[2], eps = 0.1)
    print('original is {}'.format(np.argmax(model.predict(x_test[1].reshape(1, 28, 28, 1)), axis=1)))
    print('target is {}'.format(np.argmax(model.predict(x_test[2].reshape(1, 28, 28, 1)), axis=1)))
    print('adv is {}'.format(np.argmax(model.predict(adv_x.reshape(1, 28, 28, 1)), axis=1)))
    pert1 = (adv_x - x_test[1]).reshape(-1)
    print('the score is {}'.format(np.linalg.norm(pert1)))
    from scipy.misc import imsave
    imsave('orig1.png',(x_test[1]*255).astype('int').reshape(28,28))
    imsave('adv1.png',(adv_x*255).astype('int').reshape(28,28))

    adv_x = fgsm(model, x_test[1], x_test[2], eps = 0.3)
    print('original is {}'.format(np.argmax(model.predict(x_test[1].reshape(1, 28, 28, 1)), axis=1)))
    print('target is {}'.format(np.argmax(model.predict(x_test[2].reshape(1, 28, 28, 1)), axis=1)))
    print('adv is {}'.format(np.argmax(model.predict(adv_x.reshape(1, 28, 28, 1)), axis=1)))
    # from scipy.misc import imsave
    # imsave('orig1.png',(x_test[1]*255).astype('int').reshape(28,28))
    # imsave('adv1.png',(adv_x*255).astype('int').reshape(28,28))
    pert2 = (adv_x - x_test[1]).reshape(-1)

    print(np.sum(pert1*pert2) / np.linalg.norm(pert1) / np.linalg.norm(pert2))
