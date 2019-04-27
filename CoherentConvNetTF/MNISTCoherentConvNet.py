""" Convolutional Neural Network.
Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

Extended with: https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/04_Save_Restore.ipynb

Author: Ben Steel
Date: 01/04/19
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import pickle
import os

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.05
num_steps = 5000
batch_size = 128
display_step = 10
gamma = 8
bias_var = 0
continue_training = False

# Network Parameters
# dropout = 1 # Dropout, probability to keep units

params = {
    'dim' : 28,
    'n_input' : 28*28,
    'n_channels' : 1,
    'wc1' : {
        'f' : 5,
        'n_filts' : 10
    },
    'wc2' : {
        'f' : 5,
        'n_filts' : 20
    },
    'wp3' : {
        'pool' : 4,
        'n_pools' : 20
    },
    'wd4' : {
        'n' : (28//4)*(28//4)*20,
        'm' : 512
    },
    'wd5' : {
        'n' : 512,
        'm' : 10
    },
    'n_classes' : 10
}

# tf Graph input
X = tf.placeholder(tf.float32, [None, params['n_input']])
Y = tf.placeholder(tf.float32, [None, params['n_classes']])
# keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

def save_path():
    return 'mnist_' + str(params['wc1']['n_filts']) + 'f' + str(params['wc1']['f']) \
            + '_' + str(params['wc2']['n_filts']) + 'f' + str(params['wc2']['f']) \
            + '_p' + str(params['wp3']['pool']) \
            + '_1f' + str(params['wd4']['m']) \
            + '_g' + str(gamma) + '_lr_05' \
            + '_bias' + str(bias_var) + 'b' \
            + '_try' + str(5) + '.pkl'


# Create some wrappers for simplicity
def conv2d(x, W, strides=1):
    # Conv2D wrapper, with bias and relu activation
    xa = tf.nn.conv2d(x['re'], W['re'], strides=[1, strides, strides, 1], padding='SAME')
    yb = tf.nn.conv2d(x['im'], W['im'], strides=[1, strides, strides, 1], padding='SAME')

    xb = tf.nn.conv2d(x['re'], W['im'], strides=[1, strides, strides, 1], padding='SAME')
    ya = tf.nn.conv2d(x['im'], W['re'], strides=[1, strides, strides, 1], padding='SAME')

    return {'re' : (xa - yb), 'im' : (xb + ya)}

def nonlin(x):
    return tf.truediv(0.5*gamma, tf.add(tf.square(x), (0.5*gamma)**2))

def nonlinComplex(x, x0):

    xyab = tf.square(x['re']) + tf.square(x['im']) - tf.square(x0['re']) - tf.square(x0['im'])
    g2 = tf.square(0.5 * gamma)
    denom = g2 + tf.square(xyab)

    real = ((g2 * x['re']) - (0.5 * gamma * x['im'] * xyab)) / denom

    imag = ((g2 * x['im']) - (0.5 * gamma * x['re'] * xyab)) / denom

    return {'re' : real, 'im' : imag}

def matmul(x, W):
    xa = tf.matmul(x['re'], W['re'])
    yb = tf.matmul(x['im'], W['im'])

    xb = tf.matmul(x['re'], W['im'])
    ya = tf.matmul(x['im'], W['re'])

    return {'re' : (xa - yb), 'im' : (xb + ya)}

def add(x, w):
    return {'re' : x['re'] + w['re'], 'im' : x['im'] + w['im']}

# Create model
def conv_net(image, weights, biases):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]

    image = tf.reshape(image, shape=[-1, params['dim'], params['dim'], 1])

    x = {
        're' : image,
        'im' : tf.zeros(tf.shape(image))
    }

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'])
    nonlin1 = nonlinComplex(conv1, biases['bc1'])

    # Convolution Layer
    conv2 = conv2d(nonlin1, weights['wc2'])
    nonlin2 = nonlinComplex(conv2, biases['bc2'])

    # Max Pooling (down-sampling)
    conv3 = conv2d(nonlin2, weights['wp3'], strides=4)
    nonlin3 = nonlinComplex(conv3, biases['bp3'])

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = {
        'im' : tf.reshape(nonlin3['re'], [-1, weights['wd4']['re'].get_shape().as_list()[0]]),
        're' : tf.reshape(nonlin3['im'], [-1, weights['wd4']['im'].get_shape().as_list()[0]])
    }

    fc1 = matmul(fc1, weights['wd4'])
    nonlin4 = nonlinComplex(fc1, biases['bd4'])

    # Output, class prediction
    fc2 = matmul(nonlin4, weights['wd5'])
    fc2 = add(fc2, biases['bd5'])

    out = tf.square(fc2['re']) + tf.square(fc2['im'])

    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1' : {
        're' : tf.Variable(tf.random_normal([params['wc1']['f'], params['wc1']['f'], params['n_channels'], params['wc1']['n_filts']])),
        'im' : tf.Variable(tf.random_normal([params['wc1']['f'], params['wc1']['f'], params['n_channels'], params['wc1']['n_filts']]))
    },
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2' : {
        're' : tf.Variable(tf.random_normal([params['wc2']['f'], params['wc2']['f'], params['wc1']['n_filts'], params['wc2']['n_filts']])),
        'im' : tf.Variable(tf.random_normal([params['wc2']['f'], params['wc2']['f'], params['wc1']['n_filts'], params['wc2']['n_filts']]))
    },
    # 4x4 conv, 64 inputs, 64 outputs
    'wp3' : {
        're': tf.Variable(tf.random_normal([params['wp3']['pool'], params['wp3']['pool'], params['wc2']['n_filts'], params['wp3']['n_pools']])),
        'im': tf.Variable(tf.random_normal([params['wp3']['pool'], params['wp3']['pool'], params['wc2']['n_filts'], params['wp3']['n_pools']]))
    },
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd4' : {
        're': tf.Variable(tf.random_normal([params['wd4']['n'], params['wd4']['m']])),
        'im': tf.Variable(tf.random_normal([params['wd4']['n'], params['wd4']['m']]))
    },
    # 1024 inputs, 10 outputs (class prediction)
    'wd5' : {
        're': tf.Variable(tf.random_normal([params['wd5']['n'], params['n_classes']])),
        'im': tf.Variable(tf.random_normal([params['wd5']['n'], params['n_classes']]))
    }
}

biases = {
    # 5x5 conv, 1 input, 32 outputs
    'bc1' : {
        're': tf.Variable(tf.random_uniform([params['wc1']['n_filts']], minval=-bias_var, maxval=bias_var)),
        'im': tf.Variable(tf.random_uniform([params['wc1']['n_filts']], minval=-bias_var, maxval=bias_var))
    },
    # 5x5 conv, 32 inputs, 64 outputs
    'bc2' : {
        're': tf.Variable(tf.random_uniform([params['wc2']['n_filts']], minval=-bias_var, maxval=bias_var)),
        'im': tf.Variable(tf.random_uniform([params['wc2']['n_filts']], minval=-bias_var, maxval=bias_var))
    },
    # 4x4 conv, 64 inputs, 64 outputs
    'bp3' : {
        're': tf.Variable(tf.random_uniform([params['wp3']['n_pools']], minval=-bias_var, maxval=bias_var)),
        'im': tf.Variable(tf.random_uniform([params['wp3']['n_pools']], minval=-bias_var, maxval=bias_var))
    },
    # fully connected, 7*7*64 inputs, 1024 outputs
    'bd4' : {
        're': tf.Variable(tf.random_uniform([params['wd4']['m']], minval=-bias_var, maxval=bias_var)),
        'im': tf.Variable(tf.random_uniform([params['wd4']['m']], minval=-bias_var, maxval=bias_var))
    },
    # 1024 inputs, 10 outputs (class prediction)
    'bd5' : {
        're': tf.Variable(tf.random_uniform([params['n_classes']], minval=-bias_var, maxval=bias_var)),
        'im': tf.Variable(tf.random_uniform([params['n_classes']], minval=-bias_var, maxval=bias_var))
    }
}

# Construct model
logits = conv_net(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

saver = tf.train.Saver()

# Start training
sess = tf.Session()



total_iterations = 0
best_validation_loss = 1000
last_improvement = 0
require_improvement = num_steps // 2

save_dir = 'checkpoints/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path_val = os.path.join(save_dir, 'best_validation')
save_path_loss = save_path()

if not continue_training:
    # Run the initializer
    sess.run(init)

    training_loss = []
    validation_loss = []

    training_acc = []
    validation_acc = []

else:
    saver.restore(sess=sess, save_path=save_path_val)

    with open(save_path_loss, 'rb') as file:
        [training_loss, validation_loss, training_acc, validation_acc, test_accuracy] = pickle.load(file)

for step in range(1, num_steps+1):

    total_iterations += 1

    batch_x, batch_y = mnist.train.next_batch(batch_size)
    # Run optimization op (backprop)
    sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
    if step % display_step == 0 or step == 1:
        # Calculate batch loss and accuracy
        tr_loss, tr_acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                Y: batch_y,})

        batch_x, batch_y = mnist.validation.next_batch(batch_size)

        val_loss, val_acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                Y: batch_y,})
        print("Step " + str(step) + ", Minibatch Loss= " + \
                "{:.4f}".format(tr_loss) + ", Training Accuracy= " + \
                "{:.3f}".format(tr_acc) + ", Validation Loss= " + \
                "{:.4f}".format(val_loss) + ", Validation Accuracy= " + \
                "{:.3f}".format(val_acc))

        training_loss.append(tr_loss)
        training_acc.append(tr_acc)

        validation_loss.append(val_loss)
        validation_acc.append(val_acc)

        if val_loss < best_validation_loss:
            # Update the best-known validation accuracy.
            best_validation_loss = val_loss
            
            # Set the iteration for the last improvement to current.
            last_improvement = total_iterations

            # Save all variables of the TensorFlow graph to file.
            saver.save(sess=sess, save_path=save_path_val)

        # If no improvement found in the required number of iterations.
        if total_iterations - last_improvement > require_improvement:
            print("No improvement found in a while, stopping optimization.")

            # Break out from the for-loop.
            break

print("Optimization Finished!")

# restore best variables
saver.restore(sess=sess, save_path=save_path_val)

# Calculate accuracy for 256 MNIST test images
test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images[:512],
                                                Y: mnist.test.labels[:512],})

print("Testing Accuracy:", test_accuracy)
    
to_save = [training_loss, validation_loss, training_acc, validation_acc, test_accuracy]

with open(save_path_loss, 'wb') as file:
    pickle.dump(to_save, file)

sess.close()