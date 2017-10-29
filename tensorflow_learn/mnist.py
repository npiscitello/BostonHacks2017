#!/usr/bin/python
# see https://www.tensorflow.org/get_started/mnist/beginners
from tensorflow.examples.tutorials.mnist import input_data;
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True);

import tensorflow as tf;

#####
# Create the model
#####
# placeholder - this is an input variable
x = tf.placeholder(tf.float32, [None, 784]);

# variables - these are used by the network but can also be modified by the network
# weight - the input tensors are multiplied by this tensor
W = tf.Variable(tf.zeros([784, 10]));
# bias - this tensor is added to the multiplied tensor
b = tf.Variable(tf.zeros([10]));

# define the linear regression model
y = tf.matmul(x, W) + b;


#####
# Define loss and optimizer
#####
# define the placeholder for our 'correct' answers
y_ = tf.placeholder(tf.float32, [None, 10]);

# this is the cross-entropy and minimization function, how we calculate and minimize our loss. 
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y));
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy);


#####
# Run the model!
#####
sess = tf.InteractiveSession();
tf.global_variables_initializer().run();

for _ in range(0,1000):
    if _ % 100 == 0:
        print("Training run " + str(_) + " of 1000");
    batch_xs, batch_ys = mnist.train.next_batch(100);
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys});


#####
# How well did the model do?
#####
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1));
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));
print("Model accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}));
