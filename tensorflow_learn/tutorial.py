#!/usr/bin/python

import tensorflow as tf

# Paramaters - these are the weights
M = tf.Variable([0.3], dtype = tf.float32);
b = tf.Variable([-0.3], dtype = tf.float32);

# Input and output placeholders - these are the neruon 'variables'
x = tf.placeholder(tf.float32);
y = tf.placeholder(tf.float32);

# This is what the network does
linear_model = M * x + b;

# This is the error used to back propogate the parameters
loss = tf.reduce_sum(tf.square(linear_model - y));

# This is the actual back propogation function
optimizer = tf.train.GradientDescentOptimizer(0.01);
train = optimizer.minimize(loss);

# training data
x_train = [1, 2, 3, 4];
y_train = [0, -1, -2, -3];

# training loop
init = tf.global_variables_initializer();
sess = tf.Session();
sess.run(init);

for i in range(0,1000):
    sess.run(train, {x: x_train, y: y_train});

# evaluate training accuracy
curr_M, curr_b, curr_loss = sess.run([M, b, loss], {x: x_train, y: y_train});
print("M: %s b: %s loss: %s"%(curr_M, curr_b, curr_loss));
