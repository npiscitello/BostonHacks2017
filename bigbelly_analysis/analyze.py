#!/usr/bin/python

# This uses machine learning to predict, based on location and time of day, how full a BigBelly
# compacting trash can is.
# The data set provides the following metrics:
# description: text description of the location
# timestamp: time of report
# fullness: how much trash is in the compactor (3 levels, 20%, 60%, 100%)
# collection: was this report triggered by a collection event?
# Location: GPS lat/long location

# example: Atlantic & Milk,01/01/2014 12:41:00 AM,YELLOW,FALSE,"(42.35870062, -71.051439)"
# The compactor at Atlantic and Milk (precisely 42.35870062, -71.051439) was about 60% full just
# past midnight on January 1st 2014 and was not being collected from at the time.

import tensorflow as tf

# can be thought of as a 3 pixel greyscale image: lat, long, time
x = tf.placeholder(tf.float32, [None, 3]);

# we multiply each input tensor by unique weights and add unique biases
W = tf.Variable(tf.zeros([3, 3]));
b = tf.Variable(tf.zeros([3]));

# the output is the sum of the linear regression for gps and time (scalar)
fullness = tf.matmul(x, W) + b;

# placeholder for the correct answers
fullness_ = tf.placeholder(tf.float32, [None, 1]);
