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

# made up of Examples of the following form:
#  features {
#    feature {
#      key: "fullness"
#      value {
#        float_list {
#          value: 1.0
#          value: 0.0
#          value: 0.0
#        }
#      }
#    }
#    feature {
#      key: "latitude"
#      value {
#        float_list {
#          value: 42.3471794128418
#        }
#      }
#    }
#    feature {
#      key: "longitude"
#      value {
#        float_list {
#          value: -71.09679412841797
#        }
#      }
#    }
#    feature {
#      key: "time"
#      value {
#        float_list {
#          value: 1389886848.0
#        }
#      }
#    }
#  }

def tfrecord_to_tensor(example_proto):
    features = { "latitude": tf.FixedLenFeature(shape=[1], dtype=tf.float32),
                 "longitude": tf.FixedLenFeature(shape=[1], dtype=tf.float32),
                 "time": tf.FixedLenFeature(shape=[1], dtype=tf.float32),
                 "fullness": tf.FixedLenFeature(shape=[3], dtype=tf.float32) };
    return tf.parse_single_example(example_proto, features);

training_dataset = tf.data.TFRecordDataset("alerts_train.tfrecords");
training_dataset = training_dataset.map(tfrecord_to_tensor);
training_iterator = training_dataset.make_one_shot_iterator();

verification_dataset = tf.data.TFRecordDataset("alerts_verify.tfrecords");
verification_dataset = verification_dataset.map(tfrecord_to_tensor);
verification_iterator = verification_dataset.make_one_shot_iterator();

sess = tf.Session();
print(sess.run([training_iterator.get_next()]));

# can be thought of as a 3 pixel greyscale image: lat, long, time
x = tf.placeholder(tf.float32, [None, 3]);

# we multiply each input tensor by unique weights and add unique biases
W = tf.Variable(tf.zeros([3, 3]));
b = tf.Variable(tf.zeros([3]));

# the output is the sum of the linear regression for gps and time (scalar)
fullness = tf.matmul(x, W) + b;

# placeholder for the correct answers
fullness_ = tf.placeholder(tf.float32, [None, 1]);
