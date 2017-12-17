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

def dataset_import(dataset):
    def parser(example_proto):
        features = { "latitude": tf.FixedLenFeature(shape=[1], dtype=tf.float32),
                    "longitude": tf.FixedLenFeature(shape=[1], dtype=tf.float32),
                    "time": tf.FixedLenFeature(shape=[1], dtype=tf.float32),
                    #"fullness": tf.FixedLenFeature(shape=[3], dtype=tf.float32) };
                    "fullness": tf.FixedLenFeature(shape=[1], dtype=tf.float32) };
        parsed = tf.parse_single_example(example_proto, features);
        return { "latitude": parsed["latitude"], 
                 "longitude": parsed["longitude"], 
                 "time": parsed["time"]}, parsed["fullness"];

    dataset = dataset.map(parser);
    dataset = dataset.batch(20);
    #dataset = dataset.repeat(1);
    iterator = dataset.make_one_shot_iterator();

    features, labels = iterator.get_next();
    return features, labels;

def dataset_import_train():
    return dataset_import(tf.data.TFRecordDataset("alerts_train.tfrecords"));

def dataset_import_eval():
    return dataset_import(tf.data.TFRecordDataset("alerts_verify.tfrecords"));

latitude = tf.feature_column.numeric_column("latitude");
longitude = tf.feature_column.numeric_column("longitude");
time = tf.feature_column.numeric_column("time");

estimator = tf.estimator.LinearRegressor( feature_columns = [latitude, longitude, time]);

estimator.train(dataset_import_train, steps=2000);

print(estimator.evaluate(dataset_import_eval));
