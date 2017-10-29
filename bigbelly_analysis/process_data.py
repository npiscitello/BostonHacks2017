#!/bin/python
# turn the downloaded csv into a TFRecord

from datetime import datetime;
import tensorflow as tf;

f_raw = open("big-belly-alerts-2014.csv");
f_tfr = tf.python_io.TFRecordWriter("big-belly-alerts-2014.tfrecords");

# get rid of the header
f_raw.readline();

line_raw = [];
time = 0;
fullness = [];

for i in f_raw:
    line = i.rstrip().split(',');

    # inputs
    time = datetime.strptime(line[1], "%m/%d/%Y %I:%M:%S %p").timestamp();

    # output - one-hot encoded
    if( line[2] == "GREEN" ):
        fullness = [1,0,0];
    elif( line[2] == "YELLOW" ):
        fullness = [0,1,0];
    elif( line[3] == "RED" ):
        fullness = [0,0,1];

    f_tfr.write(tf.train.Example(features=tf.train.Features(feature={
        "latitude": tf.train.Feature(float_list=tf.train.FloatList(value=[float(line[4][2:])])),
        "longitude": tf.train.Feature(float_list=tf.train.FloatList(value=[float(line[5][1:-2])])),
        "time": tf.train.Feature(float_list=tf.train.FloatList(value=[time])),
        "fullness": tf.train.Feature(float_list=tf.train.FloatList(value=fullness))
        })).SerializeToString());

f_tfr.close();
