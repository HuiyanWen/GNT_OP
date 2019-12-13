#!/usr/bin/python
#coding=utf-8
import numpy as np
import struct
import tensorflow as tf
import os
import random

flags = tf.app.flags
tf.app.flags.DEFINE_string('dstpath', '../data_tfrecord/', "the path of the output tfrecords.")
tf.app.flags.DEFINE_string('gntpath', '../gnt_data/0.gnt', "the path of the input gnt files.")
tf.app.flags.DEFINE_string('lexicon_path', '../lexicon3755.txt', "the path of the dictionary.")
tf.app.flags.DEFINE_integer('is_train', 1, 'To determine the tfrecords\' name.')
tf.app.flags.DEFINE_integer('shuffle', 1, 'To determine whether to shuffle the images\' order.')
tf.app.flags.DEFINE_integer('num_per_tfrecord', 100000, "The number of pictures in every tfrecord.")
FLAGS = tf.app.flags.FLAGS

def generate_tf():
    """
        Convert gnt to tfrecord.
    """
    # Other parameters
    if FLAGS.is_train:
        try:
            os.stat(FLAGS.dstpath+'train/')
        except:
            os.makedirs(FLAGS.dstpath+'train/')
    else:
        try:
            os.stat(FLAGS.dstpath + 'test/')
        except:
            os.makedirs(FLAGS.dstpath + 'test/')
    count = 0
    image = []
    new_image = []
    tag_code = []
    new_tag_code = []
    height_global = []
    width_global = []
    new_height_global = []
    new_width_global = []
    result = {}
    #for z in range(0, 1):
    # Generate the dictionary
    f = open(FLAGS.lexicon_path, 'rb')
    line = f.readline()
    i = 0
    while line:
        result[struct.unpack('<H', line[:2])[0]] = i
        i += 1
        line = f.readline()

    ff = FLAGS.gntpath
    f = open(ff, 'rb')
    length = os.path.getsize(ff)
    print('length:', length)
    point = 0

    while point < length:
        count += 1
        length_bytes = struct.unpack('<I', f.read(4))[0]
        point += 4
        tag_code.append(f.read(2))
        point += 2
        width = struct.unpack('<H', f.read(2))[0]
        width_global.append(width)
        point += 2
        height = struct.unpack('<H', f.read(2))[0]
        height_global.append(height)
        point += 2
        img = f.read(width * height)
        image.append(img)
        point += width * height
    f.close()

    print('pic count:', count)
    num_shards = int(np.ceil(count / FLAGS.num_per_tfrecord))
    print('numshards:', num_shards)
    # Shuffle
    if FLAGS.shuffle:
        A = np.zeros(count, np.int32)
        for i in range(count):
            A[i] = i
        random.shuffle(A)
        for temp in range(count):
            new_height_global.append(height_global[A[temp]])
            new_image.append(image[A[temp]])
            new_tag_code.append(tag_code[A[temp]])
            new_width_global.append(width_global[A[temp]])
    else:
        new_height_global = height_global
        new_image = image
        new_tag_code = tag_code
        new_width_global = width_global
    for index in range(num_shards):
        if FLAGS.is_train:
            filename = os.path.join(FLAGS.dstpath+'train/', 'train_data.tfrecord-%.5d-of-%.5d' % (index, num_shards))
        else:
            filename = os.path.join(FLAGS.dstpath+'test/', 'test_data.tfrecord-%.5d-of-%.5d' % (index, num_shards))
        writer = tf.python_io.TFRecordWriter(filename)
        print('index = ', index)

        for x in range(index*FLAGS.num_per_tfrecord, (index+1)*FLAGS.num_per_tfrecord):
            if x >= count:
                break
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[new_image[x]])),
                'label': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[result[struct.unpack('<H', new_tag_code[x])[0]]])),
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[new_height_global[x]])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[new_width_global[x]]))
            }))
            serialized = example.SerializeToString()
            writer.write(serialized)


def main(_):
    generate_tf()


if __name__ == '__main__':
    tf.app.run()

