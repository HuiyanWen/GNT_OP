#!/usr/bin/python
#coding=utf-8
import numpy as np
import struct
import tensorflow as tf
import os

def generate_tf():
    """
        Convert gnt to tfrecod.
        Args:
            dstpath: The output path of the generate tfrecord.
            count: The image count of the whole gnt file.
            path: The root path of files.
            image: To save the image info.
            tag_code: To save the label info.
            height_global: To save the height of the image.
            width_global: To save the witdh of the image.
            result: To save the one-hot code of the label.
            train_data: To determine the output tfrecord name.
    """
    # Parameters
    dstpath = '.'
    count = 0
    path = 'G:/华为生态数据GNT格式/'
    flag = 0
    image = []
    tag_code = []
    height_global = []
    width_global = []
    result = {}
    train_data = True

    for z in range(0, 1):
        # Generate the dictionary
        f = open(path+'lexicon3755.txt', 'rb')
        line = f.readline()
        i = 0
        while line:
            result[struct.unpack('<H', line[:2])[0]] = i
            i += 1
            line = f.readline()

        ff = path + str(z) + '.gnt'
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
            image.append(f.read(width * height))
            point += width * height
        f.close()

    print('pic count:', count)
    length_per_shard = 100000  # The length of every tfrecord
    num_shards = int(np.ceil(count / length_per_shard))
    print('numshards:', num_shards)

    for index in range(num_shards):
        if train_data:
            filename = os.path.join(dstpath, 'train_data.tfrecord-%.5d-of-%.5d' % (index, num_shards))
        else:
            filename = os.path.join(dstpath, 'test_data.tfrecord-%.5d-of-%.5d' % (index, num_shards))
        writer = tf.python_io.TFRecordWriter(filename)
        print('index = ', index)
        for x in range(index*length_per_shard, (index+1)*length_per_shard):
            # Read the image
            if x >= count:
                break
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image[x]])),
                'label': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[result[struct.unpack('<H', tag_code[x])[0]]])),
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height_global[x]])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width_global[x]]))
            }))
            serialized = example.SerializeToString()
            writer.write(serialized)

if __name__ == '__main__':
    generate_tf()
