#!/usr/bin/python
#coding=utf-8
import numpy as np
import struct
from PIL import Image
import tensorflow as tf
import os
dstpath = '.'
count = 0
path = 'D:/TensorFlow_project/tfrecord_read&write/'
flag = 0
image = []
tag_code = []
result = {}
###################求总长度，保存到image_list中，num_shards为tfrecord的数目######################################
for z in range(0, 1):
    ######################
    global result
    # result = {}
    f = open('lexicon94.txt', 'rb')
    line = f.readline()
    i = 0
    while line:
        # print(line[:2].decode())
        result[struct.unpack('<H', line[:2])[0]] = i
        i += 1
        # print("%s" % eval('line[:2]'))
        line = f.readline()
    ######################
    ff = str(z) + '.gnt'
    f = open(ff, 'rb')

    length = len(f.read())
    f.seek(-length, 1)
    point = 0
    while point < length:
        global count
        count += 1
        #print('f.read(4):', f.read(4))
        #f.seek(-4, 1)
        length_bytes = struct.unpack('<I', f.read(4))[0]
        #print('length_bytes:', length_bytes)
        point += 4
        global tag_code
        tag_code.append(f.read(2))
        #print('tag_code: ', tag_code)
        #if tag_code == : print("yes")
        point += 2
        width = struct.unpack('<H', f.read(2))[0]
        #print('width: ', width)
        point += 2
        height = struct.unpack('<H', f.read(2))[0]
        #print('height: ', height)
        point += 2
        #temp = f.read(width * height)
        #b = np.array(f.read(width * height))
        #image = np.row_stack((image, b))
        #mage.resize(64, 64)
        image.append(f.read(width * height))
        point += width * height
        # for x in range(0, height):
        #     for y in range(0, width):
        #         pixel = struct.unpack('<B', f.read(1))[0]
        #         point += 1
        #         point_cur += 1
        #f.seek(-point_cur, 1)
        #print(count, f.read(point_cur))
        #f.read(10)
        #image.append([f.read(point_cur-10)])
    f.close()
#print(tag_code[0], image[0])
#print(image[count-1])

#print('图片数:', count)
length_per_shard = 10000  # 每个记录文件的样本长度
num_shards = int(np.ceil(count / length_per_shard))
#################################################################################################################
#print('记录文件个数：', num_shards)

for index in range(num_shards):
    train_data = True
    if train_data:
        filename = os.path.join(dstpath, 'train_data.tfrecord-%.5d-of-%.5d' % (index, num_shards))
    else:
        filename = os.path.join(dstpath, 'test_data.tfrecord-%.5d-of-%.5d' % (index, num_shards))
    #print(filename)
    writer = tf.python_io.TFRecordWriter(filename)

    for x in range(0, count):
        # 读取图像
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image[x]])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[result[struct.unpack('<H', tag_code[x])[0]]]))}))
        print(result[struct.unpack('<H', tag_code[x])[0]])
        # 序列化
        #global tag_code
        #print(tag_code)
        serialized = example.SerializeToString()
        # 写入文件
        writer.write(serialized)
        if x == length_per_shard - 1:
            count -= length_per_shard