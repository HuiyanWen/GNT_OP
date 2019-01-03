#!/usr/bin/python
#coding=utf-8
import numpy as np
import struct
from PIL import Image
import tensorflow as tf
import os
dstpath = '.'
count = 0
path = 'H:/苏统华的空间/94类字母样本for荟俨/'
flag = 0
image = []
tag_code = []
result = {}
###################求总长度，保存到image_list中，num_shards为tfrecord的数目######################################
for z in range(2, 3):
    ###############生成字典###################
    f = open(path+'lexicon94.txt', 'rb')
    line = f.readline()
    i = 0
    while line:
        result[struct.unpack('<H', line[:2])[0]] = i
        i += 1
        line = f.readline()

    ff = path + str(z) + '.gnt'
    f = open(ff, 'rb')
    ###########################################
    length = os.path.getsize(ff)
    print('length:', length)
    point = 0

    while point < length:
        count += 1

        #print('f.read(4):', f.read(4))
        #f.seek(-4, 1)
        length_bytes = struct.unpack('<I', f.read(4))[0]
        #print('length_bytes:', length_bytes)
        point += 4
        tag_code.append(f.read(2))
        #print('tag_code: ', tag_code)
        point += 2
        width = struct.unpack('<H', f.read(2))[0]
        #print('width: ', width)
        point += 2
        height = struct.unpack('<H', f.read(2))[0]
        #print('height: ', height)
        point += 2
        image.append(f.read(width * height))
        point += width * height
        # if count == 6000000:
        #     del tag_code[:]
        #     del image[:]
        #     #print(tag_code[0])
        #     #print(image[0])
        # if count <= 6000000:
        #      #print('count continue : ', count)
        #      continue
        # if count >= 9000000:
        #     break
    f.close()
#print('count - 6000000 = ', count-6000000)
#print(tag_code[0], image[0])
#print(image[count-1])
print('图片数:', count)
length_per_shard = 10000  # 每个记录文件的样本长度
#num_shards = int(np.ceil((count-6000000) / length_per_shard))
num_shards = int(np.ceil(count / length_per_shard))
print('numshards:', num_shards)
#################################################################################################################

for index in range(num_shards):
    train_data = True
    if train_data:
        filename = os.path.join(dstpath, 'train_data.tfrecord-%.5d-of-%.5d' % (index, num_shards))
    else:
        filename = os.path.join(dstpath, 'test_data.tfrecord-%.5d-of-%.5d' % (index, num_shards))
    #print(filename)
    writer = tf.python_io.TFRecordWriter(filename)
    print('index = ', index)
    #print(index*length_per_shard, (index+1)*length_per_shard)
    for x in range(index*length_per_shard, (index+1)*length_per_shard):
        # 读取图像
        #print(x)
        if x >= count:
            break
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image[x]])),
            'label': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[result[struct.unpack('<H', tag_code[x])[0]]]))}))
        # print(result[struct.unpack('<H', tag_code[x])[0]])
        serialized = example.SerializeToString()
        writer.write(serialized)
