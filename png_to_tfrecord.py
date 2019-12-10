# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import random
import numpy as np
from PIL import Image
import logging
import matplotlib.pyplot as plt

slim = tf.contrib.slim

# 输出类别
NUM_CLASSES = 214

# 图片大小
IMAGE_SIZE = 64

logger = logging.getLogger('HIT Chinese HandWriting Recognition')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

random.seed(0)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def get_files(dirpath):
    ”“”
    获取文件相对路径和标签（非one-hot） 返回一个元组
    args：
        dirpath:数据所在的目录，记作父目录，
                假设有10000类数据，则父目录下有10000个子目录，每个子目录存放着对应的图片
    “”“
    image_list = []
    label_list = []

    # traversal
    classes = [x for x in os.listdir(dirpath) if os.path.isdir(dirpath)]

    # 遍历每一个子文件夹，使用enumerate函数可以返回下标(子文件夹的下标是标签)
    for index, name in enumerate(classes):
        class_path = os.path.join(dirpath, name)
        class_path = class_path.replace('\\', '/')
        # 遍历子目录下的每一个文件
        for img_name in os.listdir(class_path):
            # 每一个图片的完整路径（全路径）
            img_path = os.path.join(class_path, img_name)
            img_path = img_path.replace('\\', '/')
            # 追加
            image_list.append(img_path)
            label_list.append(index)

    # 保存打乱后的文件和标签
    images = []
    labels = []
    # 打乱文件顺序，连续打乱两次,list将元组转换成列表，元组不可修改，列表可以修改
    indices = list(range(len(image_list)))
    random.shuffle(indices)
    for i in indices:
        images.append(image_list[i])
        labels.append(label_list[i])
    random.shuffle([images, labels])

    print('样本长度为:', len(images))
    return images, labels

def WriteTFRecord(image_list, label_list, dstpath, train_data, IMAGE_HEIGHT=64, IMAGE_WIDTH=64):
    ”“”
    把指定目录下的数据写入同一个TFRecord格式文件中
    args:
        dirpath:数据所在的目录，记作父目录
                假设有10类数据，则父目录下有10个子目录，每个子目录存放着对应的图片
        dstpath:保存TFRecord文件的目录
        train_data:表示传入的文件是不是训练集文件所在的路径
        IMAGE_HEIGHT
        IMAGE_WIDTH
    “”“
    if not os.path.isdir(dstpath):
        os.mkdir(dstpath)
    # 获取所有数据文件路径，以及对应标签
    # image_list, label_list = get_files(dirpath)

    # 把海量数据写入多个TFRecord文件
    length_per_shard = 10000  # 每个记录文件的样本长度
    num_shards = int(np.ceil(len(image_list) / length_per_shard))

    print('记录文件个数：', num_shards)
    # 依次写入每一个TFRecord文件
    for index in range(num_shards):
        # 按0000n-of-0000m的后缀区分文件，n代表当前文件标号，m代表文件总数
        if train_data:
            filename = os.path.join(dstpath, 'train_data.tfrecord-%.5d-of-%.5d' % (index, num_shards))
        else:
            filename = os.path.join(dstpath, 'test_data.tfrecord-%.5d-of-%.5d' % (index, num_shards))

        print(filename)

        # 创建对象，用于向记录文件写入记录
        writer = tf.python_io.TFRecordWriter(filename)
        # 起始索引
        idx_start = index * length_per_shard
        # 结束索引
        idx_end = np.min([(index + 1) * length_per_shard - 1, len(image_list)])
        # 遍历子目录下的每一个文件
        for img_path, label in zip(image_list[idx_start:idx_end], label_list[idx_start:idx_end]):
            # 读取图像
            print(img_path, label)
            img_origin = Image.open(img_path)
            img = img_origin.resize((IMAGE_HEIGHT, IMAGE_WIDTH))
            #img = img.convert("RGB")
            image = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))
            # 序列化
            serialized = example.SerializeToString()
            # 写入文件
            writer.write(serialized)
        writer.close()

def file_match(s, root='.'):
    dirs = []
    matchs = []
    for current_name in os.listdir(root):
        add_root_name = os.path.join(root, current_name)
        if os.path.isdir(add_root_name):
            dirs.append(add_root_name)
        elif os.path.isfile(add_root_name) and s in add_root_name:
            matchs.append(add_root_name)
    return matchs

def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[0] == 'train_data':
                L.append(os.path.join(root, file))
    return L

def get_batch2(dirpath = 'D:/TensorFlow_project/tfrecord_read&write/94/test/',is_train = True):
    filenames = file_name(dirpath)
    dataset = tf.data.TFRecordDataset(filenames)
    def parser(record):
        features = {
                'image': tf.FixedLenFeature([], tf.string, default_value=""),
                'label': tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
            }
        parsed = tf.parse_single_example(record, features)
        image = tf.decode_raw(parsed["image"], tf.uint8)
        image = tf.reshape(image, [112, 112, 1])
        new_image = tf.image.resize_images(image, (224, 224))
        label = tf.cast(parsed["label"], tf.int32)
        return new_image, label
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=10000).batch(32)
    dataset = dataset.repeat(1)
    iterator = dataset.make_one_shot_iterator()
    # iterator = dataset.make_initializable_iterator()
    image_batch, label_batch = iterator.get_next()
    return image_batch, label_batch

def testImage():
    print('Begin test')
    sess = tf.Session()
    with sess:
        image_batch, label_batch = get_batch2(is_train=False)
        # next_elem = iterator.get_next()
        # return next_elem
        #print([next_iter['image'], next_iter['label']])
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        while True:
            try:
                #images, labels = sess.run([image_batch, label_batch])
                print("yes")
            except tf.errors.OutOfRangeError:
                print("End of dataset")
                break
            else:
                train_images_batch, train_labels_batch = sess.run([image_batch, label_batch])
                # Show images and labels
                images = train_images_batch[0]
                h, w, c = images.shape
                assert c == 1
                images = images.reshape(h, w)
                plt.imshow(images)
                print(train_labels_batch[0])
                plt.show()

def maketf():
    # 训练集所在目录
    dirpath = 'D:/TensorFlow_project/tfrecord_read&write/train'
    training_step = 1
    files = file_match('data.tfrecord')
    if len(files) == 0:
        print('开始读图片并写入tfrecord文件中...........')
        image_list, label_list = get_files(dirpath)
        WriteTFRecord(image_list, label_list, dstpath='.', train_data=True)
        print('写入完毕！\n')

if __name__ == '__main__':
    #maketf() #从png或jpg制作tfrecord
    testImage() #解析tfrecord，验证准确性
