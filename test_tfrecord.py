import tensorflow as tf
import os
import matplotlib.pyplot as plt

slim = tf.contrib.slim

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
tf.app.flags.DEFINE_string('tfrecord_path', '../dataset/train/', "The number of pictures in every tfrecord.")
FLAGS = tf.app.flags.FLAGS

def file_name(file_dir, is_train):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if is_train == True:
                if os.path.splitext(file)[0] == 'train_data':
                    L.append(os.path.join(root, file))
            else:
                if os.path.splitext(file)[0] == 'test_data':
                    L.append(os.path.join(root, file))
    return L

def get_batch(dirpath, is_train=True):
    filenames = file_name(dirpath, is_train=is_train)
    dataset = tf.data.TFRecordDataset(filenames)
    def parser(record):
        features = {
                'image': tf.FixedLenFeature([], tf.string, default_value=""),
                'label': tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
                'height': tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
                'width': tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
            }
        parsed = tf.parse_single_example(record, features)
        image = tf.decode_raw(parsed["image"], tf.uint8)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # convert to [0,1]
        width = tf.cast(parsed["width"], tf.int32)
        height = tf.cast(parsed["height"], tf.int32)
        image = tf.subtract(1.0, image)
        image = tf.reshape(image, [height, width, 1])
        new_image = tf.image.resize_images(image, (112, 112))
        label = tf.cast(parsed["label"], tf.int32)
        return new_image, label
    dataset = dataset.map(parser)
    dataset = dataset.batch(32)
    dataset = dataset.repeat(1)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()
    return image_batch, label_batch

def testImage():
    print('Begin test')
    sess = tf.Session()
    with sess:
        image_batch, label_batch = get_batch(is_train=True, dirpath=FLAGS.tfrecord_path)
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        while True:
            try:
                train_images_batch, train_labels_batch = sess.run([image_batch, label_batch])
                # Plot the image and its label
                images = train_images_batch[0]
                print(images.shape)
                h, w, c = images.shape
                assert c == 1
                images = images.reshape(h, w)
                plt.imshow(images, cmap='gray')
                print(train_labels_batch[0])
                plt.show()
            except tf.errors.OutOfRangeError:
                print("End of dataset")
                break

if __name__ == '__main__':
    testImage()
