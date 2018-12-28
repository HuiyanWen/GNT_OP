#!/usr/bin/python

import struct
from PIL import Image
import os

count = 0
path = 'D:/TensorFlow_project/tfrecord_read&write/'

for z in range(0, 1):
    ff = str(z) + '.gnt'
    f = open(ff, 'rb')
    length = len(f.read())
    f.seek(-length, 1)
    point = 0
    while point < length:
        global count
        count += 1
        length_bytes = struct.unpack('<I', f.read(4))[0]
        point += 4
        print('length_bytes:', length_bytes)
        tag_code = f.read(2)
        point += 2
        print('tag_code:', tag_code)
        width = struct.unpack('<H', f.read(2))[0]
        point += 2
        print('width:', width)
        height = struct.unpack('<H', f.read(2))[0]
        point += 2
        print('height:', height)
        im = Image.new('RGB', (width, height))
        img_array = im.load()
        print(img_array[0, 7])
        for x in range(0, height):
            for y in range(0, width):
                pixel = struct.unpack('<B', f.read(1))[0]
                img_array[y, x] = (pixel, pixel, pixel)
                point += 1
        filename = str(count) + '.png'
        if (os.path.exists(path + '%d/' % z)):
            filename = path + '%d/' % z + filename
            print(filename)
            im.save(filename)
        else:
            os.makedirs(path + '%d/' % z + '/')
            filename = path + '%d/' % z + '/' + filename
            print(filename)
            im.save(filename)
    f.close()