import os
from glob import glob
import cv2
import numpy as np
import tensorflow as tf
import math
from multiprocessing.pool import Pool
from ctpn_tools.libs import random_uniform_num, readxml, cal_rpn, IMAGE_MEAN
anno_dir = r"data\Annotations"
images_dir = r"data\JPEGImages"
main_data = glob(anno_dir+ '/*.xml') 


class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, datas, batch_size=50,images_dir=r"data\JPEGImages",shuffle=True):
        self.batch_size = batch_size
        self.datas = datas
        self.datas = np.array(self.datas)
        self.indexes = np.arange(len(self.datas))
        np.random.shuffle(self.indexes)
        self.shuffle = shuffle
        self.images_dir = images_dir
        
    def __len__(self):
        # The number of steps of each epoch
        return math.ceil(len(self.datas) / float(self.batch_size))

    def _single_sample(self, xml_path):
        gtbox, imgfile = readxml(xml_path)
        img = cv2.imread(os.path.join(self.images_dir, imgfile))
        h, w, c = img.shape
        # clip image
        if np.random.randint(0, 100) > 50:
            img = img[:, ::-1, :]
            newx1 = w - gtbox[:, 2] - 1
            newx2 = w - gtbox[:, 0] - 1
            gtbox[:, 0] = newx1
            gtbox[:, 2] = newx2

        [cls, regr], _ = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gtbox)
        # zero-center by mean pixel
        m_img = img - IMAGE_MEAN
        m_img = np.expand_dims(m_img, axis=0)

        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])

        #
        cls = np.expand_dims(cls, axis=0)
        cls = np.expand_dims(cls, axis=1)
        # regr = np.expand_dims(regr,axis=1)
        regr = np.expand_dims(regr, axis=0)

        return [m_img, {'rpn_class_reshape': cls, 'rpn_regress_reshape': regr}]
    
    def __getitem__(self, index):
        batch_index = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # print("="*60)
        # print(index)
        # print(batch_index[0])
        # print("="*60)

        # pool
        # poop = Pool(4)       
        # results = poop.map(self._single_sample,self.datas[batch_index])
        # poop.close()
        # poop.join()
        
        # Simple list
        results = [self._single_sample(items) for items in self.datas[batch_index]]
        
        return results

    def on_epoch_end(self):
        if self.shuffle == True:
            print("Epoch ends! Start shuffling")
            np.random.shuffle(self.indexes)


class DataLoader:

    def __init__(self, main_data):
        self.main_data = main_data
        self.steps_per_epoch = len(main_data)
        self.data_queue = []
        
    def _init_queue(self):
        self.data_queue = next(self.base_generstor_iterator)

    def load_data(self):
        self.base_generstor = DataGenerator(self.main_data)
        self.base_generstor_iterator = iter(self.base_generstor)
        while True:
    
            if len(self.data_queue) == 0:
                self._init_queue()
    
            result = self.data_queue.pop(0)
            yield result[0],result[1]


