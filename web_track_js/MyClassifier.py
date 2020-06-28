"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import datetime
import time
import tensorflow as tf
import numpy as np
from sklearn.svm import SVC

import My_facenet
from facenet.src import facenet
from My_align_dataset_mtcnn import My_align_dataset_mtcnn

import argparse
import sys
import pickle
from facenet.src.facenet import to_rgb, prewhiten
#
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('/home/lanmei/PycharmProjects/facenet/src')))
# sys.path.append(BASE_DIR)

# Original  >>>>>>>> src.classifier.py
# Modified  >>>>>>>> web_track_js.MyClassifier.py
# Set original main function divide into fit function and prediction function.


# 'model_status', type=str, choices=['TRAIN', 'CLASSIFY'],help='Indicates if a new classifier should be trained or a classification ' +
#                           'model should be used for classification', default='CLASSIFY'

# 'model', type=str,help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file''classifier_filename',help='Classifier model file name as a pickle (.pkl) file. ' +
#                      'For training this is the output and for classification this is an input.'
# 'classifier_filename', help='Classifier model file name as a pickle (.pkl) file. ' +'For training this is the output and for classification this is an input.'
# '--use_split_dataset', help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +
#                       'Otherwise a separate test set can be specified using the test_data_dir option.',action='store_true'
# '--batch_size', type=int,help='Number of images to process in a batch.', default=90
# '--image_size', type=int,help='Image size (height, width) in pixels.', default=160
# '--seed', type=int,help='Random seed.', default=666
# '--min_nrof_images_per_class', type=int,help='Only include classes with at least this number of images in the dataset', default=20
# '--nrof_train_images_per_class', type=int,help='Use this number of images from each class for training and the rest for testing', default=10
class ImageClassifier:

    def __init__(self,model_status='TRAIN',
                 model='D:/WorkProjects/PycharmProjects/facenet/facenet_tijiaoban/FaceNet/src/models/20180408-102900/20180408-102900.pb',
                 classifier_filename='D:/WorkProjects/PycharmProjects/facenet/facenet_tijiaoban/FaceNet/src/models/my_classifier.pkl',
                 seed=666,
                 batch_size=32,
                 image_size=160,
                 use_split_dataset=False,
                 min_nrof_images_per_class=20,
                 nrof_train_images_per_class=10):

        self.model_status=model_status
        self.model=model
        self.classifier_filename=classifier_filename
        self.seed=seed
        self.batch_size=batch_size
        self.image_size=image_size
        self.use_split_dataset=use_split_dataset
        self.min_nrof_images_per_class=min_nrof_images_per_class
        self.nrof_train_images_per_class=nrof_train_images_per_class
        # self.sess = tf.Session(graph=self.graph)
        self.need_load_model=True


    # train()
    def fit(self):
        self.model_status='TRAIN'
        self.data_dir=My_align_dataset_mtcnn(0,160,32,True,0.25).output_dir

        with tf.Graph().as_default():
            with tf.Session() as sess:
                np.random.seed(seed=self.seed)
                if self.use_split_dataset:
                    dataset_tmp = facenet.get_dataset(self.data_dir)
                    train_set, test_set = self.split_dataset(dataset_tmp, self.min_nrof_images_per_class,
                                                             self.nrof_train_images_per_class)
                    if (self.model_status== 'TRAIN'):
                        dataset = train_set
                    elif (self.model_status == 'CLASSIFY'):
                        dataset = test_set
                else:
                    dataset = facenet.get_dataset(self.data_dir)

                # Check that there are at least one training image per class
                for cls in dataset:
                    assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

                self.paths, self.labels = facenet.get_image_paths_and_labels(dataset)
                print("paths", self.paths)
                print("labels", self.labels)
                print('Number of classes: %d' % len(dataset))
                print('Number of images: %d' % len(self.paths))

                # Load the model
                print('Loading feature extraction model',self.model)
                load_model_YesOrNo=facenet.load_model(self.model)

                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                # Run forward pass to calculate embeddings
                print('Calculating features for images')
                nrof_images = len(self.paths)
                nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / self.batch_size))
                print(nrof_images, self.batch_size, nrof_batches_per_epoch, embedding_size)
                self.emb_array = np.zeros((nrof_images, embedding_size))
                for i in range(nrof_batches_per_epoch):
                    start_index = i * self.batch_size
                    end_index = min((i + 1) * self.batch_size, nrof_images)
                    paths_batch = self.paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, self.image_size)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    self.emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
                print("shape", self.emb_array.shape)

                self.classifier_filename_exp = os.path.expanduser(self.classifier_filename)
                # Train classifier
                print('Training classifier')
                self.classify_model = SVC(kernel='linear', probability=True)
                self.classify_model.fit(self.emb_array, self.labels)

                # Create a list of class names
                self.class_names = [cls.name.replace('_', ' ') for cls in dataset]


    # predict()
    def predict(self,images_load,sess,needTransform=True,threshold=0.6):
        # Start time
        startDate = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        t1 = time.time()

        self.model_status='CLASSIFY'
        #print('images_load00000', type(images_load), images_load.shape)
        #Is need to align needTransform
        if needTransform:
            self.img_align = My_align_dataset_mtcnn(images_load,160,32,True,0.25).scaled
            images_load=self.img_align
        #load_model_need
        if self.need_load_model:
            self.need_load_model=My_facenet.load_model(self.model)
            # Get input and output tensors
            self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            self.embedding_size = self.embeddings.get_shape()[1]
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = 1
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / self.batch_size))
            # (nrof_images:1) (self.batch_size:90) (nrof_batches_per_epoch:1) (embedding_size:512)
            print(nrof_images, self.batch_size, nrof_batches_per_epoch, self.embedding_size)
            self.emb_array = np.zeros((nrof_images, self.embedding_size))
            self.start_index = 0 * self.batch_size
            self.end_index = min((0 + 1) * self.batch_size, nrof_images)
        print('images_load111', type(images_load))
        self.images = My_facenet.load_data01(images_load, False, False, self.image_size)
        #type(images),images.shape <class 'numpy.ndarray'> ( 160, 160, 3)--<format_data>-->>>(1, 160, 160, 3)
        #self.images = My_facenet.load_data00(self.img_align, False, False, self.image_size)
        feed_dict = {self.images_placeholder: self.images, self.phase_train_placeholder: False}
        #feed_dict: <tf.Tensor 'input:0' shape=<unknown> dtype=float32>: array([[[[ 1.50343006,  1.51953668,  1.55174992]
        self.emb_array[self.start_index:self.end_index, :] = sess.run(self.embeddings, feed_dict=feed_dict)

        # Classify images
        # print('Testing classifier')
        predictions = self.classify_model.predict_proba(self.emb_array)
        # print(self.class_names)
        # print(predictions[0])
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        #print
        #print("predictions:",predictions)
        print('best_class_probabilities',best_class_probabilities[0])
        print('best_class:',self.class_names[best_class_indices[0]])
        # End time
        endDate = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        t2=time.time()
        print('startDate:',startDate)
        print('endDate:', endDate)
        print('识别时间:', int((t2-t1)*1000),"ms")
        # set threshold
        threshold=0.02
        if(best_class_probabilities[0]>=threshold):#threshold
            return self.class_names[best_class_indices[0]]

    # split_dataset
    def split_dataset(self,dataset, min_nrof_images_per_class, nrof_train_images_per_class):
        train_set = []
        test_set = []
        for cls in dataset:
            paths = cls.image_paths
            # Remove classes with less than min_nrof_images_per_class
            if len(paths) >= min_nrof_images_per_class:
                np.random.shuffle(paths)
                train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
                test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
        return train_set, test_set
