import base64
import re
import traceback
import cv2
import os
import datetime
import io
import numpy as np
import tensorflow as tf

import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
from tornado.options import define, options
from PIL import Image

from  MyClassifier import ImageClassifier

options.define("sess", default=None, type=object)
define("port", default=8080, help="run on the given port", type=int)


# When capturing a photo in real time through the camera, deal with the needs of face recognition, identify which user or stranger
# a base64 string of a photo,
class PredictHandler(tornado.web.RequestHandler):
    def post(self):
        # try:
        if TrainHandler.model == None:
            if TrainHandler.isTraining:
                rs = {'success': False, 'msg': '请等待模型训练完成'}
            else:
                TrainHandler.model = ImageClassifier()
        if TrainHandler.model != None:
            snapData = self.get_argument('snapData')
            if snapData == '':
                rs = {'success': False, 'msg': '请上传文件'}
            else:
                image_data = []
                base64_data = re.sub('^data:image/.+;base64,', '', snapData)
                byte_data = base64.b64decode(base64_data)
                img_data = io.BytesIO(byte_data)
                img = Image.open(img_data)
                # img.save("c://tmp/img.png");
                img = np.array(img, 'f')
                image_data.append(img)
                # image_data = np.array(image_data[0])
                # print('image_data::::',type(image_data),image_data.shape)
                # print('image_data', len(image_data))

                # face recognition
                rs = {'success': True,
                      'class': TrainHandler.model.predict(image_data, options.sess, needTransform=False, threshold=0.4)}
        # except tornado.web.MissingArgumentError as err:
        #    rs = {'success':False,'msg':"{0}".format(err)}
        # except Exception as err:
        #    rs = {'success':False,'msg':"{0}".format(err)}

        self.write(rs)


# Handling requests for training models
class TrainHandler(tornado.web.RequestHandler):
    # Initialize the model to None
    model=None
    # Is the model being trained
    isTraining=False
    # Training start time
    startDate = ''
    # Training end time
    endDate = ''
    # Asynchronous processing (multithreading)
    executor = ThreadPoolExecutor(2)
    @run_on_executor
    def get(self):
        if TrainHandler.isTraining:
            self.rs = {'success':False,'msg':'1请等待模型训练完成'}
        else:
            try:
                TrainHandler.isTraining = True
                #启动fit过程
                TrainHandler.startDate = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                TrainHandler.model=ImageClassifier()
                TrainHandler.model.fit()
                TrainHandler.endDate = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                TrainHandler.isTraining = False
                self.rs = {'success':True}
            except tornado.web.MissingArgumentError as err:
                self.rs = {'success':False,'msg':"{0}".format(err)}
                TrainHandler.isTraining = False
            except Exception as err:
                traceback.print_exc()
                self.rs = {'success':False,'msg':"{0}".format(err)}
                TrainHandler.isTraining = False
            
        self._callback()
    
    def _callback(self):  
        self.write(self.rs)


# When uploading a photo locally, deal with the needs of face recognition, identify which user or stranger
class ClassifierHandler(tornado.web.RequestHandler):
    def get(self):
        self.post()

    def post(self):
        #try:
        if TrainHandler.model==None:
            if TrainHandler.isTraining:
                rs = {'success':False,'msg':'请等待模型训练完成'}
            else:
                TrainHandler.model=ImageClassifier()
                rs = {'success': False, 'msg': '请先进行模型训练'}
        if TrainHandler.model!=None:
            file_metas = self.request.files.get('testfile', None)
            if not file_metas:
                rs = {'success':False,'msg':'Please upload file!'}
            else:
                print("文件数",len(file_metas))
                images_load=[]
                #file_metas <class 'list'>
                for meta in file_metas:
                    image_array = np.frombuffer(meta['body'], dtype=np.uint8)  # numpy array
                    img_brg = cv2.imdecode(image_array, 1)  # 效果等同于cv2.imread()
                    b, g, r = cv2.split(img_brg)
                    img_rgb = cv2.merge([r, g, b])
                    images_load.append(img_rgb)

                #recognition
                rs = {'success':True,'class':TrainHandler.model.predict(images_load,options.sess,needTransform=True,threshold=0.4)}
                # try:
                #     rs = {'success':True,'class':TrainHandler.model.predict(images_load,options.sess,needTransform=True,threshold=0.6)}
                # except Exception :
                #     rs = {'success': False, 'msg': '请先进行模型训练'}
        '''
        except tornado.web.MissingArgumentError as err:
            rs = {'success':False,'msg1:"{0}".format(err)}
        except Exception as err:
            rs = {'success':False,'msg':"{0}".format(err)}
        '''
        self.write(rs)


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html',startDate=TrainHandler.startDate,endDate=TrainHandler.endDate)


class FacedetectHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('facedetect.html', startDate=TrainHandler.startDate, endDate=TrainHandler.endDate)


class YuyinHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('yuyin.html', startDate=TrainHandler.startDate, endDate=TrainHandler.endDate)


if __name__ == '__main__':
    tornado.options.parse_command_line()
    with tf.Graph().as_default():
        with tf.Session() as sess:
            options.sess = sess
            app = tornado.web.Application(
                handlers=[#(r'/', IndexHandler),
                          (r'/', FacedetectHandler),
                          (r'/predict', PredictHandler),
                          (r'/classifier.do', ClassifierHandler),
                          (r'/train.do', TrainHandler),
                          (r'/index.html', IndexHandler),
                          (r'/yuyin.html', YuyinHandler)],
                # Template files are placed in the templates directory
                template_path=os.path.join(os.path.dirname(__file__), "templates"),
                # Static files are placed in the static directory
                static_path=os.path.join(os.path.dirname(__file__), "static"),
                debug=True
            )
            http_server = tornado.httpserver.HTTPServer(app)
            http_server.listen(options.port)
            print('system started')
            tornado.ioloop.IOLoop.instance().start()
