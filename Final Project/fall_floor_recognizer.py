import argparse
import logging
import time
import lineTool

import tensorflow as tf
import keras

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from keras.models import Model, load_model
from keras.backend.tensorflow_backend import set_session

'''
#跑GPU版本的tensorflow時可能會因為獨顯內存不夠而跑不起來,比如說同時開了遊戲或TeamView等,所以以下code是先分配好獨顯內存避免不夠用
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.6
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config)) 
'''

logger = logging.getLogger('TfPoseEstimator-Recognizer')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
#Line Notify
token = "Z7CdEtLweyMQSTnGcm1CBswCAMvdYfHr6xIYiuv4dEI"

fps_time = 0
#OPENCV旋轉圖片
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated
#判定是否求救
def whether_to_notify(state_history, frame_idx):
    safe_cnt = 0
    dangerous_cnt = 0
    for i in range(frame_idx-20, frame_idx):
        if state_history[i] == 0 and i > frame_idx-10:
            safe_cnt += 1
        elif state_history[i] == 1:
            dangerous_cnt += 1
    #最新10次結果中有超過6次以上不屬於安全狀態就要發送Line
    if safe_cnt > 5:
        return False
    if dangerous_cnt > 13:
        return True
    return False 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    #載入模型
    #lstm
    model = load_model('fall_floor_lstm_v7.h5')
    #openpose
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    #載入影像
    cap = None
    if len(args.video) > 0:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(0)

    frame_cnt = 0
    frame_idx = 0
    #最多就取32幀
    action_history = []
    #安全狀態紀錄
    state_history = []
    #使用XVID編碼
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #寫檔
    out = cv2.VideoWriter('result.avi', fourcc, 20.0, (1024, 576))

    if cap.isOpened() is False:
        print("Error opening video stream or file")
    else:
        cap.set(cv2.CAP_PROP_POS_MSEC, (frame_cnt*85))#可擷取到32個frames
        ret_val, img = cap.read()
        
        image = rotate(img, 90)

        while ret_val:
            if frame_idx > 31:
                #檢查是否要發出Line求救
                if frame_idx > 51:
                    if whether_to_notify(state_history, frame_idx):
                        lineTool.lineNotify(token, "老人家跌倒了")
                        state_history.clear()
                        state_history = [0]*frame_idx
                else:
                    state_history.append(0)
                #載入最新32幀的骨架
                action_predict = np.zeros([1, 32, 36])
                start_idx = frame_idx - 32
                for i in range(32):
                    action_predict[0, i, :] = action_history[start_idx+i]
                #預測
                result = model.predict(action_predict)
                print('Predict:', result)
                c = np.argmax(result)
                print('Action :', c)
                if c < 2:
                    print('State: Dangerous')
                    state_history.append(1)
                    cv2.putText(image, "Dangerous", (10, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)#BGR
                else:
                    print('State: Safe')
                    state_history.append(0)
                    cv2.putText(image, "Safe", (10, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                frame_cnt = 0
            else:
                state_history.append(0)
            #骨架偵測
            logger.debug('image process+')
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            #若有人就將偵測到骨架做紀錄
            action = np.zeros([36,])
            if humans:
                resize_img = cv2.resize(image, (432, 368))
                width, height, channel = resize_img.shape
                for part_idx in humans[0].body_parts:
                    action[part_idx*2] = humans[0].body_parts[part_idx].x*width + 0.5
                    action[part_idx*2+1] = humans[0].body_parts[part_idx].y*height + 0.5
            #若沒有人則要紀錄全部關節點位置為0
            action_history.append(action)
            #畫出骨架
            logger.debug('postprocess+')
            #image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            #顯示畫面和FPS
            logger.debug('show+')   
            cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', image)
            fps_time = time.time()
            #寫檔
            out.write(image) 
            if cv2.waitKey(1) == 27:
                break

            ret_val, img = cap.read()
            image = rotate(img, 90)
            frame_cnt += 1
            frame_idx += 1
    
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    logger.debug('finished+')