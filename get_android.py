import android
import cv2
import os
from AidLux import Aashmem
import json
import sys
import numpy as np
import time
import ctypes
import sys
import multiprocessing
import aidlite_gpu
from multiprocessing import Process, sharedctypes
from ssd_mobilenet_utils import preprocess_image_for_tflite_uint8, non_max_suppression, draw_ssd_result
from utils_movenet import draw_result, movenetDecode
configs, kv_shows = [], []
shareArrays, boxesArrays, boxesArrays, = [], [], []
showW=0
showH=0
numChan=1
pools = []



def androidOb(path,type,width,height,num):
    global showW,showH,numChan
    showW=width
    showH=height
    numChan=num
    print("showW",showW,"showH",showH,"numChan",numChan)
    droid = android.Android()
    jsonStr=""
    ret=""
    pid=os.getpid()
    print("pid",pid)
    with open(path,'r', encoding='utf-8') as js:
        jsonStr=js.read()
        ret = droid.stream(jsonStr,type,pid)
        result=json.loads(ret.result)
        print(result)
        if result["ret"]!=0:
            print("msg",ret.result["msg"])
            sys.exit(0)    
    print(jsonStr)
    print(ret)

def run(funC):
    global pools
    print(showW,showH,numChan)
    for i in range(numChan):
        shareArrays.append(sharedctypes.RawArray(ctypes.c_uint8, showW*showH*3))
        if funC==ai_worker1:
            boxesArrays.append(sharedctypes.RawArray(ctypes.c_float, 10*5))
        elif funC==ai_worker2:
            boxesArrays.append(sharedctypes.RawArray(ctypes.c_int16, 17*2))
        configs.append("/tmp/mmkv/tmp_ipc_rtsp"+str(i))
        kv_shows.append(Aashmem("/tmp/mmkv/tmp_ipc_display_" + str(i)))
        pools.append(Process(target = input_worker, args = (i,funC,)))
    pools.append(Process(target = funC,args=(0,)))
    time.sleep(2)
        # 开始运行
    for p in pools:
        p.start()

# get 16 rtsp streamer to sharememory and show the streamer
def input_worker(video_id,funC):
    res = configs[video_id]
    input_men = np.frombuffer(shareArrays[video_id], dtype=ctypes.c_uint8).reshape(showH, showW, 3)
    if funC==ai_worker1:
        out_men = np.frombuffer(boxesArrays[video_id], dtype=ctypes.c_float).reshape(10, 5)
    elif funC==ai_worker2:
        out_men = np.frombuffer(boxesArrays[video_id], dtype=ctypes.c_int16).reshape(17, 2)
    kv = Aashmem(res) 
    print(res)
    num = 0
    l = int.from_bytes(kv.get_bytes(4, 0), byteorder='little')
    while l==0:
        l = int.from_bytes(kv.get_bytes(4, 0), byteorder='little')
        time.sleep(0.005)
    while True:
        trig = int.from_bytes(kv.get_bytes(4, 4), byteorder='little')
        if trig==0:
            time.sleep(0.001)
            continue
        bt = kv.get_bytes(l, 8)
        trig=0
        kv.set_bytes(trig.to_bytes(4, byteorder='little', signed=True), 4, 4)
        img = np.frombuffer(bt, dtype=np.uint8).reshape(showH, showW, 3)
        input_men[:] = img
        time.sleep(0.001)
        if funC==ai_worker1:
            img = draw_ssd_result(img, out_men)
        elif funC==ai_worker2:
            img = draw_result(img, out_men)
        binput = img.tobytes()
        print(len(binput))
        kv_shows[video_id].set_bytes(len(binput).to_bytes(4, byteorder='little', signed=True), 4, 0)
        kv_shows[video_id].set_bytes(binput, len(binput), 4)

# detect per frame of rtsp streamer...
def ai_worker1(ind):
    aidlite = aidlite_gpu.aidlite(ind)
    aidlite.cpu_pin([0, 1, 2, 3])
    inShape =[1*300*300*3,]
    outShape= [1*10*4*4, 1*10*4, 1*10*4,1*4]
    model_path="ssdlite_mobilenet_v3.tflite"
    aidlite.FAST_ANNModel(model_path,inShape,outShape,4,0)
    frame_vector = []
    out_vector = []
    for video_id in range(ind,numChan):
        frame_vector.append(np.frombuffer(shareArrays[video_id], dtype=ctypes.c_uint8).reshape(showH, showW, 3))
        out_vector.append(np.frombuffer(boxesArrays[video_id], dtype=ctypes.c_float).reshape(10, 5))
    time.sleep(3)

    while True:
        for frame, out in zip(frame_vector, out_vector):
            image_data = preprocess_image_for_tflite_uint8(frame, model_image_size=300)
            aidlite.setTensor_Int8(image_data, 300, 300)
            aidlite.invoke()
            boxes = aidlite.getTensor_Fp32(0)
            classes = aidlite.getTensor_Fp32(1)
            scores = aidlite.getTensor_Fp32(2)
            box=boxes.reshape((10,4))
            box, scores, classes = np.squeeze(box), np.squeeze(scores), np.squeeze(classes).astype(np.int32)
            out_scores, out_boxes, out_classes = non_max_suppression(scores, box, classes)
            out[:, :4] = out_boxes
            out[:, 4] = out_classes
        time.sleep(0.05)

# detect per frame of rtsp streamer...
def ai_worker2(ind):
    '''
    MoveNet人体关键点检测算法
    该线程只负责从帧共享内存中拿数据进行检测,随后放入结果共享内存中
    :param ind: 检测线程id号,该id号同时指示了分配的共享内存id号
    :return None
    '''
    aidlite = aidlite_gpu.aidlite(ind)
    aidlite.cpu_pin([0, 1, 2, 3])
    inShape =[1*192*192*3,]
    outShape= [1 * 48 * 48 * 1 * 34, 1 * 48 * 48 * 1 * 1, 1 * 48 * 48 * 34 * 1, 1 * 48 * 48 * 17 * 1]
    model_path="movenet_int8_4out.tflite"
    aidlite.FAST_ANNModel(model_path,inShape,outShape,4,0)
    frame_vector = []
    out_vector = []
    for video_id in range(ind,numChan):
        frame_vector.append(np.frombuffer(shareArrays[video_id], dtype=ctypes.c_uint8).reshape(showH, showW, 3))
        out_vector.append(np.frombuffer(boxesArrays[video_id], dtype=ctypes.c_int16).reshape(17, 2))
    time.sleep(3)

    while True:
        for frame, out in zip(frame_vector, out_vector):
            image_data = cv2.resize(frame, (192, 192))
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB).astype(np.uint8)
            aidlite.setInput_Int8(image_data, 192, 192)
            aidlite.invoke()
            output1 = aidlite.getOutput_Int8(3).reshape(1, 48, 48, 17).astype(np.float32)
            output2 = aidlite.getOutput_Int8(1).reshape(1, 48, 48, 1).astype(np.float32)
            output3 = aidlite.getOutput_Int8(0).reshape(1, 48, 48, 34).astype(np.float32)
            output4 = aidlite.getOutput_Int8(2).reshape(1, 48, 48, 34).astype(np.float32)
            output1 = output1 * 0.00390625
            output2 = output2 * 0.00390625
            output3 = (output3 - 121) * 0.20757873356342316
            output4 = (output4 - 45) * 0.0073025282472372055
            pre = movenetDecode([output1.transpose(0, 3, 1, 2), output2.transpose(0, 3, 1, 2), 
                                    output3.transpose(0, 3, 1, 2), output4.transpose(0, 3, 1, 2)], mode="output")
            pre = pre.reshape(17, 2)
            pre[:, 1] = pre[:, 1] * showH
            pre[:, 0] = pre[:, 0] * showW
            pre = pre.astype(np.int16)
            out[:] = pre.astype(np.int16)
        time.sleep(0.05)