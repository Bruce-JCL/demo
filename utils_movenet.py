
import numpy as np
import cv2
import os 

current_dir = os.path.dirname(os.path.abspath(__file__))
_center_weight = np.load(f'{current_dir}/center_weight_origin.npy').reshape(48 ,48)
_range_weight_x = np.array([[x for x in range(48)] for _ in range(48)])
_range_weight_y = _range_weight_x.T

def draw_result(image ,keypoints):
    connect_list = [
        [0, 1, (255, 0, 0)],  # nose → left eye
        [0, 2, (0, 0, 255)],  # nose → right eye
        [1, 3, (255, 0, 0)],  # left eye → left ear
        [2, 4, (0, 0, 255)],  # right eye → right ear
        [0, 5, (255, 0, 0)],  # nose → left shoulder
        [0, 6, (0, 0, 255)],  # nose → right shoulder
        [5, 6, (0, 255, 0)],  # left shoulder → right shoulder
        [5, 7, (255, 0, 0)],  # left shoulder → left elbow
        [7, 9, (255, 0, 0)],  # left elbow → left wrist
        [6, 8, (0, 0, 255)],  # right shoulder → right elbow
        [8, 10, (0, 0, 255)],  # right elbow → right wrist
        [11, 12, (0, 255, 0)],  # left hip → right hip
        [5, 11, (255, 0, 0)],  # left shoulder → left hip
        [11, 13, (255, 0, 0)],  # left hip → left knee
        [13, 15, (255, 0, 0)],  # left knee → left ankle
        [6, 12, (0, 0, 255)],  # right shoulder → right hip
        [12, 14, (0, 0, 255)],  # right hip → right knee
        [14, 16, (0, 0, 255)],  # right knee → right ankle
    ]
    for (index01, index02, color) in connect_list:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        if point01[0 ]>=0 and point02[0]>=0:
            cv2.line(image, point01, point02, color, 1)

    for i in range(17):
        # x = int(point[i][0])
        # y = int(point[i][1])
        cv2.circle(image, keypoints[i], 3, (255 ,0 ,0), -1)
    return image




def maxPoint(heatmap, center=True):

    if len(heatmap.shape) == 3:
        batch_size ,h ,w = heatmap.shape
        c = 1

    elif len(heatmap.shape) == 4:
        # n,c,h,w
        batch_size ,c ,h ,w = heatmap.shape
        # print(heatmap.shape)

    if center:
        # print(heatmap.shape)
        # print(heatmap[0][27:31,27:31])
        # print(_center_weight[27:31,27:31])
        heatmap = heatmap *_center_weight  # 加权取最靠近中间的
        # print(heatmap[0][27:31,27:31])


    heatmap = heatmap.reshape((batch_size ,c, -1))  # 64,c, 48x48
    # print(heatmap.shape)
    # print(heatmap[0,0,23*48+20:23*48+30])
    max_id = np.argmax(heatmap ,2  )  # 64,c, 1
    # print(max_id)
    # print(max_id, heatmap[0][0][max_id])
    # print(np.max(heatmap))
    y = max_id//w
    x = max_id %w
    # bv
    return x ,y


def maxPoint(heatmap, center=True):

    if len(heatmap.shape) == 3:
        batch_size ,h ,w = heatmap.shape
        c = 1

    elif len(heatmap.shape) == 4:
        # n,c,h,w
        batch_size ,c ,h ,w = heatmap.shape
        # print(heatmap.shape)

    if center:
        # print(heatmap.shape)
        # print(heatmap[0][27:31,27:31])
        # print(_center_weight[27:31,27:31])
        heatmap = heatmap *_center_weight  # 加权取最靠近中间的
        # print(heatmap[0][27:31,27:31])


    heatmap = heatmap.reshape((batch_size ,c, -1))  # 64,c, 48x48
    # print(heatmap.shape)
    # print(heatmap[0,0,23*48+20:23*48+30])
    max_id = np.argmax(heatmap ,2  )  # 64,c, 1
    # print(max_id)
    # print(max_id, heatmap[0][0][max_id])
    # print(np.max(heatmap))
    y = max_id//w
    x = max_id %w
    # bv
    return x ,y


def movenetDecode(data, kps_mask=None ,mode='output', num_joints = 17,
                  img_size=192, hm_th=0.3):
    ##data [64, 7, 48, 48] [64, 1, 48, 48] [64, 14, 48, 48] [64, 14, 48, 48]
    # kps_mask [n, 7]


    if mode == 'output':
        batch_size = data[0].shape[0]

        heatmaps = data[0].copy()

        heatmaps[heatmaps < hm_th] = 0

        centers = data[1].copy()

        regs = data[2].copy()
        offsets = data[3].copy()


        cx ,cy = maxPoint(centers)
        # cx,cy = extract_keypoints(centers[0])
        # print("movenetDecode 119 cx,cy: ",cx,cy)

        dim0 = np.arange(batch_size ,dtype=np.int32).reshape(batch_size ,1)
        dim1 = np.zeros((batch_size ,1) ,dtype=np.int32)

        res = []
        for n in range(num_joints):
            # nchw!!!!!!!!!!!!!!!!!

            reg_x_origin = (regs[dim0 ,dim1 + n *2 ,cy ,cx ] +0.5).astype(np.int32)
            reg_y_origin = (regs[dim0 ,dim1 + n * 2 +1 ,cy ,cx ] +0.5).astype(np.int32)
            # print(reg_x_origin,reg_y_origin)
            reg_x = reg_x_origin +cx
            reg_y = reg_y_origin +cy
            # print(reg_x, reg_y)

            ### for post process
            reg_x = np.reshape(reg_x, (reg_x.shape[0] ,1 ,1))
            reg_y = np.reshape(reg_y, (reg_y.shape[0] ,1 ,1))
            # print(reg_x.shape,reg_x,reg_y)
            reg_x = reg_x.repeat(48 ,1).repeat(48 ,2)
            reg_y = reg_y.repeat(48 ,1).repeat(48 ,2)
            # print(reg_x.repeat(48,1).repeat(48,2).shape)
            # bb


            #### 根据center得到关键点回归位置，然后加权heatmap
            range_weight_x = np.reshape(_range_weight_x ,(1 ,48 ,48)).repeat(reg_x.shape[0] ,0)
            range_weight_y = np.reshape(_range_weight_y ,(1 ,48 ,48)).repeat(reg_x.shape[0] ,0)
            tmp_reg_x = (range_weight_x -reg_x)**2
            tmp_reg_y = (range_weight_y -reg_y )**2
            # print(tmp_reg_x.shape, _range_weight_x.shape, reg_x.shape)
            tmp_reg = (tmp_reg_x +tmp_reg_y)**0.5 +1.8  # origin 1.8
            # print(tmp_reg.shape,heatmaps[:,n,...].shape)(1, 48, 48)
            # print(heatmaps[:,n,...][0][19:25,19:25])
            # cv2.imwrite("t.jpg",heatmaps[:,n,...][0]*255)
            # print(tmp_reg[0][19:25,19:25])
            tmp_reg = heatmaps[: ,n ,... ] /tmp_reg
            # print(tmp_reg[0][19:25,19:25])



            # reg_cx = max(0,min(47,reg_x[0][0][0]))
            # reg_cy = max(0,min(47,reg_y[0][0][0]))
            # _reg_weight_part = _reg_weight[49-reg_cy:49-reg_cy+48, 49-reg_cx:49-reg_cx+48]
            # if _reg_weight_part.shape[0]!=48 or _reg_weight_part.shape[1]!=48:
            #     print(_reg_weight_part.shape)
            #     print(reg_cy,reg_cx)
            #     bbb
            # # print(_reg_weight_part[reg_cy,reg_cx])
            # #keep reg_cx reg_cy to 1
            # tmp_reg = heatmaps[:,n,...]*_reg_weight_part

            # b


            # if n==1:
            #     cv2.imwrite('output/predict/t3.jpg', cv2.resize(tmp_reg[0]*2550,(192,192)))
            tmp_reg = tmp_reg[: ,np.newaxis ,: ,:]
            reg_x ,reg_y = maxPoint(tmp_reg, center=False)

            # # print(reg_x, reg_y)
            reg_x[reg_x >47] = 47
            reg_x[reg_x <0] = 0
            reg_y[reg_y >47] = 47
            reg_y[reg_y <0] = 0

            score = heatmaps[dim0 ,dim1 +n ,reg_y ,reg_x]
            # print(score)
            offset_x = offsets[dim0 ,dim1 + n *2 ,reg_y ,reg_x]  # *img_size//4
            offset_y = offsets[dim0 ,dim1 + n * 2 +1 ,reg_y ,reg_x ]  # *img_size//4
            # print(offset_x,offset_y)
            res_x = (reg_x +offset_x) /(img_size//4)
            res_y = (reg_y +offset_y) /(img_size//4)
            # print(res_x,res_y)

            res_x[score <hm_th] = -1
            res_y[score <hm_th] = -1


            res.extend([res_x, res_y])
            # b

        res = np.concatenate(res ,axis=1)  # bs*14
    return res
