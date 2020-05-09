"""Script for single-gpu/multi-gpu demo."""
import argparse
import os
import platform
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
import joblib
import json
import pandas as pd
import cv2
import math

from detector.apis import get_detector
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter

from TPU.ppose_nms_PMPUD import write_json

"""----------------------------- train or demo options -----------------------------"""

parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--cfg', type=str,  
                    default="configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml",
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, 
                    default="pretrained_models/fast_421_res152_256x192.pth",
                    help='checkpoint file name')
parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--indir', dest='inputpath',
                    help='image-directory', default="")
parser.add_argument('--list', dest='inputlist',
                    help='image-list', default="")
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="output")
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--detbatch', type=int, default=5,
                    help='detection batch size PER GPU')
parser.add_argument('--posebatch', type=int, default=80,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')
"""----------------------------- Video options -----------------------------"""
parser.add_argument('--video', dest='video',
                    help='video-name', default="")
parser.add_argument('--webcam', dest='webcam', type=int,
                    help='webcam number', default=-1)
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video', action='store_true', default=False)

parser.add_argument('--svc', type=str, 
                    default="pretrained_models/track_phone_usage_svc_1-0.joblib",
                    help='SVC Model file name for action detection')
parser.add_argument('--train', default=False, action='store_true',
                    help='prepare data for pose track training and disable prediction')

args = parser.parse_args()
cfg = update_config(args.cfg)
svc_file = args.svc

if platform.system() == 'Windows':
    args.sp = True

args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.detbatch = args.detbatch * len(args.gpus)
args.posebatch = args.posebatch * len(args.gpus)
args.tracking = (args.detector == 'tracker')

if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')


def check_input():
    # for wecam
    if args.webcam != -1:
        args.detbatch = 1
        return 'webcam', [int(args.webcam)]

    # for video
    if len(args.video):
        if os.path.isfile(args.video):
            videofile = args.video
            return 'video', [videofile]
        elif os.path.isdir(args.video):
            video_dir = args.video
            action_labels  = os.listdir(video_dir)
            video_files = []
            for idx, class_name in enumerate(action_labels):
                video_files.extend(get_video_file_list(video_dir, class_name)) 
            return 'video', video_files
        else:
            raise IOError('Error: --video must refer to a video file, not directory.')

    # for images
    if len(args.inputpath) or len(args.inputlist) or len(args.inputimg):
        inputpath = args.inputpath
        inputlist = args.inputlist
        inputimg = args.inputimg

        if len(inputlist):
            im_names = open(inputlist, 'r').readlines()
        elif len(inputpath) and inputpath != '/':
            for root, dirs, files in os.walk(inputpath):
                im_names = files
        elif len(inputimg):
            im_names = [inputimg]

        return 'image', [im_names]

    else:
        raise NotImplementedError

def get_video_file_list(root_dir, class_name):
    class_folders = os.path.join(root_dir, class_name)
    file_list = []
    for file_name in os.listdir(class_folders): 
        file_list.append(os.path.join(class_folders, file_name))

    return file_list


def print_finish_info():
    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')


def loop():
    n = 0
    while True:
        yield n
        n += 1

def load_keypoints(kp_data_file_name, pltThres=0.35):
    data = []
    with open(os.path.join(args.outputpath, kp_data_file_name), 'r') as json_file:
        data = json.load(json_file)
    kp_data = pd.DataFrame(data)

    return kp_data

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    ang = ang + 360 if ang < 0 else ang
    ang = 360 - ang if ang > 180 else ang 
    return ang

def getListOfArmAngles(ptLists):
    rArmAngle = []
    lArmAngle = []
    for ptList in ptLists:
        # measure right arm angle and left arm angle
        for i, (a0, a1, a2) in enumerate(links) :
            pt0     = np.int32([ptList[a0[0]], ptList[a0[1]]])
            pt1     = np.int32([ptList[a1[0]], ptList[a1[1]]])
            pt2     = np.int32([ptList[a2[0]], ptList[a2[1]]])                
            
            if i == 0:
                rArmAngle.append(getAngle(pt0, pt1, pt2))
            else:
                lArmAngle.append(getAngle(pt0, pt1, pt2))                
    return (rArmAngle, lArmAngle)
    
def drawSkeleton(image, kp_data, pltThres=0.35):
    
    for l in range(kp_data.shape[0]):
        if kp_data.score[l] <= pltThres:
            print('skipping due to kp_data.score being low')
            continue
        ptList = kp_data.keypoints[l]
        lbl_name = kp_data.lbl_pred[l] 
        
        armColor = [[200,100,100], [200,100,100]]
        
        if lbl_name == 'using phone' :
            color = [0, 0, 255]  # Red
            
            angIdx, lowerAngle = 0, 360
            for i, (a0, a1, a2) in enumerate(links) :
                pt0     = np.int32([ptList[a0[0]], ptList[a0[1]]])
                pt1     = np.int32([ptList[a1[0]], ptList[a1[1]]])
                pt2     = np.int32([ptList[a2[0]], ptList[a2[1]]])                
                ang = getAngle(pt0, pt1, pt2) 
                if ang < lowerAngle :
                    angIdx = i
                    lowerAngle = ang            
            armColor[angIdx] = [0, 0, 255]
        else :
            color = [255, 0, 0] # Blue
        
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        thickness = 2
        
        ((tw,th),tb) = cv2.getTextSize(lbl_name, fontFace, fontScale, thickness )
        txtx = np.int32(ptList[18] + (ptList[15]-ptList[18])/2 - tw/2)
        txty = np.int32(ptList[1] - (ptList[19]-ptList[1])) 
        cv2.putText(image, lbl_name, (txtx,txty), fontFace, fontScale, color, thickness)
        
        for i, (a0, a1, a2) in enumerate(links) :
            
            # if -1 in link:
            #     continue
            if min([a0[2], a1[2], a2[2]]) <= pltThres:
                print('skipping due to a0, a1, a2 score is lower')
                continue
            
            pt0     = np.int32([ptList[a0[0]], ptList[a0[1]]])
            pt1     = np.int32([ptList[a1[0]], ptList[a1[1]]])
            pt2     = np.int32([ptList[a2[0]], ptList[a2[1]]])
            
            cv2.line(image,
                     (pt0[0],pt0[1]),
                     (pt1[0],pt1[1]),
                     armColor[i],
                     3,
                     cv2.LINE_AA)
            cv2.line(image,
                     (pt1[0],pt1[1]),
                     (pt2[0],pt2[1]),
                     armColor[i],
                     3,
                     cv2.LINE_AA)            
            for pt in [pt0, pt1, pt2] :
                cv2.circle(image,
                           (pt0[0],pt0[1]),
                           5,
                           [255,255,255],
                           -1,
                           cv2.LINE_AA)
                cv2.circle(image,
                           (pt[0],pt[1]),
                           5,
                           colours[i+4],
                           1,
                           cv2.LINE_AA)
    return image

def visualize_action(kp_data):
    
    video = cv2.VideoCapture(kp_data['file_name'][0])
    
    fps = video.get(cv2.CAP_PROP_FPS)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))    
    
    fourcc = cv2.VideoWriter_fourcc(*"X264")
    action_output_file = os.path.splitext(os.path.basename(input_source))[0] + '_action.mp4' 
    print(f'action output file : {action_output_file}')
    writer = cv2.VideoWriter(action_output_file,  
                             fourcc, 
                             fps,
                             (W, H), 
                             True)    
    delay = int(1000 / fps)
    ok, img = video.read()
    if not ok:
        print('Cannot read video file')
        exit(1)
    
    kp_data[['frame_nbr', 'sfx']] = kp_data["image_id"].str.split('.', expand=True)
    kp_data.set_index('frame_nbr', inplace=True)
    
    cv2.imshow('Viewing Phone', img)
    print('Press ESC to start viewing window')
    while True:
        k = cv2.waitKey(delay) & 0xff
        if k == 27:
            break
    
    for i in range(length) :
        drawSkeleton(img, kp_data.loc[kp_data.index == str(i)])
        
        writer.write(img)
        cv2.imshow('Viewing Phone', img)
        k = cv2.waitKey(delay) & 0xff
        if k == 27:
            break
        ok, img = video.read()
    
    print('Press ESC to exit viewing window')
    while True:
        k = cv2.waitKey(delay) & 0xff
        if k == 27:
            break
    writer.release()
    video.release()
    cv2.destroyAllWindows()
    print('action video writter released for file : ', action_output_file)

#------------------------------------------------------------------------------
if __name__ == "__main__":
    mode, input_sources = check_input()

    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    links = [[(18, 19, 20), (24, 25, 26), (30, 31, 32)],  # Right arm
             [(15, 16, 17), (21, 22, 23), (27, 28, 29)]]  # Left arm
    p_colors = [[0,100,255], [0,255,0]]  
    colours     = [[0,100,255],
                   [0,100,255],
                   [0,255,255],
                   [0,100,255],
                   [0,255,255],
                   [0,100,255],
                   [0,255,0],
                   [255,200,100],
                   [255,0,255],
                   [0,255,0],
                   [255,200,100],
                   [255,0,255],
                   [0,0,255],
                   [255,0,0],
                   [200,200,0],
                   [255,0,0],
                   [200,200,0],
                   [0,0,0]] 

    # Load pose model
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print(f'Loading pose model from {args.checkpoint}...')
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    if len(args.gpus) > 1:
        pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)
    else:
        pose_model.to(args.device)
    pose_model.eval()
    
    # Load action detection model if exists
    if os.path.isfile(svc_file):
        action_model = joblib.load(svc_file)
    else :
        action_model = None

    for input_source in input_sources :

        if mode == 'video' and len(input_sources) > 1 :
            lbl_name = input_source.split(os.sep)[-2]
        else:
            lbl_name = 'not_categorized'
            
        ### Moved from before load pose model
        # Load detection loader
        if mode == 'webcam':
            det_loader = WebCamDetectionLoader(input_source, get_detector(args), cfg, args).start()
        else:
            det_loader = DetectionLoader(input_source, get_detector(args), cfg, args, batchSize=args.detbatch, mode=mode).start()
    
        runtime_profile = {
            'dt': [],
            'pt': [],
            'pn': []
        }
    
        # Init data writer
        queueSize = 2 if mode == 'webcam' else args.qsize
        if args.save_video and mode != 'image':
            from alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt
            if mode == 'video':
                video_save_opt['savepath'] = os.path.join(args.outputpath, 'AlphaPose_' + os.path.basename(input_source))
            else:
                video_save_opt['savepath'] = os.path.join(args.outputpath, 'AlphaPose_webcam' + str(input_source) + '.mp4')
            video_save_opt.update(det_loader.videoinfo)
            writer = DataWriter(cfg, args, save_video=True, video_save_opt=video_save_opt, queueSize=queueSize).start()
        else:
            writer = DataWriter(cfg, args, save_video=False, queueSize=queueSize).start()
    
        if mode == 'webcam':
            print('Starting webcam demo, press Ctrl + C to terminate...')
            sys.stdout.flush()
            im_names_desc = tqdm(loop())
        else:
            data_len = det_loader.length
            im_names_desc = tqdm(range(data_len), dynamic_ncols=True)
    
        batchSize = args.posebatch
        if args.flip:
            batchSize = int(batchSize / 2)
        try:
            for i in im_names_desc:
                start_time = getTime()
                with torch.no_grad():
                    (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                    if orig_img is None:
                        break
                    if boxes is None or boxes.nelement() == 0:
                        writer.save(None, None, None, None, None, orig_img, os.path.basename(im_name))
                        continue
                    if args.profile:
                        ckpt_time, det_time = getTime(start_time)
                        runtime_profile['dt'].append(det_time)
                    # Pose Estimation
                    inps = inps.to(args.device)
                    datalen = inps.size(0)
                    leftover = 0
                    if (datalen) % batchSize:
                        leftover = 1
                    num_batches = datalen // batchSize + leftover
                    hm = []
                    for j in range(num_batches):
                        inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                        if args.flip:
                            inps_j = torch.cat((inps_j, flip(inps_j)))
                        hm_j = pose_model(inps_j)
                        if args.flip:
                            hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], det_loader.joint_pairs, shift=True)
                            hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                        hm.append(hm_j)
                    hm = torch.cat(hm)
                    if args.profile:
                        ckpt_time, pose_time = getTime(ckpt_time)
                        runtime_profile['pt'].append(pose_time)
                    hm = hm.cpu()
                    writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, os.path.basename(im_name))
    
                    if args.profile:
                        ckpt_time, post_time = getTime(ckpt_time)
                        runtime_profile['pn'].append(post_time)
    
                if args.profile:
                    # TQDM
                    im_names_desc.set_description(
                        'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                            dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
                    )
            print_finish_info()
            while(writer.running()):
                time.sleep(1)
                print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...')
            writer.stop()
            det_loader.stop()
        except KeyboardInterrupt:
            print_finish_info()
            # Thread won't be killed when press Ctrl+C
            if args.sp:
                det_loader.terminate()
                while(writer.running()):
                    time.sleep(1)
                    print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...')
                writer.stop()
            else:
                # subprocesses are killed, manually clear queues
                writer.commit()
                writer.clear_queues()
                # det_loader.clear_queues()
        final_result = writer.results()
        
        kp_data_file_name = os.path.splitext(os.path.basename(input_source))[0] + '.json'
        write_json(final_result, args.outputpath, form=args.format, for_eval=args.eval, 
                   lbl_name=lbl_name, input_file_name=input_source, 
                   output_file_name=kp_data_file_name)
        print(f"Results have been written to json for {input_source}") 
        
        if args.train :
            continue
        
        # Predict action and visualize 
        if action_model is not None :
            kp_data = load_keypoints(kp_data_file_name)
                
            kp_X = pd.DataFrame(kp_data['keypoints'].tolist())
            kp_X['score'] = kp_data['score']
            rArmAngle, lArmAngle = getListOfArmAngles(kp_data['keypoints'])
            kp_X['rArmAngle'] = rArmAngle
            kp_X['lArmAngle'] = lArmAngle
            kp_data['lbl_pred'] = action_model.predict(kp_X)
            
            del(kp_X)
            
            visualize_action(kp_data)
        
        
