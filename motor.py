import argparse

import numpy as np

import cv2

import time

from PIL import Image

from time import sleep

from edgetpu.basic import edgetpu_utils

from pose_engine import PoseEngine

#from mo import MC

 

lastresults = None

processes = []

frameBuffer = None

results = None

fps = ""

detectfps = ""

framecount = 0

detectframecount = 0

time1 = 0

time2 = 0

angle_deg = ""

l_angle = 0

r_angle = 0

 

EDGES = (

#    ('nose', 'left eye'),

#    ('nose', 'right eye'),

#    ('nose', 'left ear'),

#    ('nose', 'right ear'),

#    ('left ear', 'left eye'),

#    ('right ear', 'right eye'),

#   ('left eye', 'right eye'),

#    ('left shoulder', 'right shoulder'),

    ('left shoulder', 'left elbow'), # 5,7

#    ('left shoulder', 'left hip'),

    ('right shoulder', 'right elbow'), # 6,8

#    ('right shoulder', 'right hip'),

    ('left elbow', 'left wrist'), # 7,9

    ('right elbow', 'right wrist'), #8,10

#    ('left hip', 'right hip'),

#    ('left hip', 'left knee'),

#    ('right hip', 'rght knee'),

#    ('left knee', 'left ankle'),

#    ('right knee', 'right ankle'),

)

 

 

def draw_pose(img, pose, threshold=0.2): # 선 그리는 코드 

        xys = {}

        for label, keypoint in pose.keypoints.items():

            if keypoint.score < threshold: continue

            xys[label] = (int(keypoint.yx[1]), int(keypoint.yx[0]))

            img = cv2.circle(img, (int(keypoint.yx[1]), int(keypoint.yx[0])), 5, (0, 255, 0), -1) # 관절 만드는 포인

            #l_pt1 = xys['nose']

            #print(xys.keys()) # dictionary가 가지고 있는 키를 알려줌.key values 도 있음.

            avg_angle1(xys)

            avg_angle2(xys)

 

        for a, b in EDGES:

            if a not in xys or b not in xys: continue

            ax, ay = xys[a]

            bx, by = xys[b]

 

            img = cv2.line(img, (ax, ay), (bx, by), (0, 255, 255), 2)

 

 

def angle1(topPoint, midPoint, botPoint): # 오른팔 각도 짜는 부분

 

    topLineSlope = (midPoint[1] - topPoint[1]) / (midPoint[0] - topPoint[0]) #기울기1 구하기

    botLineSlope = (midPoint[1] - botPoint[1]) / (midPoint[0] - botPoint[0]) #기울기2 구하기

    top_actan_angle = np.arctan(topLineSlope)

    top_angle_deg = np.degrees(top_actan_angle)

    if top_angle_deg < 0:

      top_angle_res = 180+top_angle_deg

    else:

      top_angle_res = top_angle_deg

    return top_angle_res

 

    bot_actan_angle = np.arctan(botLineSlope)

    bot_angle_deg = np.degrees(bot_actan_angle)

 

    angle_res = top_angle_res+bot_angle_res

 

    return angle_result

  

def angle2(topPoint_y, midPoint_y, botPoint_y): # 왼팔 각도 짜는 부분

 

    topLineSlope_y = (midPoint_y[1] - topPoint_y[1]) / (midPoint_y[0] - topPoint_y[0]) #기울기1 구하기

    botLineSlope_y = (midPoint_y[1] - botPoint_y[1]) / (midPoint_y[0] - botPoint_y[0]) #기울기2 구하기

    top_actan_angle_y = np.arctan(topLineSlope_y)

    top_angle_deg_y = np.degrees(top_actan_angle_y)

    #print(top_angle_deg_y)

    if top_angle_deg_y < 0:

      top_angle_res_y = -top_angle_deg_y

    else:

      top_angle_res_y = 180-top_angle_deg_y

    #print(top_angle_res_y)

    

    bot_actan_angle_y = np.arctan(botLineSlope_y)

    bot_angle_deg_y = np.degrees(bot_actan_angle_y)

    #print(bot_angle_deg_y) 

 

    return top_angle_res_y

 

    angle_res_y = top_angle_res+bot_angle_res_y

   

    return angle_result_y

 

def avg_angle1(xys): # 거울모드로 반전시켰기 때문 : 유저 오른손 -> 화면 왼손으로 인식.

    try:

            global r_angle

            r_pt1 = xys['right shoulder']

            r_pt2 = xys['right elbow']

            r_pt3 = xys['right wrist']

            r_angle = angle2(r_pt3, r_pt2, r_pt1)

            #print ("왼팔: % .4f" %(r_angle))

          

    except:

 

         pass

 

def avg_angle2(xys): 

    try:

            global l_angle

            l_pt1 = xys['left shoulder']

            l_pt2 = xys['left elbow']

            l_pt3 = xys['left wrist']

            l_angle = angle1(l_pt3, l_pt2, l_pt1)

            #print ("오른팔: % .4f " %(l_angle))

 

    except:

 

         pass

 

import RPi.GPIO as GPIO

from time import sleep

 

 

# Pins for Motor Driver Inputs

#right

Motor1A = 19

Motor1B = 13

Motor1E = 26

#left

Motor2A = 5

Motor2B = 6

Motor2E = 0

 

def setup():

    GPIO.setmode(GPIO.BCM)              # GPIO Numberin

    GPIO.setup(Motor1A,GPIO.OUT)  # All pins as Outputs

    GPIO.setup(Motor1B,GPIO.OUT)

    GPIO.setup(Motor1E,GPIO.OUT)

    GPIO.setup(Motor2A,GPIO.OUT)  # All pins as Outputs

    GPIO.setup(Motor2B,GPIO.OUT)

    GPIO.setup(Motor2E,GPIO.OUT)

    

    pwmL=GPIO.PWM(Motor1E,1000)

    pwmL.start(0)

    pwmR=GPIO.PWM(Motor2E,1000)

    pwmR.start(0)

    

    return pwmL, pwmR

        

def forward(pwmL, pwmR):

    # Going forwards 

 

    

    GPIO.output(Motor1A,GPIO.HIGH)

    GPIO.output(Motor1B,GPIO.LOW)

 

    GPIO.output(Motor2A,GPIO.HIGH)

    GPIO.output(Motor2B,GPIO.LOW)

    

    pwmL.ChangeDutyCycle(100)

    pwmR.ChangeDutyCycle(100)

     

def backward(pwmL, pwmR):

 

    

    GPIO.output(Motor1A,GPIO.LOW)

    GPIO.output(Motor1B,GPIO.HIGH)

 

 

    GPIO.output(Motor2A,GPIO.LOW)

    GPIO.output(Motor2B,GPIO.HIGH)

 

    

    pwmL.ChangeDutyCycle(100)

    pwmR.ChangeDutyCycle(100)

    

def turn_R(pwmL, pwmR):

 

    GPIO.output(Motor1A,GPIO.LOW)

    GPIO.output(Motor1B,GPIO.HIGH)

 

 

    GPIO.output(Motor2A,GPIO.LOW)

    GPIO.output(Motor2B,GPIO.HIGH)

 

    

    pwmL.ChangeDutyCycle(0)

    pwmR.ChangeDutyCycle(100)

    

 

def turn_L(pwmL, pwmR):

    

    GPIO.output(Motor1A,GPIO.LOW)

    GPIO.output(Motor1B,GPIO.HIGH)

 

 

    GPIO.output(Motor2A,GPIO.LOW)

    GPIO.output(Motor2B,GPIO.HIGH)

 

    pwmL.ChangeDutyCycle(100)

    pwmR.ChangeDutyCycle(0)

    

    

def cleanup():

    GPIO.cleanup()

    

def stop(pwmL, pwmR):

    pwmL.ChangeDutyCycle(0)

    pwmR.ChangeDutyCycle(0)

 

#cleanup()

 

# GPIO.setmode(GPIO.BCM)              # GPIO Numberin

# GPIO.setup(Motor1A,GPIO.OUT)  # All pins as Outputs

# GPIO.setup(Motor1B,GPIO.OUT)

# GPIO.setup(Motor1E,GPIO.OUT)

# GPIO.setup(Motor2A,GPIO.OUT)  # All pins as Outputs

# GPIO.setup(Motor2B,GPIO.OUT)

# GPIO.setup(Motor2E,GPIO.OUT)

# 

# GPIO.output(Motor1A,GPIO.HIGH)

# GPIO.output(Motor1B,GPIO.LOW)

# GPIO.output(Motor2A,GPIO.HIGH)

# GPIO.output(Motor2B,GPIO.LOW)

# 

# 

# pwmL = GPIO.PWM(Motor1E, 1000)

# pwmL.start(70)

# pwmR = GPIO.PWM(Motor2E, 1000)

# pwmR.start(0)

#setup()

# print("1")

 

cleanup()

pwmL, pwmR = setup()

 

sleep(1)

 

turn_L(pwmL, pwmR)

    

 

sleep(100)

 

 

 

def overlay_on_image(frames, result, model_width, model_height):

 

    color_image = frames

 

    if isinstance(result, type(None)):

        return color_image

    img_cp = color_image.copy()

 

    for pose in result:

        draw_pose(img_cp, pose)

 

    cv2.putText(img_cp, fps,       (model_width-170,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)

    cv2.putText(img_cp, detectfps, (model_width-170,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)

 

    return img_cp

 

if __name__ == '__main__':

 

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite", help="Path of the detection model.")

    parser.add_argument("--usbcamno", type=int, default=0, help="USB Camera number.")

    parser.add_argument('--videofile', default="", help='Path to input video file. (Default="")')

    parser.add_argument('--vidfps', type=int, default=30, help='FPS of Video. (Default=30)')

    args = parser.parse_args()

    

   

    #mo = MC

    model     = args.model

    usbcamno  = args.usbcamno

    vidfps    = args.vidfps

    videofile = args.videofile

 

 

    camera_width  = 640

    camera_height = 480

    model_width   = 640

    model_height  = 480

    

    devices = edgetpu_utils.ListEdgeTpuPaths(edgetpu_utils.EDGE_TPU_STATE_UNASSIGNED)

    engine = PoseEngine(model, devices[0])

   

    if videofile == "":

        cap = cv2.VideoCapture(-1)

        cap.set(cv2.CAP_PROP_FPS, vidfps)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)

        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

        waittime = 1

        window_name = "Web Camera"

    else:

        cap = cv2.VideoCapture(videofile)

        waittime = vidfps - 20

        window_name = "Movie File"

 

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

 

    while True:

        t1 = time.perf_counter()    

        ret, color_image = cap.read()

        color_image = cv2.flip(color_image,1) # 좌우반전

        if not ret:

            continue

 

        # Run inference.

        color_image = cv2.resize(color_image, (model_width, model_height))

        prepimg = color_image[:, :, ::-1].copy()# BGR to RGB 

 

        tinf = time.perf_counter()

        res, inference_time = engine.DetectPosesInImage(prepimg)

        

    

        setup()

        if res:

            detectframecount += 1

            imdraw = overlay_on_image(color_image, res, model_width, model_height)

           

            if  r_angle > 90:

                 print("TURN L") 

                 turn_L()

            elif l_angle > 90:

                 print("TURN R")

                 turn_R()

            else:

                 print("FORWARD")

                 forward()

        else:

            imdraw = color_image

 

        imdraw=cv2.imshow(window_name, imdraw)

        

        if cv2.waitKey(waittime)&0xFF == 27:

            cleanup()

            break

 

        # FPS calculation

        framecount += 1

        if framecount >= 15:

            fps       = "(Playback) {:.1f} FPS".format(time1/15)

            detectfps = "(Detection) {:.1f} FPS".format(detectframecount/time2)

            framecount = 0

            detectframecount = 0

            time1 = 0

            time2 = 0

        t2 = time.perf_counter()

        elapsedTime = t2-t1

        time1 += 1/elapsedTime

        time2 += elapsedTime
