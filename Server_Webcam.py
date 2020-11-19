
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:13:21 2019

@author: doganay
"""


import numpy as np
import cv2
import cv2.aruco as aruco
import math
import socket


isServer=int(input("To create server enter 1, otherwise enter 0: "))



if(isServer):
    ## Server Initiation
    TCP_IP = '127.0.0.1'
    TCP_PORT = 30008
    BUFFER_SIZE = 1024  
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(1)
    conn, addr = s.accept()
    print ('Connection address:', addr)


## ArUco Marker Specs
id_number  = 72
markersize_mm  = 100


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


file = open("Marker Configuration.txt","w+")



## Calibration and Distortion Matrix
calib_path  = ""
camera_matrix   = np.loadtxt(calib_path+'CameraCalibration_webcam.txt', delimiter=',')
camera_distortion   = np.loadtxt(calib_path+'Distortion_webcam.txt', delimiter=',')


## Rotation around X axis
R_flip  = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] = 1.0
R_flip[1,1] =-1.0
R_flip[2,2] =-1.0


## Defining ArUco Marker Dictionary
aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters  = aruco.DetectorParameters_create()

## Defining Corner Refinement Method (Improvement)
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
parameters.aprilTagQuadDecimate=2.8


## Capturing Video Camera
cap = cv2.VideoCapture(0)

## Setting the Camera Size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

## Font of OpenCV
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

while True:
    
    ret, frame = cap.read()

    ## Converting the frame to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    ## Finding the ArUco Markers in the Frame
    corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters, cameraMatrix=camera_matrix, distCoeff=camera_distortion)
    
    if ids is not None and ids[0] == id_number:
        
        ret = aruco.estimatePoseSingleMarkers(corners, markersize_mm, camera_matrix, camera_distortion) 
        rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]

        ## Drawing the Detected Marker and Reference Frame
        aruco.drawDetectedMarkers(frame, corners)
        aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 100)

        ## Tag Position in the Frame
        str_position = "Marker Position x=%4.2f  y=%4.2f  z=%4.2f"%(tvec[0], tvec[1], tvec[2])
        cv2.putText(frame, str_position, (0, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        file.write("%4.2f  %4.2f  %4.2f"%(tvec[0], tvec[1], tvec[2]))  
         
        ## Rotation Matrix
        R_ct    = np.matrix(cv2.Rodrigues(rvec)[0])
        R_tc    = R_ct.T
        
        ## Rotation Matrix to Euler Angles
        yaw_marker, pitch_marker, roll_marker = rotationMatrixToEulerAngles(R_ct)
        
        ## Sending the Position Data to Client
        if(isServer):
            conn.send("{:+f},{:+f},{:+f}\n".format(tvec[0],tvec[1],tvec[2]).encode('utf-8'))        
        
        ## Printing in the Console only the Position
        print("{:+f},{:+f},{:+f}\n".format(tvec[0],tvec[1],tvec[2]))
        
    
        ## Tag Orientation in the Frame
        str_attitude = "Marker Orientation r=%4.2f  p=%4.2f  y=%4.2f"%(math.degrees(roll_marker),math.degrees(pitch_marker), math.degrees(yaw_marker))
        cv2.putText(frame, str_attitude, (0, 150), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        file.write(" %4.2f  %4.2f  %4.2f\n"%(math.degrees(roll_marker),math.degrees(pitch_marker),math.degrees(yaw_marker)))        
        
    ## Display the frame
    cv2.imshow('frame', frame)

    ## 'q' for Quitting
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        file.close()
        if(isServer):
            conn.close()
        break


