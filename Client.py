# -*- coding: utf-8 -*-
"""
Created on Mon Oct  13 20:39:35 2019

@author: doganay
"""


import socket

HOST = '127.0.0.1'  
PORT = 30008       
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

while True:
    msg=s.recv(128)
    if not msg: 
        break
    fdata=msg.decode('utf-8')
    print(fdata)