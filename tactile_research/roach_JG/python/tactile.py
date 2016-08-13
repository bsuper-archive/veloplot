#!/usr/bin/env python
"""
author: jgoldberg

"""
from __future__ import division
from lib import command
import time,sys,os
import threading
import serial
import shared
import numpy as np
import thread
import scipy.io #for .mat reading
  
import termios
import fcntl

from hall_helpers import *

import skinvisualizer
import skinvisualizer2
import skinvisualizer3

ROWS = 0
COLS = 0
count = 0
grid = [0,0,0,0]
bpack = 0
packetnumber = 0
mins = np.zeros(0)
maxes = np.zeros(0)
averageFrame = None
averageCount = 0
averageLines = 0
#keyboard polling block
shared.enter = threading.Event()
class KeyboardPoller( threading.Thread ) :
    def run( self ) :
        #global key_pressed
        #ch = sys.stdin.read( 1 )
        while True:
            #ch = myGetch()
            #print "pressed:",ch
            raw_input()
            #print "enter"
            shared.enter.set()
        #if ch == 'K' : # the key you are interested in
        #    key_pressed = 1
        #else :
        #    key_pressed = 0

def myGetch():
    fd = sys.stdin.fileno()

    oldterm = termios.tcgetattr(fd)
    newattr = termios.tcgetattr(fd)
    newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, newattr)

    oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)

    try:        
        while 1:            
            try:
                c = sys.stdin.read(1)
                #break
                return c
            except IOError: pass
    finally:
        termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
        fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)
#end keyboard polling block

def main():
    global poller
    poller = KeyboardPoller()
    #poller.start()
    #thread.start_new_thread(skinvisualizer3.main, ()) #uncomment this line to run opengl visualizer
    setupSerial()
    #return
    # Send robot a WHO_AM_I command, verify communications
    queryRobot()
    time.sleep(1)
    
    getSkinSize()
    time.sleep(2)
    row = 0
    col = 0
    dur = 15
    period = 500
    while True:
        temp = raw_input("enter pixel to sample (rc):")
        r = int(temp[0])
        c = int(temp[1])
        samplePixel(r, c)
    return
    startScan();
    time.sleep(1)
    return

    while True:
        
        raw_input("hit enter to start and stop");
        startScan();
        raw_input();
        stopScan();
    return

    while True:
        break
        #raw_input("hit enter")
        #print "Sampling"
        samplePixel(0, 0)
        #testFrame()
        time.sleep(0.1)
    #return
    #sampleFrame(period)
    #time.sleep(3)
    #sampleFrame(period)
    #time.sleep(3)
    #sampleFrame(period)
    #time.sleep(3)
    #pollPixel(row, col, dur, period)
    #for i in range(1000):
    global bpack
    while True:
        #print "sending test frame", bpack
        bpack = bpack + 1
        #testFrame()
        #raw_input("hit enter")
        sampleFrame(period)
        #time.sleep(.02) #minimum working
        time.sleep(.1)
        #queryRobot()
        '''
        samplePixel(1, 4)
        time.sleep(.05)
        samplePixel(2, 4)
        time.sleep(.05)
        samplePixel(6, 4)
        time.sleep(.05)
        samplePixel(11, 4)
        time.sleep(.05)
        '''

        #print "main",enter.isSet()
    period = 250
    #while True:
    #    sampleFrame(period)
    #    print "sent sample command"
    #    time.sleep(1)


def samplePixel(row, col):
	xb_send(0, command.TACTILE, 'A' + chr(row) + chr(col))

def sampleFrame(period):
    #period in microseconds
    xb_send(0, command.TACTILE, 'B' + chr(period % 256) + chr(period >> 8))

def pollPixel(row, col, duration, period):
    #duration in seconds
    #period in milliseconds (must be < 256)
    xb_send(0, command.TACTILE, 'C' + chr(row) + chr(col) + chr(duration) + chr(period))
    time.sleep(duration + 2)

def startScan():
    xb_send(0, command.TACTILE, 'E')

def stopScan():
    xb_send(0, command.TACTILE, 'F')

def getSkinSize():
    xb_send(0, command.TACTILE, 'G')


def testFrame():
    xb_send(0, command.TACTILE, 'T')

previous = -1
skip = 0
def handleTactilePacket(data):
    global ROWS
    global COLS
    global bpack
    global previous
    global skip
    global count
    global grid
    global packetnumber
    global mins
    global maxes
    global averageFrame
    global averageCount
    global averageLines
    #print "----------"
    #print "mode:", data[0]
    #for i in range(0,len(data)):
    #    print "data: ", ord(data[i])
    if data[0] == 'A' or data[0] == 'C':
        print "row:", ord(data[2]), "col:", ord(data[3])
        val = ord(data[4]) + (ord(data[5])*256)
        print "value =", val
        grid[count] = val
        count = count + 1
        count = 0
        if count == 4:
            count = 0
            packetnumber = packetnumber + 1
            print packetnumber
            print grid[2], grid[1]
            print grid[3], grid[0]
        
    elif data[0] == 'B' or data[0] == 'E':
        print "received B packet", bpack
        bpack = bpack + 1
        #bpack = bpack + 1
        if ROWS == 0 or COLS == 0:
            print "ERROR: Row and column size hasn't been set"
            return
        #print len(data)
        temp = np.zeros(len(data))
        for i in range(len(data)):
            temp[i] = ord(data[i])
        temp = np.uint8(temp)
        #print list(data)
        #print temp
        frame = temp[2:-1:2] + (temp[3::2]*256)
        #print frame
        # this does calibration
        newframe = np.zeros(ROWS*COLS)
        for i in range(ROWS*COLS):
            if frame[i] < mins[i]:
                mins[i] = frame[i]
            elif frame[i] > maxes[i]:
                maxes[i] = frame[i]
            newframe[i] = (frame[i] - mins[i]) / (maxes[i] - mins[i])
        #newframe = np.reshape(newframe, (ROWS,COLS))
        #print " ", newframe[3,0], newframe[4,0]
        #print newframe[2,0], "   ", newframe[5,0]
        #print " ", newframe[1,0], newframe[0,0]
        
        #newframe = newframe * 100
        #print mins
        #print maxes
        '''
        print("    %4.f  " % (frame[5]))
        print("%4.f    %4.f" % (frame[4], frame[0]))
        print("%4.f    %4.f" % (frame[3], frame[1]))
        print("    %4.f  " % (frame[2]))
        
        print("    %.2f  " % (newframe[5]))
        print("%.2f    %.2f" % (newframe[4], newframe[0]))
        print("%.2f    %.2f" % (newframe[3], newframe[1]))
        print("    %.2f  " % (newframe[2]))

        for i in [0,2,4,6,9,11,13,15]:
            print("%.2f " % newframe[i]),
        print
        '''
        
        '''
        -0.5  0.25   0   0.25   -0.5  0.25   0    0.25
        0    -0.25  0.5   -0.25  0   -0.25  0.5   -0.25
        0    -0.25  0    -0.25  0   -0.25  0   -0.25
        0    -0.333333333333333  0    0.333333333333333   0   0.333333333333333   0   -0.333333333333333
        0    0.0545808966861598  0.0428849902534113  0.0545808966861599  0   -0.0545808966861598 -0.0428849902534113 -0.0545808966861599
        -0.117647058823529  0.0545808966861599  0.0428849902534113  0.0545808966861598  0.117647058823529   -0.0545808966861599 -0.0428849902534113 -0.0545808966861597
        '''

        print("    %4.f      :    :      %4.f    " % (frame[11],frame[6]))
        print("%4.f    %4.f  :    :  %4.f    %4.f" % (frame[9],frame[13],frame[4],frame[0]))
        print("    %4.f      :    :      %4.f    " % (frame[15],frame[2]))
        print
        print("    %.2f      :    :      %.2f    " % (newframe[11],newframe[6]))
        print("%.2f    %.2f  :    :  %.2f    %.2f" % (newframe[9],newframe[13],newframe[4],newframe[0]))
        print("    %.2f      :    :      %.2f    " % (newframe[15],newframe[2]))
        shared.zvals = [newframe[0],newframe[2],newframe[4],newframe[6],newframe[9],newframe[11],newframe[13],newframe[15]]
        
        np.set_printoptions(precision=3,suppress=True)
        
        '''
        dist1 = 1/((frame[0]+794.39)/7326.6)
        dist2 = 1/((frame[2]+989.47)/8617)
        dist3 = 1/((frame[4]+793.08)/7328.4)
        dist4 = 1/((frame[6]+1074.3)/9582.8)
        print
        print("%.3f" % dist1)
        print("%.3f" % dist2)
        print("%.3f" % dist3)
        print("%.3f" % dist4)

        A = np.array([[8.9127,-4.4563,0,-4.4563],[0,1.5954,-3.1908,1.5954],[0,0.5,0,0.5]])
        x = np.array([dist1,dist2,dist3,dist4])
        np.set_printoptions(precision=3,suppress=True)
        xyz0 = A.dot(x)
        print xyz0

        dist5 = 1/((frame[9]+945.28)/8536.8)
        dist6 = 1/((frame[11]+1118.8)/9611.3)
        dist7 = 1/((frame[13]+881.82)/8049)
        dist8 = 1/((frame[15]+892.76)/8547.4)
        print
        print("%.3f" % dist5)
        print("%.3f" % dist6)
        print("%.3f" % dist7)
        print("%.3f" % dist8)

        A = np.array([[-8.9127,4.4563,0,4.4563],[0,-1.5954,3.1908,-1.5954],[0,0.5,0,0.5]]) #using same cal values as left
        x = np.array([dist5,dist6,dist7,dist8])
        xyz1 = A.dot(x)
        print xyz1

        shared.xyzvals = [xyz0[0],xyz0[1],xyz0[2],xyz1[0],xyz1[1],xyz1[2]]
        '''

        #FOR ENTIRE ARRAY AND 6-DOF
        '''
        dist1 = 1/((frame[0]+515.18)/5876.8)
        dist2 = 1/((frame[2]+500.3)/6171.6)
        dist3 = 1/((frame[4]+425.45)/5967.9)
        dist4 = 1/((frame[6]+590.64)/7028.5)
        dist5 = 1/((frame[9]+366.96)/5398.3)
        dist6 = 1/((frame[11]+569.35)/6501.6)
        dist7 = 1/((frame[13]+449.9)/6144)
        dist8 = 1/((frame[15]+448.44)/6253.5)
        '''

        dist1 = 1.0/((frame[0]+594.68)/7276.3)
        dist2 = 1.0/((frame[2]+868.71)/9058.7)
        dist3 = 1.0/((frame[4]+1000.2)/9529.5)
        dist4 = 1.0/((frame[6]+941.52)/10029.0)
        dist5 = 1.0/((frame[9]+1038.9)/9763.0)
        dist6 = 1.0/((frame[11]+1078.5)/9985.2)
        dist7 = 1.0/((frame[13]+774.43)/8176.5)
        dist8 = 1.0/((frame[15]+1062.4)/10272.0)
        print
        print("    %.3f     :    :     %.3f    " % (dist6,dist4))
        print("%.3f    %.3f:    :%.3f    %.3f" % (dist5,dist7,dist3,dist1))
        print("    %.3f     :    :     %.3f    " % (dist8,dist2))

        #print dist1,dist2,dist3,dist4,dist5,dist6,dist7,dist8
        A = np.array([[8.9127,-4.4563,0,-4.4563],[0,1.5954,-3.1908,1.5954],[0,0.5,0,0.5]])
        x = np.array([dist1,dist2,dist3,dist4])
        xyz0 = A.dot(x)
        print
        #print xyz0
        A = np.array([[-8.9127,4.4563,0,4.4563],[0,-1.5954,3.1908,-1.5954],[0,0.5,0,0.5]]) #using same cal values as left
        x = np.array([dist5,dist6,dist7,dist8])
        xyz1 = A.dot(x)
        #print xyz1
        shared.xyzvals = [xyz0[0],xyz0[1],xyz0[2],xyz1[0],xyz1[1],xyz1[2]]

        '''
        A = np.array([[-0.5,0.30357,0,0.30357,0.5,-0.30357,0,-0.30357],
            [0,-0.19643,0.5,-0.19643,0,0.19643,-0.5,0.19643],
            [0,-0.16667,-0.16667,-0.16667,0,-0.16667,-0.16667,-0.16667],
            [0,-0.33333,0,0.33333,0,0.33333,0,-0.33333],
            [0,0.071429,0,0.071429,0,-0.071429,0,-0.071429],
            [-0.11765,0.039216,0.039216,0.039216,-0.11765,0.039216,0.039216,0.039216]
            ])'''
        
        l1 = 8.5
        l2 = 7.0
        l3 = 5.5
        l4 = 1.5
        yscale = .1122
        xscale = .3134

        A = np.array([[0,yscale,1,0,-l1/2,l1/2*yscale], #for printed piece
            [0,0,1,-l4/2,-l2/2,0],
            [xscale,0,1,0,-l3/2,0],
            [0,0,1,l4/2,-l2/2,0],
            [0,-yscale,1,0,l1/2,l1/2*yscale],
            [0,0,1,l4/2,l2/2,0],
            [-xscale,0,1,0,l3/2,0],
            [0,0,1,-l4/2,l2/2,0]])
        A = np.linalg.pinv(A)

        x = np.array([dist1,dist2,dist3,dist4,dist5,dist6,dist7,dist8])
        xyzrpy = A.dot(x)
        print("x:%.3f y:%.3f z:%.3f roll:%.3f pitch:%.3f yaw:%.3f"%(xyzrpy[0],xyzrpy[1],xyzrpy[2],xyzrpy[3],xyzrpy[4],xyzrpy[5]))
        

        shared.xyzrpy = [xyzrpy[0],xyzrpy[1],xyzrpy[2],xyzrpy[3],xyzrpy[4],xyzrpy[5]]

        N = scipy.io.loadmat('/Users/jgoldberg/Dropbox/Research/N_matrix_trial5.mat')['N']
        A = np.array([frame[0],np.power(frame[0],2),np.power(frame[0],3),
            frame[2],np.power(frame[2],2),np.power(frame[2],3),
            frame[4],np.power(frame[4],2),np.power(frame[4],3),
            frame[6],np.power(frame[6],2),np.power(frame[6],3),
            frame[9],np.power(frame[9],2),np.power(frame[9],3),
            frame[11],np.power(frame[11],2),np.power(frame[11],3),
            frame[13],np.power(frame[13],2),np.power(frame[13],3),
            frame[15],np.power(frame[15],2),np.power(frame[15],3)])
        F = A.dot(N)
        print("Fx:%.4f Fy:%.4f Fz:%.4f Froll:%.4f Fpitch:%.4f Fyaw:%.4f"%(F[0],F[1],F[2],F[3],F[4],F[5]))

        #return
        averageMax = 50.0
        if shared.enter.isSet():
            if averageCount == 0:
                averageFrame = frame/averageMax
            elif averageCount < averageMax:
                averageFrame = averageFrame + frame/averageMax
            averageCount = averageCount + 1
            '''
            if averageCount == averageMax:
                averageFrame = averageFrame / averageMax
                print "#############"
                print averageFrame
                print "#############"
                fd = open('test.csv','a')
                if averageLines == 0:
                    myCsvRow = "Normalized Averages,Samples = " + str(averageMax) + "\n"
                    fd.write(myCsvRow)
                myCsvRow = str(averageLines) + ","
                for i in range(len(averageFrame)):
                    myCsvRow = myCsvRow + "," + str(averageFrame[i])
                myCsvRow = myCsvRow + "\n"
                fd.write(myCsvRow)
                fd.close()
                averageCount = 0
                averageLines = averageLines + 1
                shared.enter.clear()
            '''
            fd = open('test.csv','a')
            fd_avg = open('test_avg.csv','a')
            if averageLines == 0:
                myCsvRow = "Averages,Samples per div = " + str(averageMax) + "\n"
                fd.write(myCsvRow)
            timenow = '%.6f' % time.time()
            myCsvRow = str(averageLines) + "," + timenow
            for i in range(len(frame)):
                myCsvRow = myCsvRow + "," + str(frame[i])
            myCsvRow = myCsvRow + "\n"
            fd.write(myCsvRow)
            averageLines = averageLines + 1
            if averageCount == averageMax:
                #averageFrame = averageFrame / averageMax
                print "#############"
                print averageFrame
                print "#############"
                myCsvRow = str(averageLines/averageMax - 1) + ",," + timenow
                #myCsvRow = ",avg"
                for i in range(len(averageFrame)):
                    myCsvRow = myCsvRow + "," + str(averageFrame[i])
                myCsvRow = myCsvRow + "\n"
                fd_avg.write(myCsvRow)
                #averageLines = averageLines + 1
            fd.close()
            fd_avg.close()
            if averageCount == averageMax:
                averageCount = 0
                shared.enter.clear()
            
        
            #print "set"

        #print("  %.2f  %.2f" % (newframe[3], newframe[4]))
        #print("%.2f      %.2f" % (newframe[2], newframe[5]))
        #print("  %.2f  %.2f" % (newframe[1], newframe[0]))
    elif data[0] == 'G':
        ROWS = ord(data[2])
        COLS = ord(data[3])
        print "shell has", ROWS, "rows and", COLS, "columns."
        
        #calibrated values hardcoded
        mins = np.ones(ROWS*COLS) * 200
        maxes = np.ones(ROWS*COLS) * 4000

    elif data[0] == 'T':
        '''for i in range(len(data)):
            if i == 0:
                print data[i]
            else:
                print ord(data[i])'''

        print "received T packet", bpack
        bpack = bpack + 1
        print map(ord, data)
    elif data[0] == 'X':
        print ord(data[0]),ord(data[-1])
        if ord(data[0]) != (previous + 1) % 256:
            skip = skip + 1 
            print "skip:",skip/float(previous)
            
        previous = ord(data[0])

#Provide a try-except over the whole main function
# for clean exit. The Xbee module should have better
# provisions for handling a clean exit, but it doesn't.
if __name__ == '__main__':
    try:
        main()
        #time.sleep(6)
        while True:
            time.sleep(1)
        print "----------"
        #xb_safe_exit()
    except KeyboardInterrupt:
        print "\nRecieved Ctrl+C, exiting."
        stopScan()
        time.sleep(0.5)
        stopScan()
        time.sleep(0.5)
        stopScan()
        shared.xb.halt()
        shared.ser.close()
        poller._Thread__stop()
    #except Exception as args:
    #    print "\nGeneral exception:",args
    #    print "Attemping to exit cleanly..."
    #    shared.xb.halt()
    #    shared.ser.close()
    #    sys.exit()
    #except serial.serialutil.SerialException:
    #    shared.xb.halt()
    #    shared.ser.close()
