import torch
import os
import cv2 as cv
from ultralytics import YOLO
import numpy as np
from ultralytics.utils.plotting import Annotator
import serial
import time
import sys
import math
import ogl_viewer.viewer as gl
import pyzed.sl as sl
import argparse

# cv.namedWindow("Image",cv.WINDOW_NORMAL)
# cv.namedWindow("Depth",cv.WINDOW_NORMAL)
# cv.resizeWindow("Image", 600,400)
# cv.resizeWindow("Depth", 600,400)

def parse_args(init):
    # if len(opt.ip_address)>0 :
    #     ip_str = opt.ip_address
    #     if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
    #         init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
    #         print("[Sample] Using Stream input, IP : ",ip_str)
    #     elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
    #         init.set_from_stream(ip_str)
    #         print("[Sample] Using Stream input, IP : ",ip_str)
    #     else :
    #         print("Unvalid IP format. Using live stream")

    if ("HD2K" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        print("[Sample] No valid resolution entered. Using default")
    else : 
        print("[Sample] Using default resolution")


class Camera:
    def __init__(self):
        self.start_time = None
        self.init = sl.InitParameters(depth_mode=sl.DEPTH_MODE.PERFORMANCE,
                                 coordinate_units=sl.UNIT.METER,
                                 coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,
                                 camera_resolution = sl.RESOLUTION.HD720,
                                 camera_fps=60, depth_minimum_distance = 0.3)
        
        self.image_zed=sl.Mat()  #rgb image
        self.depth_image_zed=sl.Mat()  #depth image
        self.image_size=None
        self.res = sl.Resolution()       
        self.res.width = 720
        self.res.height = 404
        self.distance=0
        self.depth=sl.Mat()
        self.point_cloud = sl.Mat(self.res.width, self.res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
        self.zed = sl.Camera()  #create zed cam obj
        self.midX=None
        self.midY=None


    def initialise_cam(self):
        status = self.zed.open(self.init)  #open the camera
        if status != sl.ERROR_CODE.SUCCESS:  #if fail, print fail message and close
            print(repr(status))
            self.zed.close()
            exit()
            return False
        
        else:
            self.start_time = int(time.monotonic())
            self.runtime = sl.RuntimeParameters()       #set runtime parameters after opening cam

            return True
    
    def getDim(self):
        self.image_size = self.zed.get_camera_information().camera_configuration.resolution

        mid=int(self.image_size.width/2)
        self.image_zed = sl.Mat(self.image_size.width, self.image_size.height, sl.MAT_TYPE.U8_C4)
        self.depth_image_zed = sl.Mat(self.image_size.width, self.image_size.height, sl.MAT_TYPE.U8_C4)
        # self.point_cloud = sl.Mat(self.res.width, self.res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

        return int(self.image_size.height), int(mid)

    def read_frame(self):
        # err = self.zed.grab(self.runtime)
        # if err == sl.ERROR_CODE.SUCCESS :
        self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT, sl.MEM.CPU, self.image_size)
        self.zed.retrieve_image(self.depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, self.image_size)
        # recover data from sl.Mat to use it with opencv, use the get_data() method
        # returns a numpy array that can be used as a matrix with opencv. Does not allocate or copy new image data.
        image_ocv = self.image_zed.get_data()
        depth_image_ocv = self.depth_image_zed.get_data()

        # This creates a new NumPy array in RAM with RGB values, and no longer shares memory with sl.Mat.
        image_ocv = cv.cvtColor(image_ocv, cv.COLOR_RGBA2RGB)
        return image_ocv, depth_image_ocv
    
    def open_view(self):
        camera_model = self.zed.get_camera_information().camera_model
        self.viewer = gl.GLViewer()
        self.viewer.init(1, sys.argv, camera_model, self.res)            
        return self.viewer.is_available()  # return viewer status

    def get_dist(self,midx,midy):
        self.zed.retrieve_measure(self.point_cloud,sl.MEASURE.XYZRGBA)
        # x =round(self.image_zed.get_width()/2)
        # y =round(self.image_zed.height()/2)
        if midx and midy !=None:
            err, point_cloud_value=self.point_cloud.get_value(midx,midy)

            if math.isfinite(point_cloud_value[2]):
                distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                    point_cloud_value[1] * point_cloud_value[1] +
                                    point_cloud_value[2] * point_cloud_value[2])
                print(f"Distance to Camera at {{{midx};{midy}}}: {distance}")
                return distance
            else : 
                print(f"The distance can not be computed at {{{midx};{midy}}}")
                distance = 0
                return distance
        else:
            return 0
        
    def get_distance2(self,x1,y1,x2,y2):
        self.zed.retrieve_measure(self.point_cloud,sl.MEASURE.XYZRGBA)
        if x1 !=None:
            newy2=int(0.25*(y2-y1)+y1)
            cloud_array= self.point_cloud.get_data() #shape of (height,width,4), each pixel has 4 attributes (X,Y,Z,A)
            points = cloud_array[y1:newy2,x1:x2,:3]
            point = points.reshape((-1,3))
            mask = ~(np.isnan(point).any(axis=1))
            point=point[mask]
            distances=np.linalg.norm(point,axis=1)
            closest_d=min(distances)
            return closest_d

        else:
            return 0


class TurretTracker:
    def __init__(self):
        self.scanning = False
        self.scan_dir = 1
        self.model = YOLO(r"/home/sunwayrobocon/Documents/robocon25/best.engine")
        self.err = 0
        self.hoop = False
        self.angle=0
        self.detect=False
        self.x1=0
        self.y1=0 
        self.x2=0
        self.y2=0
        #UNCOMMENT THIS WHEN YOU WANT TO TEST WITH TURRET AIMING
        # self.turret = serial.Serial(
        #     port='/dev/ttyUSB0',
        #     baudrate=115200,
        #     bytesize=serial.EIGHTBITS,
        #     parity=serial.PARITY_NONE,
        #     stopbits=serial.STOPBITS_ONE,
        #     timeout=5
        # )
        self.fontScale = 1.5
        self.fontFace = cv.FONT_HERSHEY_PLAIN
        self.fontColor = (0, 255, 0)
        self.fontThickness = 1


    def ObjDetection(self, frame, height, mid):
        results = self.model.track(source=frame, exist_ok=True ,conf = 0.5, imgsz=640, stream=True)

        self.err=0
        self.detect = False
    

        for r in results:
            # annotator = Annotator(im=frame)
            boxes = r.boxes

            if boxes is None or len(boxes) == 0:
                continue
            
            self.detect = True

            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0] 
                x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
                self.x1, self.y1, self.x2, self.y2 = int(x1),int(y1),int(x2),int(y2)
                
                # Get centre coordinate
                self.midX = (x1+x2)/2
                self.midY = (y1+y2)/2


                conf = float(box.conf[0])  # Confidence score
                c = box.cls
                label = f"{conf:.2f}"

                # Draw bounding box
                #annotator.box_label(b, model.names[int(c)])
                cv.rectangle(frame,pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=(0, 255, 128), thickness = 3, lineType=cv.LINE_8 )
                cv.putText(frame, label, (int(x1), int(y1)-10), self.fontFace, self.fontScale, self.fontColor, self.fontThickness, cv.LINE_AA)
                cv.circle(frame, (int(self.midX), int(self.midY)), radius=4, color=(0, 0, 255), thickness=-1)
                cv.line(frame, (int(mid), 0), (int(mid), int(height)), (255, 0, 0), 2)  

                self.err = self.midX - mid  #in pixel
                return int(self.midX),int(self.midY)
        return int(0),int(0)

    def turret_aim(self,start_time,err):
        current_time = int(time.monotonic())-start_time
        if not self.detect and not self.scanning and current_time%5==0:
            self.scanning = True
            self.angle = 25

        if self.scanning:
            self.turret.write(f"{self.angle: .1f}\n".encode())
            self.angle += 3*self.scan_dir

            if self.angle >=160 and self.scan_dir==1:
                self.scan_dir=-1
                self.angle += 3*self.scan_dir
                # self.turret.write(f"{self.angle: .1f}\n".encode())

            elif self.angle <=25 and self.scan_dir==-1:
                self.scan_dir=1
                self.angle += 3*self.scan_dir
                # self.turret.write(f"{self.angle: .1f}\n".encode())

            if self.detect:
                self.scanning = False

        else:
            
            if (err)>30: #right side of middle line
                self.angle = min(self.angle + 1, 180)
                

            elif(err<-30): #target on the left
                self.angle= max(self.angle-1,0)

            self.turret.write(f"{self.angle: .1f}\n".encode())

class Shooter:
    def __init__(self):
        self.shooter = serial.Serial(
            port='/dev/ttyUSB0',
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=5
        )
        

def main():
    cap = Camera()
    # parse_args(cap.init)
    turret=TurretTracker()
    check = cap.initialise_cam()
    height,mid=cap.getDim()
    viewer_available=cap.open_view()
    while check and viewer_available:
        if cap.zed.grab() == sl.ERROR_CODE.SUCCESS:
            img,depth_img=cap.read_frame()
            cap.zed.retrieve_measure(cap.point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, cap.res)
            cap.viewer.updateData(cap.point_cloud)
            # Check if viewer is still available
            viewer_available = cap.viewer.is_available()


            midx,midy=turret.ObjDetection(img,height,mid)
            # dist=cap.get_dist(midx,midy)

            if turret.detect==True:
                dist=cap.get_distance2(turret.x1,turret.y1,turret.x2,turret.y2)
                cv.putText(img, f"Error: {turret.err:.1f}, Distance:{dist:.2f}", (10,30), turret.fontFace, 1, turret.fontColor, 2)



            # turret.turret_aim(cap.start_time)
            else:
                cv.putText(img, f"Error: {turret.err:.1f}", (10,30), turret.fontFace, 1, turret.fontColor, 2)
            # cv.putText(img, f"Error: {turret.err:.1f}, Distance:{dist:.2f}", (10,30), turret.fontFace, 1, turret.fontColor, 2)

            # cv.imshow('YOLO', frame)   #stream video

            cv.imshow("Image", img)
            cv.imshow("Depth", depth_img)



        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()
    cap.zed.close()
    cap.viewer.exit()


if __name__ == '__main__':
    #parser = argparse.ArgumentParser()  #Creates an argument parser object. Sets up the "rules" for command-line argument parsing but does not process any arguments yet.
    # parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
    #parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    #opt = parser.parse_args() #Parses the actual command-line arguments.Reads sys.argv (or a provided list), validates input, and returns an object (opt) containing the argument values
    main()
 
    # a = fork()
    # if (a == pid){
    #     # parent
    #     # shared_memory

    # } else  (a == 0){


    # }



#
