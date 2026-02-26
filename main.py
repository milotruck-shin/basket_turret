from finalver import Camera, TurretTracker
from flask import Flask, render_template,send_file
from flask_socketio import SocketIO
import threading
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import argparse
import cv2 as cv
import pyzed.sl as sl
import numpy as np
from queue import Queue, Empty




#obj detection (main thread) and aiming (another thread)
class ObjectProcess(multiprocessing.Process):  
    def __init__(self, shape, cam_active,coord_queue,distance_active,height,mid,error):
        self.event=cam_active
        self.shape=shape
        self.coord_queue=coord_queue
        self.distance_active=distance_active
        self.h=height
        self.m=mid
        self.error=error  #error sending pipeline
        super().__init__()
        


    def run(self):
        try:
            self.turret=TurretTracker()
            self.error_queue=Queue(maxsize=1)
            self.aiming_event=threading.Event()

            img_shm = SharedMemory(name='zed_frame', create=False)
            received_frame=np.ndarray(self.shape,dtype=np.uint8,buffer=img_shm.buf)

            yolo_shm = SharedMemory(name ='yolo_frame', create=False)
            self.yolo_frame_array = np.ndarray(self.shape, dtype=np.uint8, buffer=yolo_shm.buf)
            self.aiming_event.set()
            aim_thread=threading.Thread(target=self.aim_t, args=(),daemon=True)
            # aim_thread.start()

            while self.event:
                err=self.detection_t(self.turret,received_frame)

                #to pass error to aim_turret
                self.error.send(err)

                if self.error_queue.full():
                    self.error_queue.get_nowait()
                self.error_queue.put(err)

                if self.event == False:
                    break
                
        finally:
            img_shm.close()
            yolo_shm.close()
            yolo_shm.unlink()
            self.aiming_event.clear()

    def detection_t(self,tur,received_frame):
        try:
            frame=received_frame[:]
            midx, midy = tur.ObjDetection(frame, self.h, self.m)
            print("[Detection Thread] Computed error:", tur.err)


            if self.coord_queue.full():
                self.coord_queue.get_nowait()
            self.coord_queue.put_nowait((tur.x1,tur.y1,tur.x2,tur.y2))


            if tur.detect:
                self.distance_active.set()
            else:
                self.distance_active.clear()
                

            #shared memory array
            self.yolo_frame_array[:]=frame[:]

            return tur.err

        except Empty:
            print("Empty queue")

        except Exception as e:
            print(f"[Detection Thread Error] {e}")

    def aim_t(self,cap,tur):
        self.event.wait()
        while self.aiming_event.is_set():
            try:
                turret_err=self.error_queue.get(timeout=0.3)
                tur.turret_aim(cap.start_time,turret_err)
            except Exception as e:
                print(f"[AIM THREAD] Error: {e}")  # Debug line


#Main process functions
def get_dist_2t(cap,cam_active,distance_active,coord_queue,distance_queue,):

    try:
        while cam_active.is_set():

            if distance_active.is_set():
                (x1, y1, x2, y2) = coord_queue.get(timeout=0.3)
                print(f"x1: {x1}, y:{y1}, x2: {x2}, y2:{y2}")

                distance = cap.get_distance2(x1, y1, x2, y2)
                print(f"[Distance Thread] Distance: {distance:.2f}")
                
                    
                # if distance_queue.full():
                #     distance_queue.get_nowait()
                # distance_queue.put_nowait(distance)

            
    
    except Exception as e:
        print(f"[Distance Thread Error] {str(e)}")



def webstreaming(args):
    app = Flask(__name__)
    socketio=socketio=SocketIO(app,async_mode='threading')
    
    @app.route("/")
    def index():
        return render_template("index.html")


    def stream_video():
            print("Client connected, starting upload thread")
            while stream_active.is_set():
                try:
                    frame=processed_frame_queue.get(timeout=0.03)
                    _, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 75])
                    socketio.emit('video_frame',buffer.tobytes())
                except Exception as e:
                    print(f"Upload error: {e}")
    
    @socketio.on('connect')
    def connect_handler():
        socketio.start_background_task(target=stream_video)

    stream_active.wait()            
    socketio.run(app,host=args["ip"], port=args["port"], debug=True, use_reloader=False,allow_unsafe_werkzeug=True)


    
def main():
    
    height=multiprocessing.Value("i",0)
    mid = multiprocessing.Value("i", 0)
    err = multiprocessing.Value("i",0)

    #below to send error to and fro
    parent, child = multiprocessing.Pipe()

    
    x=0
    cap = Camera()

    check = cap.initialise_cam()

    with height.get_lock():
        height.value, mid.value  = cap.getDim()
    # frame_ready.clear()

    coord_queue=multiprocessing.Queue(maxsize=1)  #x1,y1,x2,y2 (pixel coordinates of bounding box)
    upload_frame=multiprocessing.Queue(maxsize=3)   #for websocket image queue
    distance_queue=Queue(maxsize=1)
    distance_ready = threading.Event()


    cam_active=multiprocessing.Event()
    stream_active=multiprocessing.Event()
    distance_active=multiprocessing.Event()
    processor2_event = multiprocessing.Event()

    distance_thread=threading.Thread(target=get_dist_2t, args=(cap,cam_active,distance_active, coord_queue,distance_queue))

    cv.namedWindow("Image",cv.WINDOW_NORMAL)
    cv.namedWindow("Depth View",cv.WINDOW_NORMAL)
    cv.resizeWindow("Image", 600,400)
    cv.resizeWindow("Depth View", 600,400)



    while check:
        if cap.zed.grab() == sl.ERROR_CODE.SUCCESS:
            img, depth_img = cap.read_frame()

            if (x==0):  #to run once
                img_shm=SharedMemory(name='zed_frame', create=True, size=img.nbytes)
                shape = img.shape
                img_array = np.ndarray(shape, dtype=img.dtype,buffer = img_shm.buf)
                yolo_shm = SharedMemory(name='yolo_frame', create=True,size=img.nbytes)
                yolo_received_frame=np.ndarray(shape, dtype=img.dtype, buffer=yolo_shm.buf)
                cam_active.set()        #set event (multiprocessor-wide)
                stream_active.set()

                print(depth_img.dtype)

                objectdetection = ObjectProcess(
                    shape=shape,
                    cam_active=cam_active,
                    coord_queue=coord_queue,
                    distance_active=distance_active,
                    height=height.value,
                    mid=mid.value,
                    error = child
                )
                
                objectdetection.start()

                distance_thread.start()

                x=x+1



            #copy the captured rgb into share memory
            img_array[:]=img[:]

            error=parent.recv()

            # try:
            #     distance = distance_queue.get(timeout=0.3)
            # except Empty:
            #     print("No distance value received in time.")


            cv.putText(yolo_received_frame, f"Error: {error}", (10,30), cv.FONT_HERSHEY_PLAIN, 1,(0, 255, 0), 2)

            # depth_colored = cv.applyColorMap(depth_8u, cv.COLORMAP_JET)



            cv.imshow("Depth View", depth_img)
            # cv.putText(yolo_received_frame, f"Error: {error}, Distance:{distance:.2f}", (10,30), cv.FONT_HERSHEY_PLAIN, 1,(0, 255, 0), 2)

            cv.imshow("Image", yolo_received_frame)            


        if cv.waitKey(1) == ord('q'):
            break
    

    # Cleanup
    processor2_event.set()
    cam_active.clear()
    cv.destroyAllWindows()
    img_shm.close()
    img_shm.unlink()
    yolo_shm.close()
    yolo_shm.unlink()
    cap.zed.close()
    cap.viewer.exit()
    distance_thread.join()

if __name__ == '__main__':
    if multiprocessing.get_start_method() != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    main()
    # ap=argparse.ArgumentParser()
    # ap.add_argument("-i", "--ip", type=str, required=True, help="ip address of the device")
    # ap.add_argument("-o", "--port", type=int, required=True, help="ephemeral port number of the server (1024 to 65535)")
    # args = vars(ap.parse_args())
    # upload_process=multiprocessing.Process(target=webstreaming, args=(args,), daemon=True)
    # upload_process.start()
    # main()

    
    
