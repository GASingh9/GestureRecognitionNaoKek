"""
Create a gesture for gesture recognition
"""
import numpy as np
import cv2
import os

class CreateGesture(object):
    def __init__(self,name):
        self.name=name

    def getvid(self):
        """
        Record a video and save.
        """
        cap = cv2.VideoCapture(0)

        # Define the codec and create VideoWriter object
        path=os.path.dirname(__file__)
        print(path+'\\data\\videos\\'+self.name+'.avi')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(path+'\\data\\videos\\'+self.name+'.avi',fourcc, 20.0, (640,480))

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret==True:
                #frame = cv2.flip(frame,0)

                # write the flipped frame
                out.write(frame)

                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def frames(self):
        path=os.path.dirname(__file__)
        vidFile=path+'\\data\\videos\\'+self.name+'.avi'
        vidcap = cv2.VideoCapture(vidFile)
        success,image = vidcap.read()
        count = 0
        success = True
        while success:
            success,image = vidcap.read()
            print('Read a new frame: ', success)
            #https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            path=os.path.dirname(__file__)
            frame_path=path+'\\data\\frames\\'+self.name
            #path = 'C:\\Users\\dse\\Desktop\\UWA\\2017 Sem 2\\GENG5508\\Project- Gesture Recognition\\naorobot\\{}'.format(sys.argv[1])   #folder to store frame
            try:
                os.makedirs(frame_path)
            except OSError as exception:
                if not os.path.isdir(frame_path):
                    raise
            cv2.imwrite(os.path.join(frame_path ,"framec%d.jpg" % count), image)     # save frame as JPEG file
            cv2.waitKey(0)
            count = count + 1

    def getframes(self):
        self.getvid()
        self.frames()
