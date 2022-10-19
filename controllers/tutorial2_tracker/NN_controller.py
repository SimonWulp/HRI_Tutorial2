from controller import Robot, Keyboard, Display, Motion, Motor, Camera, DistanceSensor
import numpy as np
import cv2


class MyRobot(Robot):
    def __init__(self, ext_camera_flag):
        super(MyRobot, self).__init__()
        print('> Starting robot controller')
        
        self.timeStep = 32 # Milisecs to process the data (loop frequency) - Use int(self.getBasicTimeStep()) for default
        self.state = 0 # Idle starts for selecting different states
        
        
        # Sensors init
        self.gps = self.getGPS('gps')
        self.gps.enable(self.timeStep)
      
        self.step(self.timeStep) # Execute one step to get the initial position
        
        self.ext_camera = ext_camera_flag        
        self.displayCamExt = self.getDisplay('CameraExt')
        #face
        self.face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        #camera
        self.camera = self.getCamera('CameraBottom')
        self.camera.enable(self.timeStep)
        devices = self.getNumberOfDevices()
        #for device in range(devices) :
        #    d = self.getDeviceByIndex(device)
        #    print(d.getName())
       
        if self.ext_camera:
            self.cameraExt = cv2.VideoCapture(0)
         
        # Actuators init
 
        self.head_yaw = self.getDevice("HeadYaw")
        self.head_pitch = self.getDevice("HeadPitch")
        self.head_yaw.setVelocity(1)
        self.head_pitch.setVelocity(1)
       
        self.forward_motion = Motion('../../motions/Forwards.motion')
        self.backward_motion = Motion('../../motions/Backwards.motion')
        self.left_step = Motion('../../motions/SideStepLeft.motion')
        self.right_step = Motion('../../motions/SideStepRight.motion')
        
        # Keyboard
        self.keyboard.enable(self.timeStep)
        self.keyboard = self.getKeyboard()
        
        

        
    # Captures the external camera frames 
    # Returns the image downsampled by 2   
    def camera_read_external(self):
        img = []
        if self.ext_camera:
            # Capture frame-by-frame
            ret, frame = self.cameraExt.read()
            # Our operations on the frame come here
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # From openCV BGR to RGB
            img = cv2.resize(img, None, fx=0.5, fy=0.5) # image downsampled by 2
                        
        return img
            
    # Displays the image on the webots camera display interface
    def image_to_display(self, img):
        if self.ext_camera:
            height, width, channels = img.shape
            imageRef = self.displayCamExt.imageNew(cv2.transpose(img).tolist(), Display.RGB, width, height)
            self.displayCamExt.imagePaste(imageRef, 0, 0)
    
    def print_gps(self):
        gps_data = self.gps.getValues();
        print('----------gps----------')
        print(' [x y z] =  [' + str(gps_data[0]) + ',' + str(gps_data[1]) + ',' + str(gps_data[2]) + ']' )
        
    def printHelp(self):
        print(
            'Commands:\n'
            ' H for displaying the commands\n'
            ' G for print the gps\n'
        )
    
    def run_keyboard(self):
        
        self.printHelp()
        previous_message = ''
        # Main loop.
        while True:
            # Deal with the pressed keyboard key.g
            k = self.keyboard.getKey()
            message = ''
            if k == self.keyboard.LEFT:
                self.head_yaw.setPosition(2)
                
            elif k == self.keyboard.RIGHT:
                self.head_yaw.setPosition(-2)
                
            elif k == self.keyboard.UP :
                self.forward_motion.play()
            
            
            elif k == self.keyboard.DOWN :
                self.backward_motion.play()
                
            elif k == ord('S') :
                self.head_yaw.setPosition(0)
                

            # Perform a simulation step, quit the loop when
            # Webots is about to quit.
            if self.step(self.timeStep) == -1:
                break
                
        # finallize class. Destroy external camera.
        if self.ext_camera:
            self.cameraExt.release() 
                
    # Face following main function
    def run_face_follower(self):
        # main control loop: perform simulation steps of self.timeStep milliseconds
        # and leave the loop when the simulation is over
      
        while self.step(self.timeStep) != -1:
            # Write your controller here
            image = self.camera_read_external()
            
            face = self.face_detect.detectMultiScale(image, 1.1, 4)
            for(x, y, w, h) in face :
                cv2.rectangle(image, (x,y), (x+w, y+h),(255,0,0),2)
                center_x,center_y = x + w/2, y + h/2
                x_min,x_max = 0, image.shape[0]
                y_min,y_max = 0, image.shape[1]
                #print(image.shape[0])
                self.look_at(center_x,center_y,x_min,y_min,x_max,y_max)
            
            self.image_to_display(image)
            
        # finallize class. Destroy external camera.
        if self.ext_camera:
            self.cameraExt.release()   
    
    
    cv2.startWindowThread()
    cv2.namedWindow("preview")
    def run_ball_follower(self) :
        

    def look_at(self,c_x, c_y, x_min, y_min,x_max,y_max) :
        pitch_max,pitch_min = 0.5,-0.6
        yaw_max,yaw_min = 1,-2 
        yaw_pos = (((c_x - x_min)/(x_max - x_min))*(yaw_max - yaw_min)) + yaw_min
        pitch_pos = (((c_y - y_min)/(y_max - y_min))*(pitch_max - pitch_min)) + pitch_min
        self.head_pitch.setVelocity(4)
        self.head_yaw.setVelocity(4)
        self.head_pitch.setPosition(pitch_pos)
        self.head_yaw.setPosition(yaw_pos)
        
        
    def run_hri(self) :
        current_head_yaw = self.head_yaw.getTargetPosition()
        current_head_pitch = self.head_yaw.getTargetPosition()
        while self.step(self.timeStep) != -1 :
            self.run_ball_follower()
            if self.head_yaw.getTargetPosition() < current_head_yaw :
                self.right_step.play()
            if self.head_yaw.getTargetPosition() > current_head_yaw :
                self.left_step.play()
            if self.head_pitch.getTargetPosition() > current_head_pitch :
                self.backward_motion.play()
                
            if self.head_pitch.getTargetPosition() < current_head_pitch :
                self.forward_motion.play()
            
    
# create the Robot instance and run the controller
print('sab changa si')
robot = MyRobot(ext_camera_flag = True)
#robot.run_keyboard()
robot.run_face_follower()
#robot.run_ball_follower()
#robot.run_hri()


