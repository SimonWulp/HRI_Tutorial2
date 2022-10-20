from controller import Robot, Keyboard, Display, Motion, Motor, Camera, DistanceSensor
import numpy as np
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
import torch
import time


class MyRobot(Robot):
    def __init__(self, ext_camera_flag):
        super(MyRobot, self).__init__()
        print('> Starting robot controller')
        
        self.timeStep = 32 # Milisecs to process the data (loop frequency) - Use int(self.getBasicTimeStep()) for default
        self.state = 0 # Idle starts for selecting different states
      
        self.step(self.timeStep) # Execute one step to get the initial position
        self.camera = self.getCamera('CameraBottom')
        self.camera.enable(self.timeStep)
        #data
        
        
        # Actuators init
        self.shoulder_pitch = self.getDevice("LShoulderPitch")
        self.shoulder_roll = self.getDevice("LShoulderRoll")
        self.elbow = self.getDevice("LElbowRoll")
        self.elbow.setPosition(0)
        self.shoulder_pitch.setVelocity(1)
        self.shoulder_roll.setVelocity(1)
        # Keyboard
        self.keyboard.enable(self.timeStep)
        self.keyboard = self.getKeyboard()
        
       
    def run_ball_follower(self,data_collection=False) :
        center_x = []
        center_y = []
        pitch = []
        roll = []
        prev_roll = 0
        prev_pitch = 0
        pitch_in_range = lambda pos : pos <= 2.09 and pos >= -2.09
        roll_in_range = lambda pos : pos >= -0.31 and pos <= 1.31
        model = FFN()
        model.load_state_dict(torch.load('FFN.pth'))
        model.eval()
        while self.step(self.timeStep) != -1 :
            k = self.keyboard.getKey()
            image_data = self.camera.getImage()
            image = np.frombuffer(image_data, np.uint8).reshape((self.camera.getHeight(), self.camera.getWidth(), 4))
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            low_green = np.array([25, 52, 72])
            high_green = np.array([102, 255, 255])
            green_mask = cv2.inRange(hsv_image, low_green, high_green)
            final_image = cv2.bitwise_and(image, image, mask=green_mask)
            gray_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
            #blur = cv2.GaussianBlur(gray_image, (5, 5), cv2.BORDER_DEFAULT)
            _,thresh = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY)
            contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            try :
                moments = cv2.moments(contours[0])
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                
                if data_collection :
                
                    dx = 0.6*((cx/120) - 0.1)
                    dy = 0.5*((cy/160) + 0.3)
                    
                    roll_pos = self.shoulder_roll.getTargetPosition() - (prev_roll+dx)
                    pitch_pos = self.shoulder_pitch.getTargetPosition() - (prev_pitch-dy)
                    prev_roll = roll_pos
                    prev_pitch = pitch_pos
                    
                    self.shoulder_pitch.setVelocity(5)
                    self.shoulder_roll.setVelocity(5)
                    
                    #if roll_in_range(roll_pos) and pitch_in_range(pitch_pos) :
                    center_x.append(cx)
                    center_y.append(cy)
                    roll.append(roll_pos)
                    pitch.append(pitch_pos)
                    time.sleep(0.08)
                    
                    if roll_in_range(roll_pos) : 
                        self.shoulder_roll.setPosition(roll_pos)
                        
                    if pitch_in_range(pitch_pos) :
                        self.shoulder_pitch.setPosition(pitch_pos)
                        
                else :
                    pitch_fnn, roll_fnn = model(torch.Tensor([cx,cy]))
                    #if roll_in_range(float(roll_fnn)) :
                    self.shoulder_roll.setPosition(float(roll_fnn))
                    
                    #if pitch_in_range(float(pitch_fnn)) :    
                    self.shoulder_pitch.setPosition(float(pitch_fnn))
                    
            except (IndexError, ZeroDivisionError):
                print('ball out of camera range')
                
            if k == ord('D') :
                self.make_data(center_x,center_y,pitch,roll)
                break
   
    def make_data(self,center_x, center_y, pitch, roll) :
        data = pd.DataFrame()
        data['center_x'] = center_x
        data['center_y'] = center_y
        data['pitch'] = pitch
        data['roll'] = roll
        data.to_csv('HRI2.csv',index=False)
           
class FFN(nn.Module) :

    def __init__(self,n_input=2,n_hidden=6,n_output=2) :
        super(FFN, self).__init__()
        self.input = nn.Linear(n_input,n_hidden)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(n_hidden,n_output)
        
    def forward(self,x) :
        x = self.input(x)
        x = self.tanh(x)
        x = self.output(x)
        return x

class MDN(nn.Module):
    def __init__(self, n_input=2, n_hidden=6, n_output=2, n_gaussians=2):
        super(MDN, self).__init__()
        self.network = nn.Sequential(nn.Linear(n_input,n_hidden),
                                     nn.tanh(),
                                     nn.Linear(n_hidden,n_hidden))
        self.coeficient = nn.Linear(n_hidden,n_gaussians)
        self.varience = nn.Linear(n_hidden,n_gaussians)
        self.mean = nn.Linear(n_hidden,n_gaussians*n_output)
      
    def forward(self, x):
        net = self.network(x)
        pi = nn.Softmax(self.coeficient(net),-1)
        sigma = torch.exp(self.varience(net))
        mu = self.mean(net)
        return pi, sigma, mu


    
def train_net(net, dataloader, max_epochs=10):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(max_epochs):        
        for index, data in dataloader.iterrows() :
            input_x,input_y,target_pitch,target_roll = data
            inputs = torch.Tensor([input_x,input_y])
            targets = torch.Tensor([target_pitch, target_roll])
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch+1, max_epochs, loss.item()))
            
    print('Finished Training')  


def gaussian_distribution(y, mu, sigma):
    sigma_1 =  torch.reciprocal(sigma) 

    exponent = -0.5 * ((y.expand_as(mu) - mu) * sigma_1)**2
    factor = 1.0 / np.sqrt(2.0*np.pi)
    p = factor * sigma_1 * torch.exp(exponent)
    return p
    
def mdn_loss(pi,sigma,mu,y):
    loss = pi * gaussian_distribution(y, mu, sigma)
    loss = torch.sum(loss, dim=1)
    loss = -torch.log(loss)
    loss = torch.mean(loss)
    return loss
    
def main():
    data = pd.read_csv('HRI2.csv')
    model = FFN()
    train_net(model,data)
    torch.save(model.state_dict(), 'FFN.pth')
    robot = MyRobot(ext_camera_flag = True)
    robot.run_ball_follower(data_collection=False)

if __name__ == "__main__":
    main()