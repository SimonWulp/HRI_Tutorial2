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
        
       
    def run_ball_follower(self,mode='') :
        center_x = []
        center_y = []
        pitch = []
        roll = []
        prev_roll = 0
        prev_pitch = 0
        pitch_in_range = lambda pos : pos <= 2.09 and pos >= -2.09
        roll_in_range = lambda pos : pos >= -0.314159 and pos <= 1.31
        model = FFN()
        model_mdn = MDN()
        model_mdn.load_state_dict(torch.load('MDN.pth'))
        model_mdn.eval()
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
                
                if mode == 'data collection' :
                
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
                   
                 
                    if roll_in_range(roll_pos) : 
                        self.shoulder_roll.setPosition(roll_pos)
                        
                    if pitch_in_range(pitch_pos) :
                        self.shoulder_pitch.setPosition(pitch_pos)
                        
                elif mode == 'ffn' :
                    pitch_fnn, roll_fnn = model(torch.tensor([cx,cy]).float())
                    #if roll_in_range(float(roll_fnn)) :
                    self.shoulder_roll.setPosition(float(roll_fnn))
                    
                    #if pitch_in_range(float(pitch_fnn)) :    
                    self.shoulder_pitch.setPosition(float(pitch_fnn))
                    
                elif mode == 'mdn' :
                    pi,sigma,mu = model_mdn(torch.tensor([cx,cy]).float())
                    pitch_mdn, roll_mdn = sample_pred(pi,sigma,mu)
                    self.shoulder_roll.setPosition(float(roll_mdn))
                    
                    #if pitch_in_range(float(pitch_fnn)) :    
                    self.shoulder_pitch.setPosition(float(pitch_mdn))
                    
                    
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
    def __init__(self, n_input=2, n_hidden=50, n_gaussians=5, n_output=2):
        super(MDN, self).__init__()
        self.n_gaussians = n_gaussians
        self.n_output = n_output
        self.network = nn.Sequential(nn.Linear(n_input,n_hidden),
                                     nn.Tanh()
                                     )
        self.softmax = nn.Softmax(dim=0)
        self.coeficient = nn.Linear(n_hidden,n_gaussians)
        self.varience = nn.Linear(n_hidden,n_gaussians)
        self.mean = nn.Linear(n_hidden,n_gaussians*n_output)
      
    def forward(self, x):
        net = self.network(x)
        pi = self.softmax(self.coeficient(net))
        sigma = torch.exp(self.varience(net))
        mu = self.mean(net)
        mu = mu.view(self.n_gaussians, self.n_output)
        return pi, sigma, mu
        
def gaussian_distribution(y, mu, sigma) :
    diff = mu - y.unsqueeze(dim=0)
    sigma_1 =  torch.reciprocal(sigma.unsqueeze(dim=1))
  
    exponent = -0.5 * ((diff) * sigma_1)**2
    factor = 1.0 / np.sqrt(2.0*np.pi)
    p = factor * sigma_1 * torch.exp(exponent)
    return torch.prod(p,-1)
    
def mdn_loss(pi,sigma,mu,y) :
    loss = torch.mul(pi,gaussian_distribution(y,mu,sigma))
    loss = torch.sum(loss,dim=-1)
    loss = torch.mean(torch.log(loss))
    return -loss


model = FFN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.006)

model_mdn = MDN()
mdn_optimizer = optim.Adam(model_mdn.parameters(),lr=0.002)

def train_net_mdn(net, dataloader, epochs=200) :
    for epoch in range(epochs):        
        for index, data in dataloader.iterrows() :
            #print(data)
            input_x,input_y,target_pitch,target_roll = data
            inputs = torch.Tensor([input_x,input_y])
            targets = torch.Tensor([target_pitch, target_roll])
            
            mdn_optimizer.zero_grad()
            pi, sigma, mu = net(inputs)
            loss = mdn_loss(pi, sigma, mu, targets)
            loss.backward()
            mdn_optimizer.step()
            
        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch+1, epochs, loss.item()))
            
    print('Finished Training')  
    
def train_net_ffn(net, dataloader, max_epochs=10):
   
    for epoch in range(max_epochs):        
        for index, data in dataloader.iterrows() :
            #print(data)
            input_x,input_y,target_pitch,target_roll = data
            inputs = torch.Tensor([input_x,input_y])
            targets = torch.Tensor([target_pitch, target_roll])
            print(targets)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch+1, max_epochs, loss.item()))
            
    print('Finished Training')  

def sample_pred(pi, sigma, mu):
    K = pi.shape[0]
    KT = mu.shape[0] * mu.shape[1] 
    NO = int(KT / K)
    pred = Variable(torch.zeros(NO))  
    r = np.random.uniform(0,1) 

    prob_sum = 0
    mu = mu.detach().numpy()
    sigma = sigma.detach().numpy()
    for k in range(K):
        prob_sum += pi.data[k] 
        if r < prob_sum :
          for t in range(NO):
                sample = np.random.normal(mu[k,t], sigma[k])
                pred[t] = sample 
          break
          
    return pred

    

    
#data = pd.read_csv('HRI2.csv')
#data = data.sample(frac=1)
#data = data[0:1000]
#train_net_mdn(model_mdn, data)
#torch.save(model_mdn.state_dict(), 'MDN.pth')
#train_net_ffn(model,data)
#torch.save(model.state_dict(), 'FFN.pth')
robot = MyRobot(ext_camera_flag = True)
''' args mode:str
    data collection - To collect training data
    ffn - To run trained feed forward network implementation
    mdn - To run trained Mixed Density network implementation '''
    
robot.run_ball_follower(mode='mdn')


