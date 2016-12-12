# 12/5 Test1120 + findNodes + finds elements only (and its center) + clusterNumber
# + resize + build 2D array + generateFile + write to file + control.py + netlist

#for control
import io
import time
import picamera
import RPi.GPIO as GPIO

#for image recognition 
import cv2
import numpy as np
#from matplotlib import pyplot as plt

###################### Initialization ############################################
GPIO.setmode(GPIO.BOARD)
GPIO.setup(7, GPIO.IN)

stream = io.BytesIO()

camera = picamera.PiCamera()
camera.resolution = (1024, 768)
camera.framerate = 10
camera.rotation = 270
count = 0
start = True
######################## Take picture ############################################
while start:
    value = GPIO.input(7)
    print str(value)
    if value == True:
        count = count + 1
    if count == 5:
        time.sleep(10)
        camera.capture(stream, format='jpeg')

        #Convert the picture into a numpy array
        buff = np.fromstring(stream.getvalue(), dtype=np.uint8)
        #Now creates an OpenCV image
        image = cv2.imdecode(buff, 1)
        resize = cv2.resize(image, (400, 300))
        #Save the result image
        cv2.imwrite('circuitrix.jpg',resize)
        start = False

GPIO.cleanup()
print "picture taken!"
####################### Loading the trained data #################################
samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.ml.KNearest_create()
model.train(samples,cv2.ml.ROW_SAMPLE,responses)

###################### Load image and apply filters ###############################
img = cv2.imread('circuitrix.jpg')
img = cv2.resize(img, (400, 300))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#These three for drawing onto a blank black
circuit = np.zeros(img.shape,np.uint8)
digits = np.zeros(img.shape, np.uint8)
gray_circuit = np.zeros(img.shape,np.uint8)

#Remove all the pixels near boundary depending upon the size of kernel
kernel = np.ones((3, 3),np.uint8)

gaussian = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

ret,invert = cv2.threshold(gaussian,127,255,cv2.THRESH_BINARY_INV)

erosion = cv2.erode(invert, kernel, iterations = 1)

#Increase white object area
bw = cv2.dilate(erosion,kernel,iterations = 2)

################################## Keep only digits ###################################
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

image,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

####################### Debugging Purpose, putText numbers onto digits##################]
for cnt in contours:
    if cv2.contourArea(cnt)<700:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>28:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255, 0, 0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            roismall = roismall.reshape((1,100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
            string = str(int((results[0][0])))
            cv2.putText(digits,string,(x,y+h),0,1,(255, 255, 255), thickness=2)

#######################################################################################
second_contours = list(contours)
length = len(contours)
j=0;

######## Delete resistor and noise from second_contours (Keep only digits) ###########
for i in range(0, length):
    if cv2.contourArea(contours[i]) > 700 or cv2.contourArea(contours[i]) < 50:
        del second_contours[j]
    else:
        j=j+1

length_second = len(second_contours)
flags = [0] * length_second
cluster_number = [0] * 30#length_second
q = 0

second_contours = sorted(second_contours, key = cv2.boundingRect, reverse = False)

############################# Find cluster numbers ###################################
for i in range(0, length_second-2):
    [x1,y1,w1,h1] = cv2.boundingRect(second_contours[i])
    if (flags[i] == 0 and h1>28):
        for j in range(i, length_second-1):
            [x2,y2,w2,h2] = cv2.boundingRect(second_contours[j])
            if (flags[j] == 0 and h2 >28):
                if ((x1 - x2) < 60):
                    distance = np.sqrt(np.power((x2 - x1), 2) + np.power((y2 - y1), 2))
                    if (distance < 60):
                        roi = thresh[y2:y2+h2,x2:x2+w2]
                        roismall = cv2.resize(roi,(10,10))
                        roismall = roismall.reshape((1,100))
                        roismall = np.float32(roismall)
                        retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
                        number = int((results[0][0]))
                        cluster_number[q] = cluster_number[q] * 10 + number
                        flags[j] = 1
        q = q + 1

print 'cluster_number: ' + str(cluster_number)

################## Get only circuit(the largest contours), no numbers ######################

#image,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key = cv2.contourArea, reverse = True)

for c in contours:
    if cv2.contourArea(c) > 10000: 
        cv2.drawContours(circuit, [c], 0, (255, 255, 255), 2)

gray_circuit = cv2.cvtColor(circuit, cv2.COLOR_BGR2GRAY)

####################### Extracting vertical and horizontal lines ##############

# Create the images that will use to extract the horizontal and vertical lines
horizontal = np.copy(gray_circuit)
vertical = np.copy(gray_circuit)

height = np.size(gray_circuit, 0)
width = np.size(gray_circuit, 1)

# Specify size on horizontal axis
horizontalsize = width / 10

# Create structure element for extracting horizontal lines through morphology operations
horizontalStructure= cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))

# Apply morphology operations
horizontal = cv2.erode(horizontal, horizontalStructure, iterations = 1)
horizontal = cv2.dilate(horizontal, horizontalStructure, iterations = 1)

# Specify size on vertical axis
verticalsize = height / 8

# Create structure element for extracting vertical lines through morphology operations
verticalStructure= cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

# Apply morphology operations
vertical = cv2.erode(vertical, verticalStructure, iterations = 1)
vertical = cv2.dilate(vertical, verticalStructure, iterations = 1)

v_h = vertical + horizontal

##################### Corner Detection (good feature method) ##############
node_x = []
node_y = []

end_points = np.float32(v_h)

corners = cv2.goodFeaturesToTrack(end_points, 20, 0.15, 45)
corners = np.int0(corners)

for corner in corners:
    x,y = corner.ravel()
    node_x.append(x)
    node_y.append(y)
    cv2.circle(img,(x,y),5,[0,0,255],-1)

print 'node_x: ' +str(node_x)
print 'node_y: ' +str(node_y)

################### Get only elements (voltage source and resistors) ####################

d_v_h = cv2.dilate(v_h,kernel,iterations = 2)

element1 = gray_circuit - d_v_h
element2 = cv2.dilate(element1, kernel, iterations = 1)
element = cv2.adaptiveThreshold(element2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

############################ Find elements in contours #######################################
image, contours, hierarchy = cv2.findContours(element.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#Sort all the contours based on the area and display only first 10
contours = sorted(contours, key = cv2.contourArea, reverse = True)[1:10]

center_r = np.empty(shape=[0, 3])
center_v = np.empty(shape=[0, 3])

r_v = np.zeros(img.shape, np.uint8)

#Loop over our sorted contours
for c in contours:
    if (cv2.contourArea(c) > 700):
        (xr,yr),radius = cv2.minEnclosingCircle(c)
        center = (int(xr),int(yr))
        radius = int(radius)
        x,y,w,h = cv2.boundingRect(c)
        if (np.absolute(w-h) < 20):
            sum = np.sum(np.multiply(np.matrix((center_v[:, 0] == center[0])*1),np.matrix((center_v[:, 1] == center[1])*1)))
            if sum ==0:
                center = (int(xr),int(yr),2)
                center_v = np.vstack([center_v,center])
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0, 255),2)
                print 'voltage'
        else:
            sum = np.sum(np.multiply(np.matrix((center_r[:, 0] == center[0])*1),np.matrix((center_r[:, 1] == center[1])*1)))
            if sum == 0:
                center = (int(xr),int(yr),1)
                center_r = np.vstack([center_r,center])
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                print 'resistor'

center_rv = np.vstack([center_r,center_v])
center_rv = center_rv[np.argsort(center_rv[:, 0])]
sum_xy = np.add(np.multiply(node_x,5),node_y)
print sum_xy
node_xy = np.vstack([node_x,node_y,sum_xy])
node_xy = node_xy.T
node_xy = node_xy[np.argsort(node_xy[:, 2])]
print node_xy
node_x = node_xy[:,0]
node_y = node_xy[:,1]
print center_rv
print center_v
print node_x
print node_y

ln = len(node_y)

#To Check whether nodes lie on horizontal or vertical line
v_tag = [0]*ln
h_tag = [0]*ln


for i in range (0,ln):
    calc_h = np.sum(d_v_h[node_y[i]-5:node_y[i]+5, node_x[i]+20])
    calc_h = calc_h + np.sum(d_v_h[node_y[i]-5:node_y[i]+5, node_x[i]-20])
    calc_v = np.sum(d_v_h[node_y[i]+10, node_x[i]-5:node_x[i]+5])
    calc_v = calc_v + np.sum(d_v_h[node_y[i]-10, node_x[i]-5:node_x[i]+5])
    
    if calc_h > 0:
        h_tag[i]=1
    if calc_v > 0:
        v_tag[i] = 1

lcv = len(center_rv)
ab = np.zeros((lcv,4))

for i in range(0,lcv):
    ab[i,2] = center_rv[i,2]
    for j in range(0,ln):
        xd = center_rv[i][0]-node_x[j]
        yd = center_rv[i][1]-node_y[j]

        if (abs(yd) <=15):
            if (xd>0):
                ab[i][0]=j
            if xd <0:
                ab[i][1]=j
                break

        if abs(xd)<=15:
            if yd>0:
                ab[i][0]=j
            if yd<0:
                ab[i][1]=j
                break

#To find the nodes between which zero voltage sources are connected
for i in range(0,ln-1):
    for j in range(i+1,ln):
        if h_tag[i]==1 and h_tag[j]==1:
            y_d = abs(node_y[i]- node_y[j])
            if y_d<10:
                if np.sum(np.multiply(np.matrix((ab[:, 0] == i)*1),np.matrix((ab[:, 1] == j)*1))) >0:
                    break
                else:
                    ab = np.vstack([ab, [i,j,0,0]])
                    break


for i in range(0,ln-1):
    for j in range(i+1,ln):
        if v_tag[i]==1 and v_tag[j]==1:
            x_d = abs(node_x[i]-node_x[j])
            if x_d<10:
                if np.sum(np.multiply(np.matrix((ab[:, 0] == i)*1),np.matrix((ab[:, 1] == j)*1))) >0:
                    break
                else:
                    ab = np.vstack([ab, [i,j,0,0]]) 
                    break
print ab
print ln
print v_tag
print h_tag

lab = len(ab)
net_1 = np.empty(shape=[0, 4])
for i in range(0,lab):
    sum_ab = np.sum(np.multiply(np.matrix((net_1[:, 0] == ab[i,0])*1),np.matrix((net_1[:, 1] == ab[i,1])*1)))
    if sum_ab ==0:
            net_1 = np.vstack([net_1,ab[i]])

print net_1
lnet = len(net_1)
for i in range (0,lnet):
    net_1[i,3] = cluster_number[i]
print net_1

file = open("pi2pc.txt", "w")
file.write(str(net_1) + "\n")
file.close()

cv2.imwrite('img.jpg',img)
#################################################################################
# Pi
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# titles = ['img', 'gray', 'gaussian', 'invert', 'filter(img)=erosion', 'd(erosion)=bw', 'gray_circuit', 'v_h', 'd_v_h', 'g_cir-d_v_h=element1', 'element2','element', 'digits']
# images = [img, gray, gaussian, invert, erosion, bw, gray_circuit, v_h, d_v_h, element1, element2, element, digits]

# for i in xrange(13):
#     plt.subplot(4,4,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])

# plt.show()
