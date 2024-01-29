import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np
#hard code the network and optimize your parameters
#next, make a node function that includes the function, weights and biases, and optimizes them inside
def softplus(x):
    return np.log(1 + np.exp(x))
def sigmoid(x):
    return np.exp(x)/(1+np.exp(x))
def processPoints(points):
    xP = []
    yP = []
    for p in points:
        xP.append(p[0])
        yP.append(p[1])
    return [xP, yP]

fig, ax = plt.subplots()
points = [[0,0],[0.5,1],[1,0]]

#MUST GENERATE THROUGH STD DEV CURVE
w1 = 0.1
b1 = 0
w3 = 0.36
w2 = -1.13
b2 = 0
w4 = 0.63 
b3 = 0

dw1 = 0
dw2 = 0
dw3 = 0
dw4 = 0 
db1 = 0
db2 = 0
db3 = 0

predictArr = []

epochs = 0
L = 0.001

for p in points:        #INITIAL NETWORK RUN TO OBTAIN PREDICTED VALUES
    inpInit = p[0]
    ans1=softplus(inpInit*w1+b1)*w3
    ans2=softplus(inpInit*w2+b2)*w4+b3
    predictArr.append(ans1+ans2)

while epochs < 20:     #GRADIENT DESCENT TO OPTIMIZE PARAMETERS
    
    for i in range(len(points)):
        observed = points[i][1]     #observed y coordinate
        inp = points[i][0]          #input x coordinate
        predicted = predictArr[i]   #prediction corresponding to point

        dw1 += (-2*(observed-predicted)*sigmoid(inp*w1+b1)*w3*inp)

    

    w1 -= L

    epochs+=1

inpInit = 0.5
ans=0
ans+=softplus(inpInit*w1+b1)*w3
ans+=softplus(inpInit*w2+b2)*w4+b3
print(ans)

xP, yP = processPoints(points)
plt.xlim(0, max(xP))
plt.ylim(0, max(yP)+max(yP)/2)
plt.grid()
plt.plot(xP, yP, 's')
plt.show()