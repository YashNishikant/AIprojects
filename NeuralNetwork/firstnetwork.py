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
def model(points):
    #MUST GENERATE THROUGH STD DEV CURVE
    w1 = 2.74
    b1 = 0
    w3 = 0.36
    w2 = -1.13
    b2 = 0
    w4 = 0.63 
    b3 = 0
    epochs = 0
    while epochs < 100000:     #GRADIENT DESCENT TO OPTIMIZE PARAMETERS
        L = 0.001
        predictArr = []
        for p in points:       #GENERATING NEW PREDICTIONS EACH EPOCH TO CALCULATE AND MINIMIZE ERROR THROUGH DERIVATIVES
            inpInit = p[0]
            ans1=softplus(inpInit*w1+b1)*w3
            ans2=softplus(inpInit*w2+b2)*w4+b3
            predictArr.append(ans1+ans2)
        dw1 = 0                 #RESET ALL DERIVATIVES
        dw2 = 0
        dw3 = 0
        dw4 = 0 
        db1 = 0
        db2 = 0
        db3 = 0
        for i in range(len(points)):
            
            observed = points[i][1]     #observed y coordinate
            inp = points[i][0]          #input x coordinate
            predicted = predictArr[i]   #prediction corresponding to point

            dw1 += (-2*(observed-predicted)*sigmoid(inp*w1+b1)*w3*inp)  #Derivative list
            dw2 += (-2*(observed-predicted)*sigmoid(inp*w2+b2)*w4*inp)
            dw3 += (-2*(observed-predicted)*softplus(inp*w1+b1))
            dw4 += (-2*(observed-predicted)*softplus(inp*w2+b2))
            db1 += (-2*(observed-predicted)*sigmoid(inp*w1+b1)*w3)
            db2 += (-2*(observed-predicted)*sigmoid(inp*w2+b2)*w4)
            db3 += (-2*(observed-predicted))

            print(f"Iteration: {epochs}/100000")

        w1 -= L*dw1     #Gradient descent optimization
        w2 -= L*dw2
        w3 -= L*dw3
        w4 -= L*dw4
        b1 -= L*db1
        b2 -= L*db2
        b3 -= L*db3 

        epochs+=1
    print("DONE")
    return [w1,w2,w3,w4,b1,b2,b3]
def predict(singularpointX):
    w1,w2,w3,w4,b1,b2,b3 = model(points)
    return softplus(singularpointX*w1+b1)*w3+softplus(singularpointX*w2+b2)*w4+b3
def predict(testinginput):
    w1,w2,w3,w4,b1,b2,b3 = model(points)
    resultArr = []
    for testx in testinginput:
        resultArr.append(softplus(testx*w1+b1)*w3+softplus(testx*w2+b2)*w4+b3)
    return resultArr

fig, ax = plt.subplots()
points = [[0,0],[0.5,1],[1,0]]

#TESTING
testinginput = [0.1,0.2,0.3,0.4,0.6,0.7,0.8]
testingresult = predict(testinginput)

xP, yP = processPoints(points)
plt.xlim(0, max(xP))
plt.ylim(0, max(yP)+max(yP)/2)
plt.grid()
ax.scatter(testinginput,testingresult,c="red")
ax.scatter(xP, yP, c="blue")
plt.show()