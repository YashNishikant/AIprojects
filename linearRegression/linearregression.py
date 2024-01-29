import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np
fig, ax = plt.subplots()

points = [[0,3],[3,9],[6,5],[9,7],[13,8*2],[16,8*2],[19,10*2]]
#Given Points

def regression(points, typeFun):
    
    xP = []
    for p in points:
        xP.append(p[0])

    start_points = 0
    end_points = max(xP)

    iter = 1000
    pderivM = 0
    pderivB = 0
    epochs = 0
    m = 1
    b = 1
    L = 0.0001

    x = np.linspace(start_points, end_points, iter) 
    y = None
    
    while epochs < 50000:
        pderivM = 0
        pderivB = 0
        for p in points:
            if typeFun == 'linear':
                pderivM += 2*p[0]*((m*p[0]+b)-p[1])
                pderivB += 2*((m*p[0]+b)-p[1])
            elif typeFun == 'softplus':
                pderivM += 2*p[0]*(m*np.log(1 + np.exp(p[0]))+b-p[1])
                pderivB += 2*(m*np.log(1 + np.exp(p[0]))+b-p[1])

        m-=L*pderivM
        b-=L*pderivB
        epochs+=1

    if typeFun == 'linear':
        y = m*x+b
    elif typeFun == 'softplus':
        y = m*np.log(1 + np.exp(x))+b

    return [x,y]
#Node Function

def processPoints(points):
    xP = []
    yP = []
    for p in points:
        xP.append(p[0])
        yP.append(p[1])
    return [xP, yP]
#Obtaining points to graph

xP, yP = processPoints(points)
x, y = regression(points, 'softplus')
#Returns X and Y coordinate outputs to graph

plt.xlim(0, max(xP))
plt.ylim(0, max(yP)+10)
plt.grid()
plt.plot(xP, yP, 's')
plt.plot(x,y)
plt.show()