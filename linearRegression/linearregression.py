import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.transforms as transforms

x = []
y = []

m = 5
b = 0
x1 = 0
x2 = 20
linestart = [x1, x1*m+b]
lineend = [x2, x2*m+b]
fig, ax = plt.subplots()
line = lines.Line2D([linestart[0],lineend[0]], [linestart[1],lineend[1]], color='red')

points = [[0,3],[3,4],[6,5],[9,7],[13,8],[16,8],[19,10]]

n = len(points)

for p in points:
    x.append(p[0])
    y.append(p[1])

iter = 0
pderivM = 0
pderivB = 0
epochs = 0
L = 0.0001
while epochs < 50000:
    print(f'EPOCH: {epochs}')
    pderivM = 0
    pderivB = 0
    #MAJOR MISTAKE: I FORGOT TO RESET THE PARTIAL DERIVATIVES AFTER EACH ITERATION
    for p in points:
        pderivM += 2*p[0]*((m*p[0]+b)-p[1])
        pderivB += 2*((m*p[0]+b)-p[1])
    m-=L*pderivM
    b-=L*pderivB
    epochs+=1

linestart2 = [x1, x1*m+b]
lineend2 = [x2, x2*m+b]
line2 = lines.Line2D([linestart2[0],lineend2[0]], [linestart2[1],lineend2[1]], color='blue')

print(f'{linestart2[0]},  {linestart2[1]}')
print(f'{lineend2[0]},  {lineend2[1]}')

plt.xlim(0, 20)
plt.ylim(0, 20)
plt.grid()
plt.plot(x, y, 's')
ax.add_line(line)
ax.add_line(line2)
plt.show()