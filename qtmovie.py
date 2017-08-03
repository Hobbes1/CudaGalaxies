import sys 
from pyqtgraph.Qt import QtCore, QtGui
from itertools import islice 
import pyqtgraph.opengl as gl
import numpy as np



### Load data depending on the input param ###

params = np.genfromtxt("data/"+sys.argv[1]+"Params", dtype=None)
print(params.shape)
print(params[0])
NPoints = int(params[0][0])
NSteps = int(params[1][0])
dt = float(params[2][0])
lines = NPoints*NSteps

data = np.array([1])
n = 0
with open("data/"+sys.argv[1]+'Run.data') as infile:
	while True:
		gen = islice(infile, NPoints)
		arr = np.genfromtxt(gen)
		data = np.append(data, arr)
		sys.stdout.write("\r	Loading: " + str(100.0*n/NSteps) + "% ")
		sys.stdout.flush()
		n+=1
		if n == NSteps:
			break

data = np.delete(data, 0, 0)
print(data)
print(data.shape)
data.shape = (lines, 3)
print(data.shape)

### Establish QtGUI and openGL view, TODO initial zoom values ###

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.resize(1600,1000)
w.setCameraPosition(distance=12, elevation=30)
w.pan(0,0,0)
w.show()

ax = gl.GLAxisItem()

ax.setSize(x=2, y=2, z=2)
w.addItem(ax)
#w.addItem(g)


start = 0
end = 0
record = False

if sys.argv[2] == 'record':
	record = True

start = 0
end = 3

colors = [] 
sizes = []
for i in range(NPoints):
	'''
	colors.append([1.0,0.6,0.3,0.9])
	sizes.append(3)
	'''
	if i < 4000: 
		colors.append([1.0, 0.65, 0.2, 1.0])
		sizes.append(3)
	if i >= 4000 and i < 6000:
		colors.append([1.0, 0.3, 1.0, 0.8])
		sizes.append(2)
	if i >= 6000 and i < 16384: 
		colors.append([0.3, 0.4, 1.0, 0.7])
		sizes.append(2)
	if i >= 16384 and i < 20384:
		colors.append([1.0, 0.65, 0.2, 1.0])
		sizes.append(3)
	if i >= 20384 and i < 22384:
		colors.append([1.0, 0.3, 1.0, 0.8])
		sizes.append(2)
	if i >= 22384:
		colors.append([0.3, 0.4, 1.0, 0.4])
		sizes.append(2)
colors = np.array(colors).reshape(NPoints,4)
sizes = np.array(sizes)
	
print("creating original scatter")
sp2 = gl.GLScatterPlotItem(pos=data[0:NPoints, start:end], size=sizes, color=colors, pxMode=True)
w.addItem(sp2)

total = data.shape[0]

frames = total//NPoints

print("num frames: " + str(frames))
i = 1

def update():
    ## update volume colors
	#w.orbit(360.0/frames, 0)
	global i, sizes, colors
	#w.setCameraPosition(distance=12+80*i/frames)
	if record == True:
		name = 'frame_' + str(i).zfill(4) + '.png'
		w.grabFrameBuffer().save(name)

	sp2.setData(pos=data[i*NPoints:i*NPoints+NPoints, start:end], size=sizes, color=colors, pxMode=True)
	if i >= frames and record == True: exit()
	if i >= frames and record == False: i = 0
	else: i+=1

t = QtCore.QTimer()
t.timeout.connect(update)
t.start(20)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, PYQT_VERSION):
        QtGui.QApplication.instance().exec_()
