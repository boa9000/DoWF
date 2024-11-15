from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

# Open the image
im = Image.open("bav.jpg")
im = im.resize((300, 300))
# Convert the image to a NumPy array
data = np.array(im)

# Create a mask for black pixels
black_mask = ((data[:, :, 0] < 40) & (data[:, :, 1] < 40) & (data[:, :, 2] < 40))

# Convert black pixels to 1 and anything else to 0
binary_array = black_mask.astype(int)


# Plot the binary 2D array
plt.imshow(binary_array, cmap='binary', interpolation='nearest', alpha=0.6)
plt.show()

np.save('bavConst.npy', binary_array)

##############################################################################################

'''from floris.tools import FlorisInterface, WindRose

numberOfTurbines = 5
initialPos = np.zeros((numberOfTurbines, 2))
farmLayer = np.zeros_like(binary_array)

for i in range(len(initialPos)):
    pos = np.random.randint(300, size=2)
    while (binary_array[pos[0]][pos[1]] == 1):
        pos = np.random.randint(300, size=2)
    initialPos[i] = pos

initialPos = initialPos.astype(int)

for i in initialPos:
    farmLayer[i[0]][i[1]] = 1

print(len(initialPos) == len(np.unique(initialPos, axis=0)))
plt.imshow(farmLayer, cmap='binary', interpolation='nearest', alpha=0.7)
plt.show()

wr = WindRose()

rose = wr.read_wind_rose_csv(filename='windrose.csv')

fi = FlorisInterface("gch.yaml")

turbPos = initialPos.copy()

xlay = turbPos[:,0]*33.33
ylay = turbPos[:,1]*33.33

fi.reinitialize(
    layout_x=xlay,
    layout_y=ylay,
)



# Counters to keep track of accepted/rejected moves and path length
acc = 0
rej = 0
accDel = 0
aepTracker = []
aep = fi.get_farm_AEP_wind_rose_class(wind_rose=wr)
aepTracker.append(aep)

print(aep)

T0 = 1.0
iter_max=10000
tim = time.time()
# Array to keep track of the cost as we iterate
T = T0
# Initiate iterative steps
#plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.imshow(binary_array, cmap='binary', interpolation='nearest', alpha=0.6)
ax1.imshow(farmLayer, cmap='Greens', interpolation='nearest', alpha=0.7)
#plt.show(block=False)
#plt.pause(1)
for i in range(1, iter_max + 1):
    farmLayer = np.zeros_like(binary_array)
    # Annealing (reduce temperature, cool down with steps)
    # Here we decrease the temperature every 100
    turbPosNew = turbPos.copy()
    #if i % 1 == 0:
    T = 0.999 * T
    # Generate two random indices and switch them (switch cities)
    # We keep first and last cities fixed (note they are also the same
    # since the path must be closed)
    pos = turbPosNew[i%numberOfTurbines] + np.random.randint(-10, 11, size=2)
    while (binary_array[pos[0]][pos[1]] == 1):
        pos = turbPosNew[i%numberOfTurbines] + np.random.randint(-10, 11, size=2)

    turbPosNew[i%numberOfTurbines] = pos

    xlay = turbPosNew[:, 0]*33.33
    ylay = turbPosNew[:, 1]*33.33

    fi.reinitialize(
        layout_x=xlay,
        layout_y=ylay,
    )

    aepNew = fi.get_farm_AEP_wind_rose_class(wind_rose=wr)


    for t in turbPosNew:
        farmLayer[t[0]][t[1]] = 1

    plt.pause(0.001)

    # Calculate length of new state (path)
    deltaAEP = aep - aepNew
    if i % 25 == 0:
        cps = 25/(time.time() - tim)
        tim = time.time()
        print("i: "+ str(i) +" T: "+ str(T) +" Accepted: " + str(acc) + " by T: "+ str(accDel)+ " Rejected: " + str(rej))
        print("CPS: {:.2f}".format(cps))
        print("maxAEP: {:.2f}".format(max(aepTracker)))
        #plt.cla()
        ax1.imshow(binary_array, cmap='binary', interpolation='nearest', alpha=0.6)
        ax1.imshow(farmLayer, cmap='Greens', interpolation='nearest', alpha=0.7)
        ax2.set_title('Number of iterations: ' + str(i))
        ax2.plot(np.arange(len(aepTracker)), aepTracker)
        plt.show(block=False)
        plt.pause(0.001)

    # Accept the move if new path is shorter than previous one
    if deltaAEP <= 0.0:
        turbPos = turbPosNew.copy()
        aep = aepNew
        acc += 1
        aepTracker.append(aepNew)

    # If new path is larger than previous, accept the move with probability
    # proportional to the Boltzmann factor
    elif deltaAEP > 0.0:
        u = np.random.uniform()
        if u < np.exp(- ((deltaAEP/1e8)) / T):
            turbPos = turbPosNew.copy()
            aep = aepNew
            acc += 1
            accDel += 1
            aepTracker.append(aepNew)

        else:
            rej += 1




plt.ioff()
plt.show()'''