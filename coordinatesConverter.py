import numpy as np



location = 'bavaria' # bavaria or denmark
lb = False # is it for LandBOSSE? to use the first turbine as the reference.

if location == 'bavaria':
       scale = 21
       ref = [264, 50]

if location == 'denmark':
       scale = 33
       ref = [170, 17]

# put coordinates here in a np.array format
coord = np.array([[216, 134],
       [224,  49],
       [158,  208],
       [127, 276],
       [182, 140],
       [142, 161],
       [206, 263],
       [ 61, 193],
       [ 98, 105],
       [263,  78],
       [ 26, 198],
       [ 49,  67],
       [109, 176],
       [240, 181],
       [ 73,  86]])

# convert the global coordinates to the reference according to the CAD map
coord = coord - ref

coord[:, [0, 1]] = coord[:, [1, 0]]       # swaps the [-y, x] into [x, -y]
coord[:,1] = -coord[:,1]    # [x, -y] to [x, y]

coord = coord * scale       # multiply the grid coordinates to the scale for the real distances

if lb:
       coord = (coord - coord[0]) /1000   # make the first turbine the reference and convert to km (for landbosse)


midPoint = sum(coord)/len(coord)   # get the midpoint (for the substation position)

print('coord:')
print(repr(coord))
print('midPoint:')
print(repr(midPoint))