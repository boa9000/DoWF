import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import time
from floris.tools import FlorisInterface, WindRose
import pandas as pd
import sys
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree


# Initializes the first positions for the turbines.
def initializer(mapArr, nTur):
    iPos = np.zeros((nTur, 2))
    for i in range(len(iPos)):
        pos = np.random.randint(300, size=2)
        while mapArr[pos[0]][pos[1]] == 1:
            pos = np.random.randint(300, size=2)
        iPos[i] = pos
    iPos = iPos.astype(int)
    return iPos

# shows the locations of the turbines on the 300x300 grid map, in case it needs to be shown on a plot
def farmAllocator(Positions, mapArr):
    farmLayout = np.zeros_like(mapArr)
    for i in Positions:
        farmLayout[i[0]][i[1]] = 1
    return farmLayout

# allocates a turbine on the map randomly within a radius.
def allocateTurbine(mapArr, position,tlocations, R):
    pos = position + np.random.randint(-R, (R+1), size=2)
    locations = np.array(tlocations)
    while True:
        if 0 <= pos[0] < 300 and 0 <= pos[1] < 300:
            if (mapArr[pos[0]][pos[1]] == 1) or (pos == locations).all(axis=1).any():
                pos = position + np.random.randint(-R, (R + 1), size=2)
            else:
                return pos
        else:
            pos = position + np.random.randint(-R, (R + 1), size=2)


# calculates the AEP of the farm at the specified position
def getAEP(PosOfTurbs, Scale, WindRose):
    xlay = PosOfTurbs[:, 1]*Scale
    ylay = -PosOfTurbs[:, 0]*Scale
    fi.reinitialize(layout_x=xlay,layout_y=ylay,)
    AEP = fi.get_farm_AEP_wind_rose_class(wind_rose=WindRose)
    return AEP

# calculates the LCOe for that specific AEP and position
def getLCOE(AEP, LengthOfCables):
    CablesCost = LengthOfCables*CostPerKM
    InitialInvestment = (LandBossPerKW + TurbineCost + CablesCost) * ProjectkW
    LCOE = InitialInvestment / ((AEP/1000) * A) + (YearlyCosts) / (AEP/1000) + OnM
    return LCOE

# calculates the cables length and the midpoint of all the turbines (where the substation is supposed to be)
def getCablesLengthAndSubstation(Positions, Scale):
    midPoint = np.round(sum(Positions)/len(Positions))
    midPoint = midPoint.astype(int)
    posandmid = np.vstack([Positions, midPoint]) * Scale
    distances = squareform(pdist(posandmid))
    mst = minimum_spanning_tree(distances)
    return mst, midPoint


wr = WindRose()


singleRun = True  # True in case it is a single run (used to further optimize a chosen wind turbine type in a region)
singleRunNoOfTurbines = 15 # decide how many Turbines for that wind turbine type to run a single run

# should include the price lists of the turbine types in the respective region
priceslist = [ 'prices_BAU_LSP_bav.csv', 'prices_IEA_kast.csv', 'prices_BAU_IEA_kast.csv', 'prices_BAU_LSP_kast.csv']


# the main loops of the script
# it should loop through each of the price lists that was defined above
for prices in priceslist:
    # reads the price list of the specific turbine in the respective region
    turbCosts = pd.read_csv(prices, index_col=0)

    # this loop would run the simulated annealing algorithm for each number of turbines, ranging from 2 to 15 turbines
    for numberOfTurbines in turbCosts.index.tolist():
        # in case it is for a single run
        if singleRun:
            numberOfTurbines = singleRunNoOfTurbines
        ################# Start of setting costs ##################
        # Prices for each region and type of turbine
        # this will also initiate which parameters floris should use
        # parameters such as wind shear, turbulence intensity, scale, and which binary_array of the map

        if prices == 'prices_IEA_kast.csv':
            CostPerKM = turbCosts.loc[numberOfTurbines, 'price_per_km']
            LandBossPerKW = turbCosts.loc[numberOfTurbines, 'price_per_kw']
            TurbineCost = 1125  # Turbine Cost per kW
            OnM = 0.012  # O&M costs per kWh
            Rental = 15  # Rental costs per kW a year
            RealDiscount = 0.04  # Real discount rate
            LifeTime = 20  # lifetime of the project in years
            ProjectkW = 3370 * numberOfTurbines
            fi = FlorisInterface("gch1.yaml")
            fi.reinitialize(wind_shear=0.17, turbulence_intensity=0.12)
            rose = wr.read_wind_rose_csv(filename='windrose_den_test.csv')
            binary_array = np.load('kastConst.npy')
            scale = 33

        if prices == 'prices_BAU_IEA_kast.csv':
            CostPerKM = turbCosts.loc[numberOfTurbines, 'price_per_km']
            LandBossPerKW = turbCosts.loc[numberOfTurbines, 'price_per_kw']
            TurbineCost = 1300  # Turbine Cost per kW
            OnM = 0.012  # O&M costs per kWh
            Rental = 15  # Rental costs per kW a year
            RealDiscount = 0.04  # Real discount rate
            LifeTime = 20  # lifetime of the project in years
            ProjectkW = 3300 * numberOfTurbines
            fi = FlorisInterface("gch2.yaml")
            fi.reinitialize(wind_shear=0.17, turbulence_intensity=0.12)
            rose = wr.read_wind_rose_csv(filename='windrose_den_test.csv')
            binary_array = np.load('kastConst.npy')
            scale = 33

        if prices == 'prices_BAU_LSP_kast.csv':
            CostPerKM = turbCosts.loc[numberOfTurbines, 'price_per_km']
            LandBossPerKW = turbCosts.loc[numberOfTurbines, 'price_per_kw']
            TurbineCost = 1400  # Turbine Cost per kW
            OnM = 0.012  # O&M costs per kWh
            Rental = 15  # Rental costs per kW a year
            RealDiscount = 0.04  # Real discount rate
            LifeTime = 20  # lifetime of the project in years
            ProjectkW = 3250 * numberOfTurbines
            fi = FlorisInterface("gch3.yaml")
            fi.reinitialize(wind_shear=0.17, turbulence_intensity=0.12)
            rose = wr.read_wind_rose_csv(filename='windrose_den_test.csv')
            binary_array = np.load('kastConst.npy')
            scale = 33

        if prices == 'prices_IEA_bav.csv':
            CostPerKM = turbCosts.loc[numberOfTurbines, 'price_per_km']
            LandBossPerKW = turbCosts.loc[numberOfTurbines, 'price_per_kw']
            TurbineCost = 1125  # Turbine Cost per kW
            OnM = 0.012  # O&M costs per kWh
            Rental = 20  # Rental costs per kW a year
            RealDiscount = 0.035  # Real discount rate
            LifeTime = 20  # lifetime of the project in years
            ProjectkW = 3370 * numberOfTurbines
            fi = FlorisInterface("gch1.yaml")
            fi.reinitialize(wind_shear=0.20, turbulence_intensity=0.18)
            rose = wr.read_wind_rose_csv(filename='windrose_bav_test.csv')
            binary_array = np.load('bavConst.npy')
            scale = 21

        if prices == 'prices_BAU_IEA_bav.csv':
            CostPerKM = turbCosts.loc[numberOfTurbines, 'price_per_km']
            LandBossPerKW = turbCosts.loc[numberOfTurbines, 'price_per_kw']
            TurbineCost = 1300  # Turbine Cost per kW
            OnM = 0.012  # O&M costs per kWh
            Rental = 20  # Rental costs per kW a year
            RealDiscount = 0.035  # Real discount rate
            LifeTime = 20  # lifetime of the project in years
            ProjectkW = 3300 * numberOfTurbines
            fi = FlorisInterface("gch2.yaml")
            fi.reinitialize(wind_shear=0.20, turbulence_intensity=0.18)
            rose = wr.read_wind_rose_csv(filename='windrose_bav_test.csv')
            binary_array = np.load('bavConst.npy')
            scale = 21

        if prices == 'prices_BAU_LSP_bav.csv':
            CostPerKM = turbCosts.loc[numberOfTurbines, 'price_per_km']
            LandBossPerKW = turbCosts.loc[numberOfTurbines, 'price_per_kw']
            TurbineCost = 1400  # Turbine Cost per kW
            OnM = 0.012  # O&M costs per kWh
            Rental = 20  # Rental costs per kW a year
            RealDiscount = 0.035  # Real discount rate
            LifeTime = 20  # lifetime of the project in years
            ProjectkW = 3250 * numberOfTurbines
            fi = FlorisInterface("gch3.yaml")
            fi.reinitialize(wind_shear=0.20, turbulence_intensity=0.18)
            rose = wr.read_wind_rose_csv(filename='windrose_bav_test.csv')
            binary_array = np.load('bavConst.npy')
            scale = 21

        # calculates the A which is needed for the LCOE calculation
        A = (((1+RealDiscount)**LifeTime)-1)/(RealDiscount*(1+RealDiscount)**LifeTime)
        YearlyCosts = (Rental*ProjectkW)
        ################ END OF SETTING COSTS ########################

        ################ Initializing the first position and setting counters ########################
        # initialize the positions of the turbines
        turbPos = initializer(binary_array, numberOfTurbines)

        # Counters to keep track of many variables
        acc = 0 # how many moves have been accepted
        rej = 0 # how many moves have been rejected
        accDel = 0  # how many moved have been accepted through "chance"
        aepatmini = 0   # the AEP at the minimum LCOE position
        aepTracker = [] # tracks all the AEPs for the accepted moves
        lcoeTracker = []    # tracks all the LCOEs of the accepted moves

        aep = getAEP(turbPos, scale, wr) # calls the function getAEP to get the AEP
        aepTracker.append(aep)
        lengths, substationPos = getCablesLengthAndSubstation(turbPos, scale)   # calls the getCablesLengthAndSubstation function to get the lengths of the cables
        cablesLength = np.sum(lengths.toarray()) / 1000 # sums the cables lengths and converts them to km
        lcoe = getLCOE(aep, cablesLength)   # calls the function getLCOE to get the LCOE
        lcoeTracker.append(lcoe)

        mini = 1 # this would be the minimum LCOE discovered
        maxy = 0 # this would be the maximum AEP discovered
        miniPos = turbPos.copy() # miniPos would be the position of turbines for the minimum LCOE
        miniPosTracker = [] # Tracks the positions of all the minimum LCOEs obtained
        maxPos = turbPos.copy() # position of turbines with the amximum AEP discovered
        minChanges = 0  # how many times the min LCOE changed
        maxChanges = 0  # how many times the max AEP changed

        R = 50 # the set radius for the turbine movement

        iter_max=5000   # the number of intended iterations of the SA algorithm
        tim = time.time() # captures the time for time measuring
        T = lcoe/50 # the initial temperature of the simulated annealing "system"

        ################ END OF initializing the first position and setting counters ########################

        ############### START of the main loop for Simulated Annealing ######################
        for i in range(1, iter_max + 1):

            turbPosNew = turbPos.copy() # sets the new position of turbines as the older one

            T = 0.999 * T # cools the system's temperature

            # this would decrease the search radius by 1 every 1/50th of the total iterations period
            if i % int(iter_max/50) == 0 and R > 7:
                R -= 1

            # once the iterations reach (0.8 * iter_max), start looking around the minimum LCOE position found
            if i == int(iter_max*0.8):
                turbPosNew = miniPos.copy()
                lcoe = mini.copy()

            # selects a turbine and then calls the allocateTurbine function to move it to a new position
            turbPosNew[i%numberOfTurbines] = allocateTurbine(binary_array, turbPosNew[i%numberOfTurbines], turbPosNew, R)

            # calculates the AEP of the new position
            aepNew = getAEP(turbPosNew,scale,wr)

            # farmLayer would be the location of the turbines in the 300x300 grid
            farmLayer = farmAllocator(turbPosNew,binary_array)

            # calculates cables lengths, lcoe and the position of the substation
            lengths, substationPos = getCablesLengthAndSubstation(turbPosNew, scale)
            cablesLength = np.sum(lengths.toarray()) / 1000
            lcoeNew = getLCOE(aepNew, cablesLength)
            # calculates the difference between the new LCOE and the previous one
            deltaLCOE = lcoeNew - lcoe



            # if the new AEP is higher than the max AEP discovered, save it as the new max AEP along with positions
            if aepNew > maxy:
                maxy = aepNew.copy()
                maxPos = turbPosNew.copy()
                maxChanges += 1

            # Accept the move if the new LCOE is smaller than the previous one
            if deltaLCOE <= 0.0:
                turbPos = turbPosNew.copy()
                lcoe = lcoeNew
                acc += 1
                aepTracker.append(aepNew)
                lcoeTracker.append(lcoe)

                # if the new LCOE is the smaller than smallest LCOE discovered, save it with positions
                if lcoe < mini:
                    mini = lcoe.copy()
                    miniPos = turbPos.copy()
                    aepatmini = aepNew
                    minChanges += 1
                    miniPosTracker.append(miniPos)


            # if the new LCOE is larger, accept with a chance proportional to the Boltzmann factor
            elif deltaLCOE > 0.0:
                u = np.random.uniform()
                if u < np.exp(- (deltaLCOE) / T):
                    turbPos = turbPosNew.copy()
                    lcoe = lcoeNew
                    acc += 1
                    accDel += 1
                    aepTracker.append(aepNew)
                    lcoeTracker.append(lcoe)
                # if not accepted, turbPosNew wont take palce of turbPos, and the rejection counter increases
                else:
                    rej += 1

            # every 100 iterations, display and SAVE all the important parameters discovered such as:
            # how many iterations, the temperature, how many accepted, accepted with chance, and rejected positions
            # iterations per second, max AEPm times the max AEP changed, AEP at min LCOE, min LCOE, times it changed
            if i % 100 == 0:
                cps = 100 / (time.time() - tim) # calculations per second (iterations per second)
                tim = time.time()
                print("i: " + str(i) + " T: " + str(T) + " Accepted: " + str(acc) + " by T: " + str(accDel) + " Rejected: " + str(
                    rej))
                print("CPS: {:.2f}".format(cps))
                print("maxAEP: {:.2f}".format(max(aepTracker)))
                print("times maxAEP changed: {:d}".format(maxChanges))
                print("AEP@mini: {:.10f}".format(aepatmini))
                print("minLCOE: {:.10f}".format(mini))
                print("times minLCOE changed: {:d}".format(minChanges))

                # creates a dataframe with the AEP and LCOE as columns
                df = pd.DataFrame()
                df['AEP'] = aepTracker
                df['LCOE'] = lcoeTracker
                df.to_csv('output/{:s}_{:d}_turbines.csv'.format(prices, numberOfTurbines)) # saves the dataframe
                del df  # deletes the dataframe for memory saving

                # saves the arrays of the minimum LCOE positions
                np.save('output/{:s}_{:d}_turbines.npy'.format(prices, numberOfTurbines), miniPosTracker)

                # saves all the improtant parameters such as the ones that were displayed
                # but also saves the arrays of the wind turbine positions (in the 300x300 coordinates)
                with open('output/{:s}_{:d}_turbines.txt'.format(prices, numberOfTurbines), 'w') as f:
                    f.write(
                        "i: " + str(i) + " T: " + str(T) + " Accepted: " + str(acc) + " by T: " + str(accDel) + " Rejected: " + str(
                            rej) +'\n')
                    f.write("CPS: {:.2f}\n".format(cps))
                    f.write("maxAEP: {:.2f}\n".format(max(aepTracker)))
                    f.write("times maxAEP changed: {:d}\n".format(maxChanges))
                    f.write("maxAEP Pos:\n" + str(repr(maxPos)))
                    f.write("\nAEP@mini: {:.2f}\n".format(aepatmini))
                    f.write("minLCOE: {:.10f}\n".format(min(lcoeTracker)))
                    f.write("times minLCOE changed: {:d}\n".format(minChanges))
                    f.write("minLCOE Pos:\n" + str(repr(miniPos)))

            # in case the script is set as a single run, it would end the script if it reaches the maximum iterations
            if i == iter_max:
                if singleRun:
                    sys.exit()
