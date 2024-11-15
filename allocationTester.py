import numpy as np
import matplotlib.pyplot as plt
import time
from floris.tools import FlorisInterface, WindRose
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import floris.tools.visualization as wakeviz


def initializer(mapArr, nTur):
    iPos = np.zeros((nTur, 2))
    for i in range(len(iPos)):
        pos = np.random.randint(300, size=2)
        while mapArr[pos[0]][pos[1]] == 1:
            pos = np.random.randint(300, size=2)
        iPos[i] = pos
    iPos = iPos.astype(int)
    return iPos


def farmAllocator(Positions, mapArr):
    farmLayout = np.zeros_like(mapArr)
    for i in Positions:
        farmLayout[i[0]][i[1]] = 1
    return farmLayout


def allocateTurbine(mapArr, position, R):
    pos = position + np.random.randint(-R, (R + 1), size=2)
    while True:
        if 0 <= pos[0] < 300 and 0 <= pos[1] < 300:
            if mapArr[pos[0]][pos[1]] == 1:
                pos = position + np.random.randint(-R, (R + 1), size=2)
            else:
                return pos
        else:
            pos = position + np.random.randint(-R, (R + 1), size=2)


def getAEP(PosOfTurbs, Scale, WindRose):
    xlay = PosOfTurbs[:, 1] * Scale
    ylay = -PosOfTurbs[:, 0] * Scale
    fi.reinitialize(layout_x=xlay, layout_y=ylay, )
    AEP = fi.get_farm_AEP_wind_rose_class(wind_rose=WindRose)
    return AEP


def getLCOE(AEP, LengthOfCables):
    CablesCost = LengthOfCables * CostPerKM
    InitialInvestment = (LandBossPerKW + TurbineCost + CablesCost) * ProjectkW
    lbcosts = (LandBossPerKW + CablesCost) * ProjectkW
    LCOE = InitialInvestment / ((AEP / 1000) * A) + (YearlyCosts) / (AEP / 1000) + OnM
    return LCOE, lbcosts


def getCablesLengthAndSubstation(Positions, Scale):
    midPoint = np.round(sum(Positions)/len(Positions))
    midPoint = midPoint.astype(int)
    posandmid = np.vstack([Positions, midPoint]) * Scale
    distances = squareform(pdist(posandmid))
    mst = minimum_spanning_tree(distances)
    return mst, midPoint


wr = WindRose()

serious = True


'''array([[151,  79],
       [ 56, 180],
       [ 85, 184],
       [123, 100],
       [198, 215],
       [134, 191],
       [206,  57],
       [164, 202],
       [175,  61],
       [141, 143]]) # for denmark IEA'''

'''array([[216, 134],
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
       [ 73,  86]]) # for bavaria LSP'''



LOC = np.array([[151,  79],
       [ 56, 180],
       [ 85, 184],
       [123, 100],
       [198, 215],
       [134, 191],
       [206,  57],
       [164, 202],
       [175,  61],
       [141, 143]])


prices = 'prices_IEA_kast.csv'

turbCosts = pd.read_csv(prices, index_col=0)


numberOfTurbines = len(LOC)
########################################
# Prices
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
    rose = wr.read_wind_rose_csv(filename='windrose_den.csv')
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
    rose = wr.read_wind_rose_csv(filename='windrose_den.csv')
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
    rose = wr.read_wind_rose_csv(filename='windrose_den.csv')
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
    rose = wr.read_wind_rose_csv(filename='windrose_bav.csv')
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
    rose = wr.read_wind_rose_csv(filename='windrose_bav.csv')
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
    rose = wr.read_wind_rose_csv(filename='windrose_bav.csv')
    binary_array = np.load('bavConst.npy')
    scale = 21



A = (((1 + RealDiscount) ** LifeTime) - 1) / (RealDiscount * (1 + RealDiscount) ** LifeTime)
YearlyCosts = (Rental * ProjectkW)

########################################



turbPos = LOC.astype(int)
substationPos = np.array(sum(turbPos)/len(turbPos))
substationPos = substationPos.astype(int)


aep = getAEP(turbPos, scale, wr)
lengths, substationPos = getCablesLengthAndSubstation(turbPos, scale)
cablesLength = np.sum(lengths.toarray()) / 1000
lcoe, lb = getLCOE(aep, cablesLength)

print("AEP: " + str(aep))
print("LandBOSSE costs: " + str(lb))
print("LCOE: " + str(lcoe))
print(f"CablesCost: {cablesLength*CostPerKM}")

print("AEP with no wake: " + str(fi.get_farm_AEP_wind_rose_class(wind_rose=wr, no_wake=True)))

farmLayer = farmAllocator(turbPos, binary_array)

if serious:
    fi.reinitialize(wind_directions=[270])
    horizontal_plane = fi.calculate_horizontal_plane(
        x_resolution=200,
        y_resolution=100,
        height=120.0
    )

    wakeviz.visualize_cut_plane(
        horizontal_plane,
        label_contours=True,
        title="Horizontal at direction "
    )
    fi.calculate_wake()

plt.figure()
plt.imshow(binary_array, cmap='binary', interpolation='nearest', alpha=0.6)
plt.imshow(farmLayer, cmap='Greens', interpolation='nearest', alpha=0.7)
plt.show()

