import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


aep = 153601758468.35364 / 1000  # AEP of the farm, in Wh gives in kWh
# define the spot prices file.
# 'denmark_spot_price.csv' for denmark
# 'bavaria_spot_price.csv' for bavaria
spot_prices = pd.read_csv('bavaria_spot_price.csv', index_col=1)
numberOfTurbines = 15 # define the number of turbines of the farm
location = 'bavaria' # 'denmark' or 'bavaria'
turbineType = 'LSP' # 'IEA' 'BAU' 'LSP'


LandBossPerProject = 39877128 # LandBOSSE project cost

if location == 'denmark':
    Rental = 15  # Rental costs per kW a year
    RealDiscount = 0.04  # Real discount rate

if location == 'bavaria':
    Rental = 20  # Rental costs per kW a year
    RealDiscount = 0.035  # Real discount rate

if turbineType == 'IEA':
    TurbineCost = 1125  # Turbine Cost per kW
    ProjectkW = 3370 * numberOfTurbines  # total kW for the farm

if turbineType == 'BAU':
    TurbineCost = 1300  # Turbine Cost per kW
    ProjectkW = 3300 * numberOfTurbines  # total kW for the farm

if turbineType == 'LSP':
    TurbineCost = 1400  # Turbine Cost per kW
    ProjectkW = 3250 * numberOfTurbines  # total kW for the farm

OnM = 0.012  # O&M costs per kWh
LifeTime = 20  # lifetime of the project in years


InitialInvestment = TurbineCost * ProjectkW + LandBossPerProject    # calculates initial investment


NPV = -InitialInvestment - (Rental*ProjectkW)

print(f"Initial Investment: {InitialInvestment:.2f}\n")
TLCCinitial =InitialInvestment
TLCCrental = (Rental*ProjectkW)
TLCCOnM = 0
revenue = 0
NPVenergy = 0

# calculates the NPV for each year
for year in spot_prices.index.tolist()[1:]:
    spot = spot_prices.loc[year, 'spot_price']
    TLCCOnM += OnM*aep / ((1+RealDiscount)**(year-2020))
    revenue += ((aep/1000)*spot) / ((1+RealDiscount)**(year-2020))
    TLCCrental += (Rental*ProjectkW)/ ((1+RealDiscount)**(year-2020))
    NPV += (((aep/1000)*spot) - ((Rental*ProjectkW + OnM*aep))) / ((1+RealDiscount)**(year-2020))
    NPVenergy += aep / ((1+RealDiscount)**(year-2020))
    print(f"NPV: {NPV:.2f}")

# calculates the PI from the NPV
PI = 1 + (NPV/InitialInvestment)
print(f"\nPI: {PI:.3f}")

LCOE = (TLCCinitial+TLCCrental+TLCCOnM)/NPVenergy
print(f"\nLCOE: {(LCOE*1000):.3f}$ per MWh")

# calculate A and the LCOE if no rent was paid on year 0
A = (((1 + RealDiscount) ** LifeTime) - 1) / (RealDiscount * (1 + RealDiscount) ** LifeTime)
LCOE = (InitialInvestment / ((aep) * A)) + ((Rental*ProjectkW) / (aep)) + OnM

print(f"\nLCOE if no rent paid on year 0: {(LCOE*1000):.3f}$ per MWh")



NPVIRR = []
IRRList = []

# calculates different NPV for the IRR graph
for val in np.arange(0.01, 0.25, 0.01):
    NPVdisocunt = -InitialInvestment
    IRRList.append(val)
    for year in spot_prices.index.tolist()[1:]:
        spot = spot_prices.loc[year, 'spot_price']
        NPVdisocunt += (((aep/1000)*spot) - ((Rental*ProjectkW + OnM*aep))) / ((1+val)**(year-2020))
    NPVIRR.append(NPVdisocunt/1e6)


# shows the NPV vs discount rate for getting the IRR graphically
plt.plot(IRRList, NPVIRR, label= 'NPV vs. discount rate')
plt.title('NPV vs. IRR')
plt.xlabel('IRR [%]')
plt.ylabel('NPV [million $]')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()



labels = ['Construction', 'Turbines']
sizes = [LandBossPerProject, TurbineCost * ProjectkW]
plt.pie(sizes,  labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Initial investment cost breakdown')
plt.axis('equal')
plt.show()


labels = ['Construction', 'Turbines', 'Rental', 'Operation and maintenance']
sizes = [(LandBossPerProject / ((aep) * A)),(TurbineCost * ProjectkW / ((aep) * A)), ((Rental*ProjectkW) / (aep)), OnM]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=-20)
plt.title('LCOE cost breakdown for ')
plt.axis('equal')
plt.show(block=False)

plt.figure()
labels = ['Construction', 'Turbines', 'Rental', 'Operation and maintenance']
sizes = [LandBossPerProject, TurbineCost * ProjectkW, TLCCrental, TLCCOnM]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=-20)
plt.title('TLCC cost breakdown for ')
plt.axis('equal')
plt.show()

print("TLCC stuff:")
print(f"Construction: {LandBossPerProject}\nTurbines: {TurbineCost * ProjectkW}\nRental: {TLCCrental}\nOperation and maintenance: {TLCCOnM}")
print(f"Revenue: {revenue}")