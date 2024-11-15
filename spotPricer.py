import pandas as pd

def denmarkSpot(A, WS):
    price = A * (1.235-0.04295*((110/120)**0.17)*WS)
    return price

def bayernSpot(A, WS):
    price = A * (1.145-0.03755*((110/120)**0.20)*WS)
    return price

numberOfYears = 20
denmark_avg_speed = 8.41441
bayern_avg_speed = 5.856243093210501

location = 'bayern'  # 'bayern' or 'denmark'

spotPrice = pd.read_csv('spotPrice.csv', index_col=0)

startingYear = 2020

spotPriceList = []
yearList = []

for year in spotPrice.index.tolist():
    if location == 'denmark':
        a = spotPrice.loc[year, 'den_A']
        p = denmarkSpot(a, denmark_avg_speed)
        yearList.append(year)
        spotPriceList.append(p)

    if location == 'bayern':
        a = spotPrice.loc[year, 'bayern_A']
        p = bayernSpot(a, bayern_avg_speed)
        yearList.append(year)
        spotPriceList.append(p)


df = pd.DataFrame()

df['year'] = yearList
df['spot_price'] = spotPriceList

df.to_csv(f"{location}_spot_price.csv")