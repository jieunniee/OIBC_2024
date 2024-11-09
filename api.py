

import requests
import pandas as pd

with open('API_KEY.txt', 'rt') as file:
    API_KEY = file.read()

dates = [
    # '2024-10-22',
    '2024-10-23',
    '2024-10-24',
    '2024-10-25',
    '2024-10-26',
    '2024-10-27',
    '2024-10-28',
    '2024-10-29',
    '2024-10-30',
    '2024-10-31',
    '2024-11-01',
    '2024-11-02',
    '2024-11-03',
    '2024-11-04',
    '2024-11-05',
    '2024-11-06',
    '2024-11-07',
    '2024-11-08',
    '2024-11-09',
    # '2024-11-10',
]

actual_weathers = []
for date in dates:
    actual_weather = requests.get(f'https://research-api.solarkim.com/data/cmpt-2024/actual-weather/{date}', headers={
                                'Authorization': f'Bearer {API_KEY}'
                            }).json()

    actual_weathers.extend(actual_weather['actual_weather_1'])
    actual_weathers.extend(actual_weather['actual_weather_2'])

df = pd.DataFrame(actual_weathers)
df.to_csv("기상실측데이터_3.csv", index=False)

smps = []
for date in dates:
    smp = requests.get(f'https://research-api.solarkim.com/data/cmpt-2024/smp-da/{date}', headers={
                                'Authorization': f'Bearer {API_KEY}'
                            }).json()
    # print(smp)
    # print(smp[0])
    # print()
    smps.extend(smp)

df2 = pd.DataFrame(smps)
df2.to_csv("제주전력시장_시장전기가격_하루전가격_2.csv", index=False)

states = []
for date in dates:
    state = requests.get(f'https://research-api.solarkim.com/data/cmpt-2024/elec-supply/{date}', headers={
                                'Authorization': f'Bearer {API_KEY}'
                            }).json()
    print(state)
    print()
    states.extend(state)

df3 = pd.DataFrame(states)
df3.to_csv("제주전력시장_현황데이터_2.csv", index=False)