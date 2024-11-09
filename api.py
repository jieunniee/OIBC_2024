

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