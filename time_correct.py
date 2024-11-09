
# Define the function to convert from seconds since "1970-01-01 09:00" KST to a readable date format

from datetime import datetime, timedelta, timezone

import pandas as pd

# Base time "1970-01-01 09:00 KST" in UTC
base_time = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc) + timedelta(hours=9)

def convert_from_base_time(seconds_since_base):
    # Add the seconds to the base time to get the target date
    target_date = base_time + timedelta(seconds=seconds_since_base)
    return target_date

# Testing the function with an example value
example_seconds = 1729612800  # Example seconds since 1970-01-01 09:00 KST
print(convert_from_base_time(example_seconds))

# Define the function to convert a given datetime to seconds since "1970-01-01 09:00 KST"


def convert_to_base_time(year, month, day, hour):
    target_datetime = datetime(year, month, day, 0, 0, 0, tzinfo=timezone.utc) + timedelta(hours=hour)
    # Calculate the time difference from the base time
    time_difference = target_datetime - base_time
    # Convert the time difference to seconds
    seconds_since_base = int(time_difference.total_seconds())
    return seconds_since_base


date_and_values = []

start_time = 1729609200
df = pd.read_csv('OIBC_2024_DATA/data/smpDataDa_2024.csv')
print(df.columns)
for index, row in df.iterrows():
    date = str(row['Date/Hour'])
    # print(date)
    year = int(date[:4])
    month = int(date[4:6])
    date = int(date[6:8])

    if convert_to_base_time(year, month, date, 0) < start_time:
        continue

    for hour in range(1, 24 + 1):
        value = row.iloc[hour]
        key = convert_to_base_time(year, month, date, hour)

        date_and_values.append((key, value))

print(date_and_values[:10])

df2 = pd.DataFrame(date_and_values, columns=['ts', '하루전가격(원/kWh)'])
df2.to_csv('OIBC_2024_DATA/data/제주전력시장_시장전기가격_하루전가격_3.csv', index=False)
