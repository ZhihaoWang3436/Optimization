import csv
import gzip
import numpy as np
import pandas as pd

filename = '..\\data\\citys_data.csv'

data = pd.read_csv(filename)[['横坐标', '纵坐标']].values
print(data)