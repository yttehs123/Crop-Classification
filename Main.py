
#https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/

#Data loading
#Libraries

import tifffile as tiff
import datetime
import matplotlib.pyplot as plt
import tifffile as tiff
import numpy as np
import glob
from tqdm.notebook import tqdm
from skimage import exposure

def load_file(fp):
    """Takes a PosixPath object or string filepath
    and returns np array"""
    
    return tiff.imread(fp.__str__())
  
 bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'CLD']


#Getting list of dates
tile_dates = {}
for f in glob.glob('**/*.tif', recursive=True):
    if len(f.split('/')) != 4:
        continue
    tile_id = f.split('/')[1]
    date = datetime.datetime.strptime(f.split('/')[2], '%Y%m%d')
    if tile_dates.get(tile_id, None) is None:
        tile_dates[tile_id] = []
    tile_dates[tile_id].append(date)

for tile_id, dates in tile_dates.items():
    tile_dates[tile_id] = list(set(tile_dates[tile_id]))
    
    
selected_tile = list(tile_dates.keys())[0]
dates = sorted(tile_dates[selected_tile])

bands = [ 'B02', 'B03', 'B04', 'B08','CLD']
def load_image(date):
    img = list()
    for band in bands:
        file_name = f"data/{selected_tile}/{date}/{int(selected_tile)}_{band}_{date}.tif"
        img.append(load_file(file_name))
    return np.dstack(img)

#Loading data as numpy array  
def load_timeseries():
    tstack = list()
    with tqdm(dates, total=len(dates), desc="reading images") as pbar:
        for date in pbar:
            tstack.append(load_image(date.strftime("%Y%m%d")))
    return np.stack(tstack) 

timeseries = load_timeseries()
print(timeseries.shape)
