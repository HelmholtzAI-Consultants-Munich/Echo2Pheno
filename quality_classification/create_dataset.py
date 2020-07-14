import pydicom as dcm
import numpy as np
import os
import pandas as pd
from PIL import Image
import cv2
from scipy.signal import find_peaks, savgol_filter

def make_timeseries(filepath):
    # load dicom file
    ds = dcm.dcmread(filepath)
    # get numpy pixel array
    img_data_raw  = ds.pixel_array
    # make grayscale
    rgb_weights = [0.2989, 0.5870, 0.1140]
    img_data_raw = np.dot(img_data_raw [:,:,:,:3], rgb_weights)
    # every 10 frames we have no repeating data (visualize dicom to understand)
    region_min_x = ds[0x0018, 0x6011][2][0x0018, 0x6018].value 
    region_max_x = ds[0x0018, 0x6011][2][0x0018, 0x601c].value
    region_min_y = ds[0x0018, 0x6011][2][0x0018, 0x601a].value
    region_max_y = ds[0x0018, 0x6011][2][0x0018, 0x601e].value
    img_data = img_data_raw [:, region_min_y:region_max_y, region_min_x:region_max_x] 
    #img_data = img_data_raw [:,112:728,28:1052] 
    list_frames = [10, 20, 30, 40]
    timeframe = img_data[0]
    for i in list_frames:
        timeframe = np.concatenate((timeframe, img_data[i]), axis=1)
    # add the last part missing from frame 40 - hardcoded value!
    timeframe = np.concatenate((timeframe, img_data[-1, :, 302:]), axis=1)
    return ds, timeframe

def get_good_bad(good_start, good_end, bad_start, bad_end, timeseries, tot_time):
    
    good_regions = []
    bad_regions = []    
    timeseries_len = timeseries.shape[1]  # this gives how many pixels are in the time axis

    good_start_split = good_start.split('  ')
    good_stop_split = good_end.split('  ')
    bad_start_split = bad_start.split(' ')
    bad_stop_split = bad_end.split(' ')

    for idx, good_start in enumerate(good_start_split):
        # translate to pixel values
        pix_val_good_start = int(float(good_start)*timeseries_len/tot_time)
        pix_val_good_stop = int(float(good_stop_split[idx])*timeseries_len/tot_time)
        # keep only good part of measurement
        timeseries_cropped = timeseries[:,pix_val_good_start:pix_val_good_stop]
        if timeseries_cropped.shape[1] == 0:
            continue
        good_regions.append(timeseries_cropped)

    for idx, bad_start in enumerate(bad_start_split):
        # translate time information from csv to pixel values
        pix_val_bad_start = int(float(bad_start)*timeseries_len/tot_time)
        pix_val_bad_stop = int(float(bad_stop_split[idx])*timeseries_len/tot_time)
        # keep only bad part of measurement
        timeseries_cropped = timeseries[:,pix_val_bad_start:pix_val_bad_stop]
        if timeseries_cropped.shape[1] == 0:
            continue
        bad_regions.append(timeseries_cropped)
    
    return good_regions, bad_regions

def find_custom_peaks(timeseries5):
    # find black line which should be in the middle of the heart wave and at least 100 pixels
    black_line = np.zeros((50, timeseries5.shape[1])) #
    min_l1 = 0
    row = 0
    for i in range(50, timeseries5.shape[0] - 300):
        l1 = np.mean(abs(black_line - timeseries5[i:i+50,:]))
        if i == 50:
            min_l1 = l1
            continue
        if l1 < min_l1:
            min_l1 = l1
            row = i
    # create marker image
    marker = np.zeros(timeseries5.shape)
    marker[row:row+50,:] = 2
    # this needs to be better defined
    if row < 100:
        marker[0:25,:] = 1
    elif row >=100 and row< 250:
        marker[0:100,:] = 1
    else:
        marker[0:200] = 1
    
    marker[-20:-1,:] = 1
    
    # blur image and make 3 channels
    color_timeseries = cv2.cvtColor(timeseries5.astype('uint8'), cv2.COLOR_GRAY2RGB)
    blur = cv2.GaussianBlur(color_timeseries,(5,5),0)
    smooth = cv2.addWeighted(blur,1.5,color_timeseries,-0.5,0)
    # watershed segmentation
    res = cv2.watershed(smooth, marker.astype('int32')) #int64
    # find max point in column and create a list of these
    points = []
    for col_idx in range(res.shape[1]):
        if col_idx == 0 or col_idx == res.shape[1]-1:
            continue
        borders = np.where(res[1:-1,col_idx]==-1)[0] # remember first and last point will be top and bottom of img so don't include
        min_border = min(borders)
        points.append(timeseries5.shape[0] - min_border) # in numpy image indexing is 691->0 instead of 0->691
    # smooth points
    points = np.array(points)
    points_smooth = savgol_filter(points, 85, 3) # window size 51, polynomial order 3
    # find peaks
    peaks, _ = find_peaks(points_smooth, distance=85) # consider than maximum heartbeat is 850 beats per minute - distance 80 pixels
    mid_row = row + 70
    crop_size = int(3*timeseries5.shape[1]/(peaks.shape[0]-1)) 
    print('Crop size: ', crop_size)
    return crop_size, mid_row

def make_custom_windows(crop_size, mid_row, timeseries5, window_idx, window_step):
    # get length to crop considering we want ~ 3 heartbeats in each image
    if mid_row < crop_size//2:
        cropped = timeseries5[0:crop_size,(window_idx*window_step):(window_idx*window_step + crop_size)]
    else:
        cropped = timeseries5[(mid_row-crop_size//2):(mid_row+crop_size//2),(window_idx*window_step):(window_idx*window_step + crop_size)]
    return cropped

def make_windows(good_imgs, bad_imgs, output_dir, filename):
    window_step = 100
    out_good_npy = os.path.join(output_dir, 'good','npys')
    if not os.path.exists(out_good_npy):
        os.makedirs(out_good_npy)
    out_good_png = os.path.join(output_dir, 'good','pngs')
    if not os.path.exists(out_good_png):
        os.makedirs(out_good_png)
    out_bad_npy = os.path.join(output_dir,'bad', 'npys')
    if not os.path.exists(out_bad_npy):
        os.makedirs(out_bad_npy)
    out_bad_png = os.path.join(output_dir, 'bad','pngs')
    if not os.path.exists(out_bad_png):
        os.makedirs(out_bad_png)

    for np_array in good_imgs:
        timeseries_len = np_array.shape[1]
        crop_size, mid_row = find_custom_peaks(np_array)
        if timeseries_len < crop_size: #img_width:
            print('image is too small to create samples')
            continue
        else:
            num_windows = (timeseries_len - crop_size + window_step) // window_step  
            for i in range(num_windows):
                window_img = make_custom_windows(crop_size, mid_row, np_array, i, window_step)
                out_file_name = filename + '_nwin' + str(i) + '.npy'
                np.save(os.path.join(out_good_npy, out_file_name), window_img)
                out_file_name = filename + '_nwin' + str(i) + '.png'
                window_img = window_img.astype(np.uint8)
                pil_img = Image.fromarray(window_img)
                pil_img.convert('L')
                pil_img.save(os.path.join(out_good_png, out_file_name))
    
    for np_array in bad_imgs:
        timeseries_len = np_array.shape[1]
        crop_size, mid_row = find_custom_peaks(np_array)
        if timeseries_len < crop_size:
            print('image is too small to create samples')
            continue
        else:
            num_windows = (timeseries_len - crop_size + window_step) // window_step 
            for i in range(num_windows):
                window_img = make_custom_windows(crop_size, mid_row, np_array, i, window_step)
                out_file_name = filename + '_win' + str(i) + '.npy'
                np.save(os.path.join(out_bad_npy, out_file_name), window_img)
                out_file_name = filename + '_win' + str(i) + '.png'
                window_img = window_img.astype(np.uint8)
                pil_img = Image.fromarray(window_img)
                pil_img.convert('L')
                pil_img.save(os.path.join(out_bad_png, out_file_name))

if __name__ == '__main__':

    dicom_dir = # add the  input directory here, e.g. /home/data/raw 
    output_dir = # add the destination dir here, e.g. '/home/data/train'
    annotation_filename = os.path.join('TimeSeries.csv'). #add the csv file of good and bad regions annotations here
    
    file_list = os.listdir(dicom_dir)
    df = pd.read_csv(annotation_filename)
    df.set_index("Recording", inplace=True)
    
    for filename in (filename for filename in file_list if filename.endswith('.dcm')):
        print('Going to open file: ', filename)
        # connect all frames of image into one long image
        dcmdata, timeframe = make_timeseries(os.path.join(dicom_dir, filename))
        print('Timeframe size: ', timeframe.shape)
        time_res = dcmdata[0x0018, 0x6011][2][0x0018, 0x602c].value 
        tot_time = time_res*timeframe.shape[1]
        print('Total time of acquisition: ', tot_time)        
        # get annotation regions for this file
        mouse_id = filename.split('.')[0]
        good_start = df.loc[[mouse_id],['Good-Start']].astype(str).values[0][0]
        good_end = df.loc[[mouse_id],['Good-End']].astype(str).values[0][0]
        bad_start = df.loc[[mouse_id],['Bad-Start']].astype(str).values[0][0]
        bad_end = df.loc[[mouse_id],['Bad-End']].astype(str).values[0][0]
        good_imgs, bad_imgs = get_good_bad(good_start, good_end, bad_start, bad_end, timeframe, tot_time)
        # create and save windows
        make_windows(good_imgs, bad_imgs, output_dir, mouse_id)
