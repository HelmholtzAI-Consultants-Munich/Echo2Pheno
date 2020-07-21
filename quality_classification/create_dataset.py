import pydicom as dcm
import numpy as np
import os
import pandas as pd
from PIL import Image
import cv2
from scipy.signal import find_peaks, savgol_filter
import argparse

def make_timeseries(filepath):
    """
    This function opens a DICOM echocardiography, converts it to grayscale, crops so only image information is available, connects frames into one
    long continuous numpy array and returns it
    Parameters
    ----------
        filepath: string
            The path of the DICOM file
    Returns
    -------
        ds: pydicom.dataset.FileDataset
            The dicom object, including pixel array and metadata
        timeframe: numpy array
            The long concatenated numpy array
    """
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
    """
    This function crops the echocardiogram timeseries into smaller images, either contatining only regions of good or bad acquisition quality 
    Parameters
    ----------
        good_start: float
            The start time in seconds of good quality acquisition region
        good_end: float
            The end time in seconds of good quality acquisition region
        bad_start: float
            The start time in seconds of bad quality acquisition region
        bad_end: float
            The end time in seconds of bad quality acquisition region
        timeseries: numpy array
            The long numpy array contatining the echocardiogram
        tot_time: float
            The total time in seconds of the echocardiogram
    Returns
    -------
        good_regions: list of numpy arrays
            The list contains all good regions of acquisition cropped according to good_start and good_end from timeseries
        bad_regions: list of numpy arrays
            The list contains all bad regions of acquisition cropped according to bad_start and bad_end from timeseries
    """
    
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

def find_custom_peaks(timeseries):
    """
    This function find peaks in an echocardiogram which shoule help give a rough estimation of the average heartrate. 
    This is then used to estimate how to crop the timeseries so roughly ~3 heartbeats are included in every cropped image.
    Parameters
    ----------
        timeseries: numpy array
            The (region of) echocardiogram for which we wish to find the peaks
    Returns
    -------
        crop_size: int
            Gives the size of crop. After cropping each image should include about 3 heartbeats.
        mid_row: int
            Gives the row of the image which was estimated to be in the middle of the inner heart. We later use this to crop around it
    """
    # find black line which should be in the middle of the heart wave and at least 100 pixels
    black_line = np.zeros((50, timeseries.shape[1])) #
    min_l1 = 0
    row = 0
    for i in range(50, timeseries.shape[0] - 300):
        l1 = np.mean(abs(black_line - timeseries[i:i+50,:]))
        if i == 50:
            min_l1 = l1
            continue
        if l1 < min_l1:
            min_l1 = l1
            row = i
    # create marker image
    marker = np.zeros(timeseries.shape)
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
    color_timeseries = cv2.cvtColor(timeseries.astype('uint8'), cv2.COLOR_GRAY2RGB)
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
        points.append(timeseries.shape[0] - min_border) # in numpy image indexing is 691->0 instead of 0->691
    # smooth points
    points = np.array(points)
    points_smooth = savgol_filter(points, 85, 3) # window size 51, polynomial order 3
    # find peaks
    peaks, _ = find_peaks(points_smooth, distance=85) # consider than maximum heartbeat is 850 beats per minute - distance 80 pixels
    mid_row = row + 70 
    crop_size = int(3*timeseries.shape[1]/(peaks.shape[0]-1))  # we want ~3 heartbeats per image
    #print('Crop size: ', crop_size)
    return crop_size, mid_row

def make_custom_windows(crop_size, mid_row, timeseries, window_idx, window_step):
    """
    This function crops an image and returns it
    Parameters
    ----------
        crop_size: int
            The size to which we wish to crop
        mid_row: int
            The row around which we wish to crop
        timeseries: numpy array
            The (region of) echocardiogram which we wish to crop
        window_idx: int
            The current window we are cropping from the long numpy array
        window_step: int
            We include a window step so overlapping regions occur in the image. Cropping will be performed on the timeseries numpy
            array with a sliding window approach of step window_step
    Returns
    -------
        cropped: numpy array
            The cropped image of size crop_size x crop_size
    """
    # get length to crop considering we want ~ 3 heartbeats in each image
    if mid_row < crop_size//2:
        cropped = timeseries[0:crop_size,(window_idx*window_step):(window_idx*window_step + crop_size)]
    else:
        cropped = timeseries[(mid_row-crop_size//2):(mid_row+crop_size//2),(window_idx*window_step):(window_idx*window_step + crop_size)]
    return cropped

def make_windows(good_imgs, bad_imgs, output_dir, filename):
    """
    This function crops windows from the good and bad regions of the image and saves them
    Parameters
    ----------
        good_imgs: list of numpy arrays
            The list contains all good regions of acquisition cropped as computed in get_good_bad
        bad_imgs: list of numpy arrays
            The list contains all bad regions of acquisition cropped as computed in get_good_bad
        output_dir: string
            The root directory for saving good and bad images    
        filename: string
            The name of the current echocardiogram (i.e. mouse) we are creating images from
    """
    # window step is defined for the sliding window approach when cropping
    window_step = 100
    # create good and bad directories to save images
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
    # go first through good regions of image
    for np_array in good_imgs:
        timeseries_len = np_array.shape[1]
        # estimate the crop size and the around which row in the image to crop
        crop_size, mid_row = find_custom_peaks(np_array)
        if timeseries_len < crop_size: #img_width:
            print('image is too small to create samples')
            continue
        else:
            # calculate how many windows will be created with the sliding window approach
            num_windows = (timeseries_len - crop_size + window_step) // window_step  
            # then crop all these 
            for i in range(num_windows):
                window_img = make_custom_windows(crop_size, mid_row, np_array, i, window_step)
                # and save as npy 
                out_file_name = filename + '_nwin' + str(i) + '.npy'
                np.save(os.path.join(out_good_npy, out_file_name), window_img)
                # and png
                out_file_name = filename + '_nwin' + str(i) + '.png'
                window_img = window_img.astype(np.uint8)
                pil_img = Image.fromarray(window_img)
                pil_img.convert('L')
                pil_img.save(os.path.join(out_good_png, out_file_name))
    # then through bad images - same procedure as above
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


def get_args():
    '''
    Required arguments
    ------------------
        -i: The path to the dataset, i.e. dicom files
        -o: The path in which we wish to save the created dataset, i.e. train data set
        -a: The path to the annotations file. Must be a csv file. For more details see readme
    '''
    parser = argparse.ArgumentParser(description='Create a dataset for training a acquisition quality classification network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--datapath', required=True, type=str, 
                       help='Provide path to dicom files')
    parser.add_argument('-o', '--outputpath', required=True, type=str, 
                       help='Provide path of output directory - root directory of data set you wish to create')
    parser.add_argument('-a', '--annotations', required=True, type=str, 
                       help='provide path to csv file with annotations of good and bad regions')
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
    dicom_dir = args.datapath  # e.g. /home/data/raw 
    output_dir = args.outputpath # e.g. '/home/data/train'
    annotation_filename = args.annotations # e.g.'TimeSeries.csv'
    
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
        # get good and bad regions of recording according to annotations provided
        good_imgs, bad_imgs = get_good_bad(good_start, good_end, bad_start, bad_end, timeframe, tot_time)
        # and create and save windows from good and bad regions
        make_windows(good_imgs, bad_imgs, output_dir, mouse_id)
