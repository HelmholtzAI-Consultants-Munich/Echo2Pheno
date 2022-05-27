import pydicom as dcm
import cv2
import numpy as np
from scipy.signal import find_peaks

class EchoCard(object):
    '''
    This class collects all information needed for one mouse scan during the end2end framework
    Parameters
    ----------
        filepath: string
            Path of the file (echocardiogram) to be loaded
    Attributes
    ----------
        dcm_file: pydicom.dataset.FileDatase
            The loaded dicom file 
        time_res: float
            Pixel resolution in x axis (time in seconds)
        len_res: float
            Pixel resolution in y axis (length in mm)
        region_min_x: int
            Min x in pixels of raw image information in original image
        region_max_x: int
            Max x in pixels of raw image information in original image
        region_min_y: int
            Min y in pixels of raw image information in original image
        region_max_y: int
            Max y in pixels of raw image information in original image
        image: numpy array
            The full image concatenated into one long array
        crop_size: int
            The size in which to crop image into windows for quality acquisition classification task
            Depends on the weight of the mouse
        overlap: int
            The number of pixels in which the last window created for the segmentation task overlaps with the seconds to last.
            Needed for when we connect segmentation mask back to one long image
        mask: numpy array
            The full concatenated segmentation mas
        vols: list of floats
            Contatins the LV volumes calculated for each point in the image (column)
        lvids:
            Contatins the LVIDs calculated for each point in the image (column)
    '''
    def __init__(self, filepath):
        # load dicom file
        self.dcmfile = dcm.dcmread(filepath)
        # pixel time resolution is in seconds
        self.time_res = self.dcmfile[0x0018, 0x6011][2][0x0018, 0x602c].value 
        # pixel length resolution is converted to mm (is in cm)
        self.len_res = self.dcmfile[0x0018, 0x6011][2][0x0018, 0x602e].value * 10 
        # the window borders where raw image is
        self.region_min_x = self.dcmfile[0x0018, 0x6011][2][0x0018, 0x6018].value 
        self.region_max_x = self.dcmfile[0x0018, 0x6011][2][0x0018, 0x601c].value
        self.region_min_y = self.dcmfile[0x0018, 0x6011][2][0x0018, 0x601a].value
        self.region_max_y = self.dcmfile[0x0018, 0x6011][2][0x0018, 0x601e].value
    
    # this function is used as the original dicom file is in 49 frames of overlapping time regions
    # here we concatenate it all into one large numpy array
    def make_timeseries(self):
        """
        Takes the dicom file and creates one long numpy array containing only image information in grayscale
        """
        # get numpy pixel array from dicom
        img_data_raw  = self.dcmfile.pixel_array
        # make grayscale
        rgb_weights = [0.2989, 0.5870, 0.1140]
        img_data_raw = np.dot(img_data_raw [:,:,:,:3], rgb_weights)
        # crop to include only raw image 
        img_data = img_data_raw [:,self.region_min_y:self.region_max_y, self.region_min_x:self.region_max_x] 
        '''
        # every 10 frames we have no repeating data (visualize dicom to understand) - hardcoded values ahead!
        list_frames = [10, 20, 30, 40]
        self.image = img_data[0]
        for i in list_frames:
            self.image = np.concatenate((self.image, img_data[i]), axis=1)
        # add the last part missing from frame 40 - hardcoded value ahead!
        self.image = np.concatenate((self.image, img_data[-1, :, 302:]), axis=1)
        '''
        self.image = np.copy(img_data[0,:,:])
        for i in range(img_data.shape[0]-1):
            frame = img_data[i]
            next_frame = img_data[i+1]
            concat_part = find_overlap(frame, next_frame)
            # if no overlap region was found
            if concat_part is None:
                check_equal = frame==next_frame
                # either frames are identical (length//2 + 1 not checked - see find_overlap loop range)
                if check_equal.all():
                    #print('Identical frames found. Going to continue')
                    pass
                # or they are completely different
                else:
                    print('No overlap found for frame: ', i,'. Taking entire frame')
                    self.image =  np.concatenate((self.image, next_frame), axis=1)
            else:
                self.image =  np.concatenate((self.image, concat_part), axis=1)
    
    def make_quality_windows_aut(self):
        """
        This function is used to create windows from the large numpy array (image) which will be sent into the 
        quality classification network. The start and end times of each window are also returned so we can later 
        show in which regions a good and bad classification was made
        Returns
        ----------
            windows: list of numpy arrays
                Contains the cropped windows of the image
            times: list of ints
                A list containing the LV Volume either for systole, diastole, or all points
        """
        windows = []
        times = []
        length = self.image.shape[1]
        num_windows = length // self.crop_size +1 
        for i in range(num_windows):
            if i<num_windows-1:
                # hard coded value following - make var in future
                windows.append(self.image[60:(60+self.crop_size), i*self.crop_size:(i+1)*self.crop_size])
                times.append((i*self.crop_size, (i+1)*self.crop_size))
            else:
                # even though the last window overlaps with the previous use the classification label only for new part
                windows.append(self.image[60:(60+self.crop_size), (length-self.crop_size):])
                times.append((i*self.crop_size, length))
        return windows, times

    def make_quality_windows_man(self):
        """
        This function is used to create windows from the large numpy array (image) which will be sent into the 
        quality classification network. The start and end times of each window are also returned so we can later 
        show in which regions a good and bad classification was made.
        This function also take segmentation result into account when cropping to ensure we crop around the middle
        of where the heart has been located
        Returns
        ----------
            windows: list of numpy arrays
                Contains the cropped windows of the image
            times: list of ints
                A list containing the LV Volume either for systole, diastole, or all points
        """
        windows = []
        times = []
        length = self.image.shape[1]
        num_windows = length // self.crop_size +1 
        wave_heights, _ = np.where(self.mask==1)
        wave_loc_avg = int(np.mean(wave_heights))
        if wave_loc_avg - self.crop_size//2 < 0: 
            top=0
            bottom=self.crop_size
        else:    
            top = wave_loc_avg - self.crop_size//2
            bottom = wave_loc_avg + self.crop_size//2
        
        for i in range(num_windows):
            if i<num_windows-1:
                # hard coded value following - make var in future
                windows.append(self.image[top:bottom, i*self.crop_size:(i+1)*self.crop_size])
                times.append((i*self.crop_size, (i+1)*self.crop_size))
            else:
                # even though the last window overlaps with the previous use the classification label only for new part
                windows.append(self.image[top:bottom, (length-self.crop_size):])
                times.append((i*self.crop_size, length))
        return windows, times

    def make_seg_windows(self):
        """
        This function is used to create windows from the large numpy array (image) which will be sent into the 
        segmentation network.
        Returns
        ----------
            windows: list of numpy arrays
                Contains the cropped windows of the image
        """
        height, length = self.image.shape
        num_windows = length//height + 1
        windows = []
        for i in range(num_windows):
            if i<num_windows-1:
                windows.append(self.image[:,i*height:(i+1)*height])
            else:
                windows.append(self.image[:,(length-height):])
                #overlap = (length-height, (num_windows-1)*height)
                self.overlap = (i+1)*height - length #i*height - (length-height)
        return windows

    # after the segmentations have been made by the segmentation network this function connects all masks into one timeseries again (as self.image)
    def connect_masks(self, masks):
        """
        This function is used to create one segmentation mask corresponding to image. This is done by resize the segmentations masks
        back to the original window shapes and concatenating them into one long numpy array.
        Parameters
        ----------
            masks: list of numpy arrays
                A list of the segmentation network outputs. The cropped masks we wish to connect back into one long mask to match image
        """
        for idx, window_mask in enumerate(masks):
            # resize to original size as the images have been downsampled for the segmentation
            window_mask = cv2.resize(window_mask, dsize=(self.image.shape[0],self.image.shape[0]), interpolation=cv2.INTER_NEAREST)
            # fill the mask to make one timeframe
            if idx==0:
                self.mask = np.copy(window_mask)
            elif idx < len(masks) -1:
                self.mask = np.concatenate((self.mask, window_mask), axis=1)
            else:
                self.mask = np.concatenate((self.mask, window_mask[:,self.overlap:]), axis=1)
    
    # this function 
    def get_vols(self):
        """
        This function calculates the LVID and LV Vol in each column of the mask (for each instance)
        """
        heart_vols = []
        heart_lvids = []
        for i in range(self.mask.shape[1]):
            column = self.mask[:,i]                                 # get column
            unique, counts = np.unique(column, return_counts=True)  # get unique values and number of occurences
            uncounts = dict(zip(unique, counts))
            try:
                # the count of 1s from the segmentation will be the LVID
                heart_len = uncounts[1.0]                           
                real_heart_len = heart_len * self.len_res
                # use Teichholz formula (LV Vol = 7/2.4+LVID * LVID^3) to calculate LV Vol
                heart_vol = (7 * real_heart_len**3)/(2.4+real_heart_len)
                heart_lvids.append(real_heart_len)
                heart_vols.append(heart_vol)
            # if for this image there is no heart beat
            except KeyError:
                heart_lvids.append(0)
                heart_vols.append(0)
        self.vols = heart_vols
        self.lvids = heart_lvids
        
    def get_diastoles(self):
        """
        This function calculates the LVIDs and LV Vols in diastole as well as the time of their occurence
        Returns
        ----------
            A tuple containing:
            peaks: numpy array
                The time which the diastoles were detected
            diatols: list of floats
                Contains all the detected LVIDs in diastole
            vols: list of floats
                Contains all the LV Volumes in diastole
        """
        peaks, _ = find_peaks(self.lvids, distance=85) # consider than maximum heartbeat is 850 beats per minute - distance 80 pixels
        diastoles = [self.lvids[x] for x in peaks]
        vols = compute_teicholz(diastoles)
        return (peaks, diastoles, vols)
    
    def get_systoles(self):
        """
        This function calculates the LVIDs and LV Vols in systole as well as the time of their occurence
        Returns
        ----------
            A tuple containing:
            peaks: numpy array
                The time which the systole were detected
            diatols: list of floats
                Contains all the detected LVIDs in systole
            vols: list of floats
                Contains all the LV Volumes in systole
        """
        peaks, _ = find_peaks(-np.array(self.lvids), distance=85) # consider than maximum heartbeat is 850 beats per minute - distance 80 pixels
        systoles = [self.lvids[x] for x in peaks]
        vols = compute_teicholz(systoles)
        return (peaks, systoles, vols)

    def get_heartrate(self, peaks):
        """
        This function calculates the heartrate for each heart beat by counting the distance between two diastoles 
        Parameters
        ----------
            peaks: numpy array
                Contains the points at which there is a diastole in the image
        Returns
        ----------
            heartrates: list of floats
                The heartrates in [bpm] for each heart beat
        """
        heartrates = []
        for idx, peak in enumerate(peaks):
            if idx < len(peaks)-1:
                # distance between two peaks = 1 T
                pixel_length = peaks[idx+1] - peak 
                # translate pixes to seconds
                time_length = pixel_length*self.time_res
                # get heartrate in beats per second
                heartrates.append(60/time_length)
        return heartrates
    
    def weight_to_size(self, weight):
        """
        This function maps the mouse's weight to a the size we need to crop our images for the acquisition quality classification
        Parameters
        ----------
            weight: int
                The weight of the mouse currently being evaluated
        """
        # define weight regions
        if weight < 15:
            size_ratio = 0.5
        elif weight < 25:
            size_ratio = 0.6
        elif weight < 35:
            size_ratio = 0.75
        elif weight < 45:
            size_ratio = 0.9
        else:
            size_ratio = 1
        size = int(self.image.shape[0] * size_ratio)
        self.crop_size = size


def compute_teicholz(lvids):
    """
    This function uses the Teichholz formula to compute the LV Vol given a LVID
    Parameters
    ----------
        lvids: list of floats
            Containt the lvids
    Returns
    ----------    
        vols: list of floats
            Contains the computed LV Volumes    
    """
    vols = []
    for lvid in lvids:
        # Teichholz formula = 7/2.4+LVID * LVID^3
        vols.append((7 * lvid**3)/(2.4+lvid))
    return vols


def find_overlap(cur_frame, next_frame):
    """
    This function receives two successive frames and finds the overlap between the two. This is done by assuming that at least half 
    of the frame is overlapping. The second half of the current image is taken and with a step of one is slid accross and compared to
    the next frame. The point where the two matrices are identical marks the begining of the new region of the next frame. This is then
    extracted from the frame and returned
    Parameters
    ----------
        cur_frame: numpy array
            The current frame
        next_frame: numpy array
            The next frame. The first part should overlap with cur_frame and at a point it should have new image information.
            
    Returns
    ----------
        new_part_start: numpy array
            The part of next_frame which is unique and not appearing in the current_frame
    """
    new_part_start = 0
    assert cur_frame.shape == next_frame.shape,'Size mismatch in frames - Exiting!'
    length = cur_frame.shape[1]
    # we assume at least half of the image to overlap
    overlap_part = cur_frame[:,length//2:]
    # slide through the next frame to find where overlapping part starts
    for i in range(length//2):
        next_frame_part = next_frame[:,i:i+length//2]
        check_equal = overlap_part==next_frame_part
        # if the matrices match we have found the start point of the overlapping region. Store it and exit the loop
        if check_equal.all():
            new_part_start = i+length//2
            break
    if new_part_start == 0:
        return None    
    else:
        new_part = next_frame[:, new_part_start:]
        return new_part



