import os
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import numpy as np
import statistics as st
import csv

from timeseries import EchoCard
from quality_classification import predict_single as predict_acq
from quality_classification import CardioNet 
from heart_segmentation import QuickNat 
from heart_segmentation import predict_single as predict_vol


class Graph(object):
    '''
    This class collects all information needed to plot graphs and has functions which creates and saves graphs
    Parameters
    ----------
        time_per_pixel: float
            pixel resolution in x axis (time in seconds)
        labels: list of ints with values 0 or 1
            A list containing the model's quality assessment classification results   
        sigmoids: list of floats between [0,1]
             A list containing the model's quality assessment sigmoid outputs
        BEtimes: A list of tuples of floats 
            A list containing the Begin and End times (in pixels) of the windowing performed on the original timeseries image in 
            order to perform classification
    Attributes
    ----------
         time_per_pixel: float
            pixel resolution in x axis (time in seconds)
        labels: list of ints with values 0 or 1
            A list containing the model's quality assessment classification results   
        sigmoids: list of floats between [0,1]
             A list containing the model's quality assessment sigmoid outputs
        BEtimes: A list of tuples of floats 
            A list containing the Begin and End times (in pixels) of the windowing performed on the original timeseries image in 
            order to perform classification
        tot_time: float
            The total time of acquisition
        heatmap: numpy array with the same width as the original timeseries image and 1/3 of its height
            An "image" showing the sigmoid outputs of the network in the regions of the original image
    '''
    def __init__(self, time_per_pixel, labels, sigmoids, BEtimes):
        self.time_per_pixel = time_per_pixel
        self.BEtimes = BEtimes
        self.labels = labels
        self.sigmoids = sigmoids
        self.tot_time = (self.BEtimes[-1][1]-self.BEtimes[0][0])*self.time_per_pixel
        print('Total time of acquisition:', self.tot_time)

    def add_axvspan(self):
        """
        This function is called to add the classification results of the quality assessment as colors in regions of plots 
        """
        i=0
        j=0
        for BEtime, label in zip(self.BEtimes, self.labels):
            timeB = BEtime[0]*self.time_per_pixel #int(BEtime[0]*time_per_pixel*len(volumes)/tot_time)
            timeE = BEtime[1]*self.time_per_pixel #int(BEtime[1]*time_per_pixel*len(volumes)/tot_time)
            if label == 1:
                plt.axvspan(timeB, timeE, facecolor='darkcyan', alpha=0.5, label='_'*i +'Good')
                i+=1
            else:
                plt.axvspan(timeB, timeE, facecolor='orchid', alpha=0.5, label='_'*j +'Bad')
                j+=1

    def make_graph(self, points, volumes, lvids, title, output_path):
        """
        This function creates and saves a graph with two subplots. The first shows the LV Volume over time and
        the second shows the LVID over time. The quality of the image is represented with colors on the graph 
        according to the classification results by calling the add_axvspan function.
        Parameters
        ----------
            points: numpy array 
                contains the corresponding points of the occurences in volumes and lvids
            volumes: list of floats
                A list containing the LV Volume either for systole, diastole, or all points
            lvids: list of floats
                A list containing the LV Inner Diameters either for systole, diastole, or all points
            title: string
                The title of the figure to be saved
            output_path: string
                The name of the file to be saved
        """
        #f = plt.figure(figsize[12.8, 9.6])
        volume = np.array(volumes)
        lvid = np.array(lvids)

        plt.figure(figsize=[12.8, 9.6])
        # plot LV Vol
        plt.subplot(121) # plt
        plt.plot(points*self.time_per_pixel, volume) #*10**(9) # [::3] to take every third 
        self.add_axvspan()
        plt.legend()
        plt.grid(True)
        plt.ylabel('LV Vol [mm^3]')
        plt.xlabel('Time [sec]')
        plt.xticks(np.arange(0, self.tot_time, 0.5))
        plt.title('LV Volume')
        # and LVID
        plt.subplot(122)
        plt.plot(points*self.time_per_pixel, lvid)
        self.add_axvspan()
        plt.legend()
        plt.grid(True)
        plt.ylabel('LVID [mm]')
        plt.xlabel('Time [sec]')
        plt.xticks(np.arange(0, self.tot_time, 0.5))
        plt.title('LV Inner Diameters')
        
        plt.suptitle(title)
        plt.savefig(output_path)

        plt.close() #'all'

    def make_custom_heatmap(self, img):
        """
        This function creates a heatmap with the same width as the original timeseries image and 1/3 of its height
        It is an "image" showing the sigmoid outputs of the network in the regions of the original image
        Parameters
        ----------
            img: numpy array
                The original timeseries image
        """
        self.heatmap = np.zeros((img.shape[0]//3, img.shape[1]))
        for (Btime, Etime), label in zip(self.BEtimes, self.sigmoids):
            self.heatmap[:,Btime:Etime] = label #255*

    def map_sigs_to_colors(self, peaks):
        """
        This function calculates the sigmoid value (continuous value of quality acquisition) at each time point in the list peaks
        Parameters
        ----------
            peaks: numpy array
                A list containing points in time (represented in pixels) during which a diastole occurs
        Returns
        -------
            new_labels: list of floats
                A list containing the corresponding sigmoid value (quality of acquisition) for each point in peaks
        """
        new_labels = []
        for peak in peaks:
            for (Btime, Etime), label in zip(self.BEtimes, self.sigmoids):
                if peak >= Btime and peak < Etime:
                    new_labels.append(label)
                    break
        return new_labels

    def make_hr_graph(self, heartrate, peaks, vols, output_path):
        """
        This function creates a graph with two subplots. The first sublots shows the heartrate over time and the second subplot
        show the LV Vol;d over the heartrates. The quality of the image is represented as an image with continuos colors under the first 
        sbuplot according to the classification results. In the second subplot the quality of the image is represented as a heatmap
        where each point in the plot is represented by a different color representing the quality of acquisition during the time the measurement
        was mede
        Parameters
        ----------
            heartrate: list of ints
                Contains a list of the heartrate calculated for each heart beat in [bpm]
            peaks: numpy array
                Contains the points (pixels) corresponding to when the heart is in diastole which were used for the heartrate calculation
            vols: list of floats
                Contains the LV Vol in diastole 
            output_path: string
                The name of the file to be saved
        """
        plt.figure(figsize=[12.8, 9.6])
        grid = plt.GridSpec(6, 2, hspace=0.0, wspace=0.2)
        ax_hrt = plt.subplot(grid[:-1, 0])  # grid for graph heartrate-time
        ax_h = plt.subplot(grid[-1, 0])     # grid for classification regions
        ax_vhr = plt.subplot(grid[:, 1])    # grid for graph volume-heartrate
        sig_colors = self.map_sigs_to_colors(peaks)
        
        ax_hrt.set_xlabel('Time [sec]')
        ax_hrt.set_ylabel('Heart rate [bpm]]')
        ax_hrt.set_xticks(np.arange(0, peaks[-1]*self.time_per_pixel, 0.5))
        ax_hrt.grid()
        ax_hrt.plot(peaks*self.time_per_pixel, np.array(heartrate), '-o')
        
        ax_h.axis('off')
        h = ax_h.imshow(self.heatmap)
        plt.colorbar(h, ax=ax_h, orientation='horizontal')
        
        v = ax_vhr.scatter(heartrate, vols, c=sig_colors, cmap='viridis')
        ax_vhr.set_xlabel('Heart rate [bpm]]')
        ax_vhr.set_ylabel('LV Vol;d [mm^3]]')
        ax_vhr.grid()
        plt.colorbar(v, ax=ax_vhr, orientation='horizontal')
        
        plt.suptitle("Heart rate plots")
        plt.savefig(output_path)
        plt.close()

    def plot_img_mask(self, img, mask, output_path):
        """
        This function plots and saves the original timeseries image and the superimposed segmentation mask of the heart 
        Parameters
        ----------
            img: numpy array
                The original timeseries image
            mask: numpy array
                The segmentation mask of the heart inner diameter
            output_path: string
                The name of the file to be saved
        """
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray') 
        ax.imshow(mask, cmap='gray', alpha=0.3)
        xt = np.arange(0, img.shape[1], step=int(0.5/self.time_per_pixel))
        ax.set_xticks(xt)
        xl = np.round_(xt*self.time_per_pixel, 1)
        ax.set_xticklabels(xl)
        ax.set_yticks([])
        plt.xlabel('Time [sec]')
        plt.savefig(output_path, bbox_inches = 'tight', dpi=1200)
        plt.close() 

    def plot_img(self, img, output_path):
        """
        This function plots and saves the original timeseries image and above that the heatmap created by the function make_custom_heatmap
        Parameters
        ----------
            img: numpy array
                The original timeseries image
            output_path: string
                The name of the file to be saved
        """
        heights = [a.shape[0] for a in [self.heatmap, img]]
        widths = [self.heatmap.shape[1]]
        fig_width = 8
        fig_height = fig_width*sum(heights)/sum(widths)
        f, axarr = plt.subplots(2,1, figsize=(fig_width, fig_height+0.4), gridspec_kw={'height_ratios': heights})
        ax = axarr[0].imshow(self.heatmap, cmap='viridis')
        axarr[0].axis('off')
        axarr[1].imshow(img, cmap='gray')
        xt = np.arange(0, img.shape[1], step=int(0.5/self.time_per_pixel))
        axarr[1].set_xticks(xt)
        xl = np.round_(xt*self.time_per_pixel, 1)
        axarr[1].set_xticklabels(xl)
        axarr[1].set_yticks([])
        axarr[1].set_xlabel('Time [sec]')

        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        plt.colorbar(ax, ax=axarr[:]) #, orientation='horizontal'
        plt.savefig(output_path, bbox_inches = 'tight', dpi=1200)
        plt.close()

''' ----------- DONE WITH GRAPH CLASS ----------- '''

''' ----------- NEXT COME HELPER FUNCTIONS FOR GETTING STATISTICS AND LOADING MODELS ----------- '''

def get_stats_good(labels, times, peaks, ds, time_res):
    """
    This function calculates various statistics for either the LVIDs in diastole or systole during good quality of acquisition
    Parameters
    ----------
        labels: list of ints, 0 or 1
            Holds the acquisition quality classification result fror each time window 
        times: list of tuples
            Each tuple holds the begin and end time of the window cut from the timeseries for the acquisition quality classification
        peaks: numpy array
            Holds the time (in pixels) of the events in ds
        ds: list of floats
            Holds either LVID;d, LVID;s or heartrates
        time_res: float
            The resolution on the x axis, i.e. to how many seconds one pixel corresponds

    Returns
    -------
        med_ds: float
            The median value of all LVIDs in either systols or diastole, or heartrates, captured during good quality of acquisition. 
            If no good acquisition regions were found 0 is returned
        avg_ds: float
            The average value of all LVIDs in either systols or diastole, or heartrates, captured during good quality of acquisition.
            If no good acquisition regions were found 0 is returned
        max(good_ds): float
            The maximum value of all LVIDs in either systols or diastole, or heartrates, captured during good quality of acquisition.
            If no good acquisition regions were found 0 is returned
        min(good_ds): float
            The minimum value of all LVIDs in either systols or diastole, or heartrates, captured during good quality of acquisition.
            If no good acquisition regions were found 0 is returned
        good_ds: list of floats
            Includes a list of LVIDs either in systole or diastole, or heartrates, only during good quality of acquisition.
            If no good acquisition regions were found an empty list is returned
        good_times: list of ints
            A list of corresponding times (in seconds) of the above good_ds
            If no good acquisition regions were found an empty list is returned
    """ 
    good_ds = []
    good_times = []
    for peak, ds in zip(peaks, ds):
        for label, (Btime, Etime) in zip(labels, times):
            if peak >=Btime and peak < Etime:
                if label == 1:
                    good_ds.append(ds)
                    good_times.append(peak*time_res)
                break
    try:
        med_ds = st.median(good_ds)
        avg_ds = sum(good_ds)/len(good_ds)
        return med_ds, avg_ds, max(good_ds), min(good_ds), good_ds, good_times
    except (ZeroDivisionError, st.StatisticsError):
        return 0, 0, 0, 0, good_ds, good_times

def load_model_device(network):
    """
    This function loads the appropriate model and gets the current device
    Parameters
    ----------
        network: string
            Should be either 'echo' or 'quicknat' defining the which model is to be loaded
    Returns
    -------
        net: model.QuickNat or model.CardioNet
            The network model instance
        device: torch.device
            The currently running divice, e.g. cpu, gpu0    
    """ 
    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if network=='quicknat':
        model_path = './heart_segmentation/checkpoints/heart-seg-net.pt'
         # load model
        params = {'num_channels':1,
                    'num_class':1,
                    'num_filters':64,
                    'kernel_h':5,
                    'kernel_w':5,
                    'kernel_c':1,
                    'stride_conv':1,
                    'pool':2,
                    'stride_pool':2,
                    'se_block': "NONE",
                    'drop_out':0.2}
        net = QuickNat(params) 
        net.to(device=device)
        state_dict = torch.load(os.path.join(model_path), map_location=device)
        net.load_state_dict(state_dict['net_state_dict'])
    else:
        model_path = './quality_classification/checkpoints/quality-clas-net.pth'
        net = CardioNet(n_channels=1, n_classes=1)
        net.to(device=device)
        net.load_state_dict(torch.load(model_path, map_location=device))
    return net, device


''' ----------- DONE WITH HELPER FUNCTIONS ----------- '''

''' ----------- NEXT COMES THE FUNCTION WHERE EVERYTHING IS RUN ----------- '''


def run(input_path, output_path, weight, graphs=True, write=None, write_file=None):
    """
    This function is where the end2end dramework is run.
    Parameters
    ----------
        input_path: string
            Path of file to be loaded
        output_path: string
            Directory to save results
        weight: int
            The weight of the mouse we are evaluating
        graphs: bool
            If true graphs will be created and saved in the ouput_path directory
        write: string
            If 'stats' then values such as max, min, median etc of the LVIDs etc are written to a csv file
            If 'all' then LVIDs etc. are written for all good classified regions
        write_file: string
            The csv file to write results to according to what has been given to write 
    """ 
    labels = []
    sigs = []
    masks = []
    # create an echo card instance
    ec = EchoCard(input_path)
    # fill in timeseries attribute of class - a numpy array of entire time of acquisition
    ec.make_timeseries()
    # split timeseries to get images for segmentation network
    vol_windows = ec.make_seg_windows()
    # load models for testing
    echo_net, device = load_model_device('echo')
    quicknat_net, _ = load_model_device('quicknat')
    print("Using device ", device)
    print("Loaded models")
    print('Image shape:', ec.image.shape)

    '''-----------SEGMENTATION PART-----------'''
    # get masks
    for img in vol_windows:
        mask, _ = predict_vol(quicknat_net, device, img, 256)
        masks.append(mask)
    # connect back to one timeseries
    ec.connect_masks(masks)
    # compute volumes and lvids for all points in timeseries
    ec.get_vols()
    # get diastole and systole lvid, lv vol and time of occurence (in pixel values)
    dpeaks, dlvids, dvols = ec.get_diastoles()
    speaks, slvids, svols = ec.get_systoles() 
    # get heartrate in [bpm]
    heartrate = ec.get_heartrate(dpeaks)

    '''-----------QUALITY ACQUISITION PART-----------'''
    # split timeseries to get images for quality classification
    # two lists are returned - one with numpy arrays (image) one with a tuple (startTime, endTime)
    ec.weight_to_size(weight)
    qual_windows, BEtimes = ec.make_quality_windows_man()
    # classify each window as good or bad
    for img in qual_windows:
        label, _ = predict_acq(echo_net, device, img, 256)
        sigs.append(label)
        labels.append(np.round(label))

    '''-----------GRAPHS PART-----------'''
    if graphs:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        graphs = Graph(ec.time_res, labels, sigs, BEtimes)
        graphs.make_custom_heatmap(ec.image)
        graphs.make_graph(np.arange(len(ec.vols)), ec.vols, ec.lvids, 'Heart Values Estimation', os.path.join(output_path, 'output_vol.png'))
        graphs.make_graph(dpeaks, dvols, dlvids, 'Diastole', os.path.join(output_path, 'output_diastole.png'))
        graphs.make_graph(speaks, svols, slvids, 'Systole', os.path.join(output_path, 'output_systole.png'))
        graphs.plot_img_mask(ec.image, ec.mask, os.path.join(output_path, 'output_img_mask.png'))
        graphs.plot_img(ec.image, os.path.join(output_path, 'output_img.png'))
        graphs.make_hr_graph(heartrate, dpeaks[:-1], dvols[:-1], os.path.join(output_path, 'output_heartrate.png'))
    
    '''-----------WRITING TO FILES PART-----------'''
    med_diastole, avg_diastole, max_diastole, min_diastole, good_lvid_d, times_lvid_d = get_stats_good(labels, BEtimes, dpeaks, dlvids, ec.time_res)
    med_systole, avg_systole, max_systole, min_systole, good_lvid_s, times_lvid_s = get_stats_good(labels, BEtimes, speaks, slvids, ec.time_res)
    med_heartrate, avg_heartrate, max_heartrate, min_heartrate, good_heartrates, times_hr = get_stats_good(labels, BEtimes, dpeaks[:-1], heartrate, ec.time_res)
    print('Average lvid;d is: ', avg_diastole, ' and average lvid;s is: ', avg_systole)
    print('Median lvid;d is: ', med_diastole, ' and median lvid;s is: ', med_systole)
    print('The average heartbeat is: ', avg_heartrate, 'and the median heart rate is: ', med_heartrate)
    # append results to file if a csv file has been given in write
    # either stats such as mean, median etc. (1 value for each file)
    if write=='stats':
        filename = input_path.split('/')[-1]
        # if the file doesn't already exist add first row with column names first
        if not os.path.isfile(write_file):
            with open(write_file, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['file','median_diastole', 'median_systole', 'median_heartrate', 'avg_diastole', 'avg_systole', 'avg_heartrate', 'max_diastole', 'max_systole', 'max_heartrate', 'min_diastole', 'min_systole', 'min_heartrate'])
        # append new line to file
        with open(write_file, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([filename, med_diastole, med_systole, med_heartrate, avg_diastole, avg_systole, avg_heartrate, max_diastole, max_systole, max_heartrate, min_diastole, min_systole, min_heartrate])
    # or heartrare, lvid;d etc. during all good acquisition regions
    elif write=='all':
        filename = input_path.split('/')[-1]
        # if the file doesn't already exist add first row with column names first
        if not os.path.isfile(write_file):
            with open(write_file, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['file','lvid;d', 'lvid;d time', 'lvid;s', 'lvid;s time', 'heart rate', 'heart rate time'])
        # append new lines to file
        with open(write_file, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            i = 0
            # depending on windowing and classification these will not necessarily have the exact same length - take the smallest
            min_len = min([len(good_lvid_s), len(good_lvid_s), len(good_heartrates)])
            for i in range(min_len):
                writer.writerow([filename, good_lvid_d[i], times_lvid_d[i], good_lvid_s[i], times_lvid_s[i], good_heartrates[i], times_hr[i]])
                i += 1

def get_args():
    parser = argparse.ArgumentParser(description='Predict heart volume for test image',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', '-i', metavar='INPUT', required=True,
                        help='Specify path of input image - must be in DICOM format')
    parser.add_argument('--mass', '-m', type=int, required=True,
                        help='Specify the body mass of the mouse')
    parser.add_argument('--output', '-o', default='.', 
                        help='Specify output path to save graphs')
    parser.add_argument('--graphs', '-g', default=True,
                        help='Specify True or False depending on whether you want to save figures')
    parser.add_argument('--write', '-w', default='all',
                        help='Specify wheter to save all good features or statistics of features. Give all or stars as input.')
    parser.add_argument('--writefile', '-f', default='output_all.csv',
                        help='Specify in which file to save features')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    run(args.input, args.output, args.mass, graphs=args.graphs, write=args.write, write_file=args.writefile)