import numpy as np
import os
import sys
import time
import wandb



class Visualizer:

    def __init__(self,
                 name: str,
                 isTrain=True,
                 checkpoints_dir='./checkpoints'):
        """
        :param name: name of the experiment. It decides where to store samples and models
        :param checkpoints_dir: models are saved here
        """
        self.isTrain = isTrain
        self.name = name
        self.saved = False
        wandb.init(project=name)


    def reset(self):
        """Reset the self.saved status"""
        self.saved = False


    def display_map(self, map_images):
        pass
        #map_images = (map_images+1) / 2.0 * 255.0
        #self.vis.image(map_images, win='map')

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on visdom; save current results to an HTML file.
        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        pass

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values
        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs

        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()
        """
        pass
