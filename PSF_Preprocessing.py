import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import tifffile
from tqdm import tqdm
import logging


import trackpy as tp

def get_PSF_frame(movie_frame, x, y, frame_size, keep_edges=False):
    half_frame = int(frame_size/2)
    int_x = round(x); int_y = round(y)
    lower_x = int_x - half_frame
    lower_y = int_y - half_frame
    upper_x = int_x + half_frame + 1
    upper_y = int_y + half_frame + 1
    upper_limit_x = movie_frame.shape[1]
    upper_limit_y = movie_frame.shape[0]
    if keep_edges:
        if lower_x < 0:
            lower_x = 0
        if lower_y < 0:
            lower_y = 0
        if upper_x > upper_limit_x:
            upper_x = upper_limit_x
        if upper_y > upper_limit_y:
            upper_y = upper_limit_y
    else:
        if lower_x < 0 or lower_y < 0 or upper_x > upper_limit_x or upper_y > upper_limit_y:
            return False
    return np.array(movie_frame)[lower_y:upper_y, lower_x:upper_x]

def get_PSF_frames(movie, minmass=2000, separation=3, diameter=7, frame_size=13, percentile=0.9, print_progress=False, to_plot=False, movie_frames=True, dpi=100):
    if type(movie_frames) == bool:
        movie_frames = range(len(movie))
    
    # Initialize an empty list to store the PSF frames
    PSF_frames = []

     # Loop through each frame in the movie
    if print_progress:
        with tqdm(movie_frames) as pbar:
            for i in pbar:
                # Perform localization on the frame
                full_frame = tp.locate(movie[i], diameter=diameter, 
                                minmass=minmass, max_iterations=10,
                                separation=separation, percentile=percentile)  # Adjust the minmass parameter as needed

                # Plot the frame with the identified features
                if to_plot:
                    fig, ax = plt.subplots(1, dpi=dpi)
                    ax.imshow(movie[i], cmap='gray')

                # Draw a box around each identified feature
                for index, row in full_frame.iterrows():
                    if to_plot:
                        rect = patches.Rectangle((row['x']-int(frame_size/2), row['y']-int(frame_size/2)), frame_size, frame_size, linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                    PSF_frame = get_PSF_frame(movie[i], row['x'], row['y'], frame_size)
                    if type(PSF_frame) == bool:
                        continue
                    PSF_frames.append(PSF_frame)
                if to_plot:
                    plt.title(f'Full Frame {i}')
                    plt.show()
    else:
        for i in movie_frames:
            # Perform localization on the frame
            full_frame = tp.locate(movie[i], diameter=diameter, 
                            minmass=minmass, max_iterations=10,
                            separation=separation, percentile=percentile)  # Adjust the minmass parameter as needed

            # Plot the frame with the identified features
            if to_plot:
                fig, ax = plt.subplots(1, dpi=dpi)
                ax.imshow(movie[i], cmap='gray')

            # Draw a box around each identified feature
            for index, row in full_frame.iterrows():
                if to_plot:
                    rect = patches.Rectangle((row['x']-int(frame_size/2), row['y']-int(frame_size/2)), frame_size, frame_size, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                PSF_frame = get_PSF_frame(movie[i], row['x'], row['y'], frame_size)
                if type(PSF_frame) == bool:
                    continue
                PSF_frames.append(PSF_frame)
            if to_plot:
                plt.title(f'Full Frame {i}')
                plt.show()

    # Convert the list of frames to a numpy array
    PSF_frames = np.array(PSF_frames)
    return PSF_frames

def get_PSF_frames_movie_list(movies, minmass=2000, separation=3, diameter=7, frame_size=13, percentile=0.9, print_progress=False):
    first = True
    for i in range(len(movies)):
        movie = movies[i]
        if print_progress: print(f"Processing Movie {i+1}/{len(movies)}")
        if first:
            PSF_list = get_PSF_frames(movie, minmass, separation, diameter, frame_size, percentile, print_progress=print_progress)
            first = False
        else:
            PSF_list = np.concatenate((PSF_list, get_PSF_frames(movie, minmass, separation, diameter, frame_size, percentile, print_progress=print_progress)))
    return PSF_list

def show_PSFs(PSF_frames, random=True):
    print(PSF_frames.shape)
    num_PSFs = 15
    if not random:
        random_seed = 1 # random seed for reproducibility
        np.random.seed(random_seed)
    PSF_indices = np.random.choice(PSF_frames.shape[0], size=num_PSFs, replace=False)
    image_list = PSF_frames[PSF_indices]

    # Creating a 2x5 subplot grid
    fig, axes = plt.subplots(3, 5, figsize=(10, 4), dpi=150)

    # Flattening the 2D array of axes to a 1D array
    axes = axes.flatten()

    # Iterating over each image and corresponding axis
    for i in range(len(axes)):
        # Displaying the image on the corresponding axis
        axes[i].imshow(image_list[i], cmap='gray')
        axes[i].axis('off')

    plt.suptitle("Examples of PSF Frames", fontsize=24)
    plt.tight_layout()
    plt.show()

def load_tif_movies(folder_path):
    movies = []
    logger = logging.getLogger('tifffile')
    prev_level = logger.getEffectiveLevel()  # get the current logging level
    logger.setLevel(logging.ERROR)  # set level to ERROR, only ERROR level and above will be shown
    try:
        for filename in os.listdir(folder_path):
            # Check if the current file is a .tif file
            if filename.endswith('.tif'):
                # Construct the full filepath by joining the directory path and the filename
                filepath = os.path.join(folder_path, filename)
                movies.append(tifffile.imread(filepath))
    finally:
        logger.setLevel(prev_level)  # reset the logging level to its previous state
    return movies
