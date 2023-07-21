import numpy as np
import pandas as pd
import json
import tifffile
import logging
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Localizations:
    """A class used to represent and process localizations from ThunderSTORM.

    This class loads and processes localizations from a given path and filename, 
    it extracts the images of the PSFs in the .tif movie via the provided localizations.

    Attributes:
        pixel_size (float): The size of each pixel in the image in nm.
        movie (array-like): The loaded movie data.
        frames (array-like): The frames of the movie for each localization.
        base_localizations (array-like): Raw localizations found by frame.
        psf_frame_size (int): The size of the square PSF frames.
        psf_frames (array-like): The PSF frames for each frame of the movie
        filtered_localizations (array-like): Non-edge localizations by frame
        all_localizations (array-like): All localizations data in order (not by frame)
        all_psf_frames (array-like): All PSF frames data in order (not by frame)
        
    """
    def __init__(self, path, filename, psf_frame_size=19):
        """Initializes the Localizations class with a given path and filename.

        Expected File Structure:
            /path
                /localizations
                    <filename>-protocol.txt,
                    <filename>.csv
                /movies
                    <filename>.tif
        
        Args:
            path (str): The directory where the files are located.
            filename (str): The base name of the files to be processed.
            psf_frame_size (int, optional): The size of the PSF frame. Defaults to 19.

        Raises:
            ValueError: If the 'pixelSize' value is not found in the protocol file.
        """
        protocol_path = os.path.join(path, "localizations", filename + "-protocol.txt")
        localization_path = os.path.join(path, "localizations", filename + ".csv")
        movie_path = os.path.join(path, "movies", filename + ".tif")
        
        self.pixel_size = self.get_pixel_size(protocol_path)
        self.movie = self.load_tif_movie(movie_path)
        self.frames, self.base_localizations = self.load_localizations(localization_path)
        self.psf_frame_size = psf_frame_size
        self.psf_frames, self.filtered_localizations = self.get_psf_frames(psf_frame_size)
        
        self.all_localizations = []; self.all_psf_frames = []
        for i in range(len(self.filtered_localizations)):
            self.all_localizations.extend(self.filtered_localizations[i])
            self.all_psf_frames.extend(self.psf_frames[i])
        self.all_localizations = np.array(self.all_localizations)
        self.all_psf_frames = np.array(self.all_psf_frames)

    def get_pixel_size(self, path):
        """Extracts the pixel size from the protocol file.

        Args:
            path (str): The path to the protocol file.

        Returns:
            float: The pixel size in nm.

        Raises:
            ValueError: If the 'pixelSize' value is not found in the protocol file.
        """
        pixel_size = None
        with open(path, 'r') as file:
            json_lines = []
            json_start = False
            for line in file:
                line = line.strip()  # remove leading/trailing white space
                if line.startswith('{'):
                    json_start = True
                if json_start:
                    json_lines.append(line)
                if line.endswith('}'):  # end of JSON
                    json_start = False
                    data = json.loads(' '.join(json_lines))
                    if 'pixelSize' in data:
                        pixel_size = data['pixelSize']
                        break
                    else:  # reset json_lines for next JSON object
                        json_lines = []
        if pixel_size is None:
            raise ValueError(f"No 'pixelSize' value in {path}")
        return pixel_size

    def load_tif_movie(self, path):
        """Loads a movie file in TIFF format.

        Args:
            path (str): The path to the TIFF file.

        Returns:
            array-like: The loaded movie data.
        """
        logger = logging.getLogger('tifffile')
        prev_level = logger.getEffectiveLevel()  # get the current logging level
        logger.setLevel(logging.ERROR)  # set level to ERROR, only ERROR level and above will be shown
        try:
            movie = tifffile.imread(path)
        finally:
            logger.setLevel(prev_level)  # reset the logging level to its previous state
        return movie

    def load_localizations(self, path):
        """Loads localizations from a CSV file.

        Args:
            path (str): The path to the CSV file.

        Returns:
            tuple: A tuple containing an array-like object of frame indices and a list of arrays of localizations for each frame.
        """
        localization_df = pd.read_csv(path, delimiter=',')
        x = localization_df['x [nm]']/self.pixel_size
        y = localization_df['y [nm]']/self.pixel_size
        pts = np.array(list(zip(x, y)))
        frame_idxs = localization_df['frame'].astype(int)
        frames_dict = {frame_idx: [] for frame_idx in set(frame_idxs)}
        for pt, frame_idx in zip(pts, frame_idxs):
            frames_dict[frame_idx].append(pt)
        return frame_idxs, [np.array(frames_dict.get(frame_idx+1, [])) for frame_idx in range(max(frames_dict.keys()))]
    
    def get_psf_frames(self, psf_frame_size):
        """Generates PSF frames and filtered localizations.

        Args:
            psf_frame_size (int): The size of the PSF frame.

        Returns:
            tuple: A tuple containing a list of arrays of PSF frames for each frame and a list of arrays of filtered localizations for each frame.
        """
        psf_frames = []
        filtered_localizations = []
        for i in range(len(self.movie)):
            movie_frame = self.movie[i]
            localizations = self.base_localizations[i]
            subimages = []
            sublocs = []
            for x, y in localizations:
                subimage = self.get_psf_frame(movie_frame, x, y, psf_frame_size, keep_edges=False)
                if subimage is not None:  # ignore localizations near the edge
                    subimages.append(subimage)
                    sublocs.append((x, y))
            psf_frames.append(np.array(subimages))
            filtered_localizations.append(np.array(sublocs))
        return psf_frames, filtered_localizations
    
    def get_psf_frame(self, movie_frame, x, y, frame_size, keep_edges=False):
        """Generates a PSF frame for a given localization.

        Args:
            movie_frame (array-like): The movie frame.
            x (float): The x-coordinate of the localization.
            y (float): The y-coordinate of the localization.
            frame_size (int): The size of the frame.
            keep_edges (bool, optional): Whether to keep localizations near the edge. Defaults to False.

        Returns:
            array-like: The PSF frame or None if keep_edges is False 
                and the frame overlaps the edge of the movie frame.
        """
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
                return None
        return np.array(movie_frame)[lower_y:upper_y, lower_x:upper_x]
    
    def view_frame(self, idx, size=1, boxes=False, dpi=150):
        """Displays a frame with localizations.

        Args:
            idx (int): The index of the frame.
            size (int, optional): The size of the markers for the localizations. Defaults to 1.
            boxes (bool, optional): Whether to draw boxes around the localizations. Defaults to False.
            dpi (int, optional): The resolution in dots per inch. Defaults to 150.
        """
        fig, ax = plt.subplots(1)
        fig.set_dpi(dpi)
        ax.imshow(self.movie[idx], cmap='gray')
        ax.set_title(f"Showing Frame {idx}: {len(self.filtered_localizations[idx])} Localizations")
        half_psf = int(self.psf_frame_size / 2)

        # Draw localizations
        for loc in self.filtered_localizations[idx]:
            x, y = loc
            if boxes:
                rect = patches.Rectangle((x-half_psf, y-half_psf), self.psf_frame_size, self.psf_frame_size, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            else:
                ax.plot(x, y, 'go', markersize=size)  # green dot

        plt.show()

    def show_psfs(self, random=True):
        """Displays a sample of PSF frames.

        Args:
            random (bool, optional): Whether to randomly select the PSF frames. Defaults to True.
        """
        print(self.all_psf_frames.shape)
        num_PSFs = 15
        if not random:
            random_seed = 1 # random seed for reproducibility
            np.random.seed(random_seed)
        PSF_indices = np.random.choice(self.all_psf_frames.shape[0], size=num_PSFs, replace=False)
        image_list = self.all_psf_frames[PSF_indices]

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

    def view_localizations_in_nm(self, size=0.001, point_color='white', background_color='black', dpi=150):
        """Displays all localizations in nm scale.

        Args:
            size (int, optional): The size of the markers for the localizations. Defaults to 0.001.
            point_color (str, optional): The color of the markers for the localizations. Defaults to 'white'.
            background_color (str, optional): The color of the background. Defaults to 'black'.
            dpi (int, optional): The resolution in dots per inch. Defaults to 150.
        """
        fig, ax = plt.subplots(1)
        fig.set_dpi(dpi)
        ax.set_title(f"All Localizations")
        ax.set_xlabel("x [nm]")
        ax.set_ylabel("y [nm]")
        # Scale localization coordinates to nm and split into x and y
        locs_nm = self.all_localizations * self.pixel_size
        x, y = locs_nm[:, 0], locs_nm[:, 1]
        
        # Draw all localizations at once using scatter
        ax.scatter(x, y, s=size, c=point_color)

        # Set background color to black
        ax.set_facecolor(background_color)

        plt.show()

    def __len__(self):
        """Returns the number of all localizations.

        Returns:
            int: The number of all localizations.
        """
        return len(self.all_localizations)
    
    def __getitem__(self, i):
        """Returns a tuple containing the frame, coordinates, and PSF of the localization 
            at the given index

        Args:
            i (int): The index.

        Returns:
            tuple: A tuple containing the frame, coordinates, and PSF frame
        """
        return (self.frames[i], self.all_localizations[i], self.all_psf_frames[i])