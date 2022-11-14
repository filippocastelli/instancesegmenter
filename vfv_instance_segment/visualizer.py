from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple
from collections import OrderedDict

from tqdm import tqdm
import PIL
import datashader as ds
import zetastitcher as zs
import colorcet as cc

class PointCloudVisualizer:
    def __init__(
        self,
        csv_dir_path: Path,
        stitchfile_path: Path = None,
        image_shape: tuple = None,
        axes = ("z", "y", "x")

    ):
        self.csv_dir_path = csv_dir_path
        self.csv_paths = sorted(list(self.csv_dir_path.glob("*.csv")))

        self.stitchfile_path = stitchfile_path

        # combine all csv files into one dataframe
        dataframes = []
        for csv_path in tqdm(self.csv_paths):
            df = pd.read_csv(csv_path, index_col="uuid", header=0, low_memory=False)
            dataframes.append(df)
        
        self.dataframe = pd.concat(dataframes, axis=0, ignore_index=False)
        # applying some basic refactoring
        self.dataframe = self._refactor_df(self.dataframe)

        if image_shape is not None:
            self.image_shape = image_shape
        elif self.stitchfile_path is not None:
            self.vfv = zs.VirtualFusedVolume(str(self.stitchfile_path))
            self.image_shape = self.vfv.shape
        else:
            raise ValueError("Either image_shape or stitchfile_path must be given.")

        self.axes = axes
        assert len(self.axes) == 3, "Only 3 axes are supported."
        for ax in axes:
            assert ax in ["x", "y", "z"], "Only x, y, and z axes are supported."


    @staticmethod
    def _refactor_df(df: pd.DataFrame) -> pd.DataFrame:
        """Refactor the dataframe to make it more suitable for visualization."""
        column_rename_dict = {
            "centroid_global_unsheared_unscaled-0" : "vfv-z",
            "centroid_global_unsheared_unscaled-1" : "vfv-y",
            "centroid_global_unsheared_unscaled-2" : "vfv-x",
            "centroid_global_sheared_scaled-0"     : "z",
            "centroid_global_sheared_scaled-1"     : "y",
            "centroid_global_sheared_scaled-2"     : "x",
        }

        df = df.rename(columns=column_rename_dict)
        return df

    def _get_subframe(self,
        axes: tuple = ("z", "y", "x"),
        bounds: dict = None,
        image_scale: float = 1.0) -> Tuple[pd.DataFrame, tuple]:
        """Get a subframe of the dataframe within the given bounds."""

        if bounds is None:
            mins = self.dataframe[[axes[0], axes[1], axes[2]]].values.min(axis=0)
            maxs = self.dataframe[[axes[0], axes[1], axes[2]]].values.max(axis=0)
            bounds_ordered = OrderedDict()
            bounds_ordered[axes[0]] = (mins[0], maxs[0])
            bounds_ordered[axes[1]] = (mins[1], maxs[1])
            bounds_ordered[axes[2]] = (mins[2], maxs[2])
        else:
            bounds_ordered = OrderedDict()
            bounds_ordered[axes[0]] = bounds[axes[0]]
            bounds_ordered[axes[1]] = bounds[axes[1]]
            bounds_ordered[axes[2]] = bounds[axes[2]]
        

        bounds_npy = np.array(list(bounds_ordered.values()))
        bound_shapes = bounds_npy[:, 1] - bounds_npy[:, 0]

        image_shape = (bound_shapes * image_scale)[:2].astype(int)

        subframe = self.dataframe[
            (self.dataframe[axes[0]] >= bounds_ordered[axes[0]][0]) & (self.dataframe[axes[0]] < bounds_ordered[axes[0]][1])
        ]
        subframe = subframe[
            (subframe[axes[1]] >= bounds_ordered[axes[1]][0]) & (subframe[axes[1]] < bounds_ordered[axes[1]][1])
        ]
        subframe = subframe[
            (subframe[axes[2]] >= bounds_ordered[axes[2]][0]) & (subframe[axes[2]] < bounds_ordered[axes[2]][1])
        ]

        return subframe, image_shape

    def visualize(self,
        axes: tuple = ("z", "y", "x"),
        image_scale: float = 1.0,
        bounds: dict = None,) ->  PIL.Image.Image:
        """Visualize the point cloud."""
        subframe, image_shape = self._get_subframe(axes, bounds, image_scale)
        canvas = ds.Canvas(plot_width=image_shape[0], plot_height=image_shape[1])
        agg = canvas.points(subframe, axes[0], axes[1])
        img = ds.tf.set_background(ds.tf.shade(agg, how="log", cmap=cc.fire), "black").to_pil()  # create a rasterized image
        return img




    