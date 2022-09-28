from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple
from functools import partial
import uuid
import warnings
import yaml
import multiprocessing
from multiprocessing import shared_memory
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import wait
import itertools
import time
#import warnings
#warnings.filterwarnings("error")
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from skimage.measure import regionprops, regionprops_table
from skimage.measure._regionprops import RegionProperties
from skimage.measure._regionprops import COL_DTYPES, PROPS
from skimage.measure._regionprops import _props_to_dict
from skimage import io as skio
import tifffile
import pyclesperanto_prototype as cle
from zetastitcher import VirtualFusedVolume, InputFile

from vfv_instance_segment.autocropper import AutoCropper
from vfv_instance_segment.integershearingcorrect import IntegerShearingCorrect

# monkeypatched properties
def err_minor_axis_length(self):
    try:
        minaxislenght = self.minor_axis_length
        return minaxislenght
    except Exception as e:
        return min(self.major_axis_length, self._spacing[0])

RegionProperties.err_minor_axis_length = property(err_minor_axis_length)
COL_DTYPES["err_minor_axis_length"] = float
PROPS["err_minor_axis_length"] = "err_minor_axis_length"

def centroid_unscaled(self):
    return tuple(self.coords.mean(axis=0))

RegionProperties.centroid_unscaled = property(centroid_unscaled)
COL_DTYPES["centroid_unscaled"] = float
PROPS["centroid_unscaled"] = "centroid_unscaled"

def inertia_tensor_eigvecs(self):
    # should be hermitian
    eigvals, eigvecs = np.linalg.eigh(self.inertia_tensor)
    return eigvecs

RegionProperties.inertia_tensor_eigvecs = property(inertia_tensor_eigvecs)
COL_DTYPES["inertia_tensor_eigvecs"] = float
PROPS["inertia_tensor_eigvecs"] = "inertia_tensor_eigvecs"

# extra properties
def n_slices(self) -> int:
    return self.image.shape[0]

RegionProperties.n_slices = property(n_slices)
COL_DTYPES["n_slices"] = str
PROPS["n_slices"] = "n_slices"

def label_uuid(self) -> str:
    return str(uuid.uuid4())
RegionProperties.uuid = property(label_uuid)
COL_DTYPES["uuid"] = str
PROPS["uuid"] = "uuid"

props = [
    "label",
    "area",
    "bbox",
    "centroid",
    "centroid_local",
    "centroid_unscaled",
    #"coords",
    #"coords_scaled",
    "equivalent_diameter_area",
    "extent",
    "inertia_tensor",
    "inertia_tensor_eigvals",
    "inertia_tensor_eigvecs",
    "moments",
    "moments_central",
    "slice",
    "major_axis_length",
    "err_minor_axis_length",
    "n_slices",
    #"uuid"
]

DF_LIGHT_COLS = [
    "label",
    "area",
    "uuid",
    "bbox-0",
    "bbox-1",
    "bbox-2",
    "bbox-3",
    "bbox-4",
    "centroid_global_unsheared_unscaled-0",
    "centroid_global_unsheared_unscaled-1",
    "centroid_global_unsheared_unscaled-2",
    "centroid_global_sheared_scaled-0",
    "centroid_global_sheared_scaled-1",
    "centroid_global_sheared_scaled-2",
    "slice",
    "vfv_slices-0",
    "vfv_slices-1",
    "vfv_slices-2",
    "major_axis_length",
    "minor_axis_length",
    "n_slices",
    "equivalent_diameter_area",
    "extent"
]
class SharedNumpyArray:
    def __init__(self, arr: np.ndarray, name: str = "shnp"):
        self.arr = arr
        self.dtype = arr.dtype
        self.shape = arr.shape
        self.dsize = np.dtype(arr.dtype).itemsize * np.prod(self.shape)
        self.name = f"{name}_{uuid.uuid4()}"
        self.dst = None
        self.get_shm()

    def get_shm(self):
        self.shm = shared_memory.SharedMemory(create=True, size=self.dsize, name=self.name)
        self.dst = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        self.dst[:] = self.arr[:]
        print(f"created shared array {self.shm.name} with shape {self.shape} and dtype {self.dtype}")
        print(f"size: {(self.dst.nbytes / 1024) / 1024} MB")

    def release(self):
        self.shm.close()
        self.shm.unlink()

class VirtualFusedVolumeVoronoiSegmenter:

    def __init__(self,
        in_fpath: Path,
        output_path: Path = None,
        z_overlap: int = 5,
        y_overlap: int = 5,
        y_batch_size: int = 2048,
        z_batch_size: int = 200,
        y_matching_batch_size: int = 2000,
        sigma_detection: float = 5.,
        sigma_outline: float = 1.,
        device_id: str = None,
        autocrop: bool = True,
        shear_shift: int = -7,
        voxel_size: tuple = (3.64, 0.52, 0.52),
        n_workers: int = -1,
        n_saving_workers: int = -1,
        df_light_cols: list = None,
        resume: bool = False,
        ):

        self.in_fpath = in_fpath
        self.output_path = output_path if output_path is not None else in_fpath.parent.joinpath(in_fpath.stem)
        self.output_path.mkdir(exist_ok=True, parents=True)

        self.out_csv_path = self.output_path.joinpath("stats.csv")
        self.out_stitchfile_path = self.output_path.joinpath("out_stitchfile.yml")
        self.z_overlap = z_overlap
        self.y_overlap = y_overlap

        self.z_batch_size = z_batch_size
        self.y_batch_size = y_batch_size
        self.y_matching_batch_size = y_matching_batch_size

        if self.in_fpath.suffix == ".yml":
            self.vfv = VirtualFusedVolume(str(self.in_fpath))
        else:
            self.vfv = InputFile(self.in_fpath)
        
        self.sigma_detection = sigma_detection
        self.sigma_outline = sigma_outline

        self.autocrop = autocrop
        self.shear_shift = shear_shift

        device_id = device_id if device_id is not None else 0
        dev = cle.select_device(device_id)

        self.device = dev.name

        self.voxel_size = voxel_size
        self.filematrix_dict = {}

        self.n_workers = n_workers if n_workers > 0 else multiprocessing.cpu_count()
        self.n_saving_workers = n_saving_workers if n_saving_workers > 0 else multiprocessing.cpu_count()
        self.df_light_cols = df_light_cols if df_light_cols is not None else DF_LIGHT_COLS

        self.resume = resume

    @staticmethod
    def _get_volume_slices(
        volume_shape: tuple,
        df: pd.DataFrame,
        label_str: str,
        selection_width: tuple = (20, 50, 50)) -> Tuple[slice, slice, slice]:
        """
        Given a volume and a label, return the slices that contain the label.
        """
        # get the label
        label = int(label_str.split("_")[0])

        label_df = df.loc[df.label == label]
        if len(label_df) == 0:
           warnings.warn(f"Label {label} not found in dataframe")
           return None
        
        if len(label_df) > 1:
            warnings.warn(f"Label {label} found multiple times in dataframe")

        label_series = label_df.iloc[0]
        centroid = []
        for i in range(3):
            centroid.append(int(label_series[f"centroid_stack_unsheared_unscaled-{i}"]))
        
        z_slice = slice(min(centroid[0] - selection_width[0], 0), max(centroid[0] + selection_width[0], volume_shape[0]))
        y_slice = slice(min(centroid[1] - selection_width[1], 0), max(centroid[1] + selection_width[1], volume_shape[1]))
        x_slice = slice(min(centroid[2] - selection_width[2], 0), max(centroid[2] + selection_width[2], volume_shape[2]))

        return z_slice, y_slice, x_slice


    @classmethod
    def _get_matching_worker(cls,
        idxs: tuple,
        stack_labels: list,
        stack_labels_str: list,
        overlap_labels: list,
        overlap_labels_str: list,
        stack_shm_name: str,
        overlap_shm_name: str,
        arr_shape: tuple,
        shm_dtype: str,
        df_overlap: pd.DataFrame):

        i, j = idxs
        stack_label = stack_labels[i]
        stack_label_str = stack_labels_str[i]
        overlap_label = overlap_labels[j]
        overlap_label_str = overlap_labels_str[j]
        stack_shm = shared_memory.SharedMemory(name=stack_shm_name)
        overlap_shm = shared_memory.SharedMemory(name=overlap_shm_name)
        stack_array = np.ndarray(arr_shape, dtype=shm_dtype, buffer=stack_shm.buf)
        overlap_array = np.ndarray(arr_shape, dtype=shm_dtype, buffer=overlap_shm.buf)

        centroid_slices = cls._get_volume_slices(arr_shape, df_overlap, overlap_label_str)
        if centroid_slices is None:
            return None
        
        stack_sub = stack_array[centroid_slices[0], centroid_slices[1], centroid_slices[2]]
        overlap_sub = overlap_array[centroid_slices[0], centroid_slices[1], centroid_slices[2]]
        binarized_stack = np.where(stack_sub>0, 1, 0)
        binarized_overlap = np.where(overlap_sub>0, 1, 0)
        intersection_arr = binarized_stack*binarized_overlap
        n_intersect = np.sum(intersection_arr[overlap_sub==overlap_label])

        if n_intersect > 0:
            n_union = np.sum(binarized_stack[stack_sub==stack_label]) + np.sum(binarized_overlap[overlap_sub==overlap_label]) - n_intersect
            if n_union == 0:
                return None
            else:
                jaccard_index = n_intersect / n_union
                return (overlap_label_str, stack_label_str, jaccard_index)
        else:
            return None
    
    @classmethod
    def _get_label_matching(cls,
        stack: np.ndarray,
        overlap: np.ndarray,
        df_stack: pd.DataFrame = None,
        df_overlap: pd.DataFrame = None,
        n_workers: int = 1) -> list:
        """
        Get best matching labels between the stack and the overlap
        :param stack: the stack to get the overlap from
        :param overlap: the overlap to get the overlap from
        :param df_overlap: overlap df, optional
        :param df_stack: stack df, optional
        :return: the overlap between the labels in the stack and the overlap
        """

        assert stack.shape == overlap.shape, f"stack shape {stack.shape} != overlap shape {overlap.shape}"

        stack_shape = stack.shape
        # get the labels in the overlap
        overlap_labels = df_overlap["label"].values if df_overlap is not None else np.unique(overlap)
        overlap_labels = overlap_labels[overlap_labels != 0]
        overlap_labels_str = [f"{l}_overlap" for l in overlap_labels] 

        # get the overlap between the labels in the stack and the labels in the overlap
        stack_labels = df_stack["label"].values if df_stack is not None else np.unique(stack)
        stack_labels = stack_labels[stack_labels != 0]
        stack_labels_str = [f"{l}_stack" for l in stack_labels]


        label_graph = nx.Graph()
        label_graph.add_nodes_from(stack_labels_str)
        label_graph.add_nodes_from(overlap_labels_str)

        binarized_stack = np.where(stack>0, 1, 0)
        binarized_overlap = np.where(overlap>0, 1, 0)

        intersection = binarized_stack * binarized_overlap
        edge_list = []
        
        enable_df_search = df_overlap is not None and df_stack is not None
        
        if n_workers > 1 and enable_df_search:
            shared_stack = SharedNumpyArray(stack, name="sh_stack_matching")
            shared_overlap = SharedNumpyArray(overlap, name="sh_overlap_matching")

            matching_worker_partial = partial(cls._get_matching_worker,
                stack_labels=stack_labels,
                stack_labels_str=stack_labels_str,
                overlap_labels=overlap_labels,
                overlap_labels_str=overlap_labels_str,
                stack_shm_name=shared_stack.name,
                overlap_shm_name=shared_overlap.name,
                arr_shape=stack_shape,
                shm_dtype=stack.dtype,
                df_overlap=df_overlap)
                
            idxlist = list(itertools.product(range(len(stack_labels)), range(len(overlap_labels))))
            futures = []

            SINGLE_THREAD_DEBUG = False
            if SINGLE_THREAD_DEBUG:
                for idxtuple in tqdm(idxlist):
                    edge_list.append(matching_worker_partial(idxtuple))
                edge_list = [e for e in edge_list if e is not None]
            else:
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    print(f"Starting {n_workers} workers for label matching...")
                    results = list(tqdm(executor.map(matching_worker_partial, idxlist), total=len(idxlist)))
                    
                    #for idxs in idxlist:
                    #    futures.append(executor.submit(matching_worker_partial, idxs=idxs))
                    #futures, _ = wait(futures)
                #edge_list = [future.result() for future in futures]
                edge_list = [edge for edge in results if edge is not None]
            
            shared_stack.release()
            shared_overlap.release()
            
        else:
        # can possibly be parallelized using multiprocessing
            for i, label_o_str in enumerate(overlap_labels_str):
                label_o = overlap_labels[i]
                for j, label_s_str in enumerate(stack_labels_str):
                    label_s = stack_labels[j]
                    # get number of voxels in the intersection
                    if enable_df_search:
                        centroid_slices = cls._get_volume_slices(stack_shape, df_overlap, label_o_str)
                        if centroid_slices is None:
                            # mismatch between df and stack, ignore this label
                            continue
                        intersect_arr = intersection[centroid_slices[0], centroid_slices[1], centroid_slices[2]]
                        overlap_arr = overlap[centroid_slices[0], centroid_slices[1], centroid_slices[2]]
                        n_intersect = np.sum(intersect_arr[overlap_arr==label_o])
                    else:
                        n_intersect = np.sum(intersection[overlap==label_o])

                    if n_intersect > 0:
                        if df_overlap is not None and df_stack is not None:
                            binarized_stack_arr = binarized_stack[centroid_slices[0], centroid_slices[1], centroid_slices[2]]
                            binarized_overlap_arr = binarized_overlap[centroid_slices[0], centroid_slices[1], centroid_slices[2]]
                        else:
                            binarized_stack_arr = binarized_stack
                            binarized_overlap_arr = binarized_overlap
                        # if there is an intersection get number of union voxels
                        n_union = np.sum(binarized_stack_arr[stack==label_s]) + np.sum(binarized_overlap_arr[overlap==label_o]) - n_intersect
                        # mismatch between df and actual stack
                        if n_union == 0:
                            jaccard_index = 0
                        # get the jaccard index
                        else:
                            jaccard_index = n_intersect / n_union
                            edge_list.append((label_o_str, label_s_str, jaccard_index))
                    else:
                        jaccard_index = 0
                        pass

        label_graph.add_weighted_edges_from(edge_list)
        matching = nx.max_weight_matching(label_graph, maxcardinality=True)
        # order the matching by the stack labels
        label_val = lambda label: int(float(label.split("_")[0]))
        label_group = lambda label: label.split("_")[1]
        val_tuple = lambda tup: (label_val(tup[0]), label_val(tup[1]))
        invert_tuple = lambda tup: (tup[1], tup[0])
        order_tuple = lambda tup: val_tuple(tup) if label_group(tup[0]) == "stack" else invert_tuple(val_tuple(tup))
        #order_tuple = lambda x: (x[0], x[1]) if x[0] in stack_labels else (x[1], x[0])
        return [order_tuple(x) for x in matching]
    
    @classmethod
    def _segment_substack(cls,
        stack: np.ndarray,
        overlap_arr: np.ndarray = None,
        start_label: int = 0,
        sigma_detection: float = 5.,
        sigma_outline: float = 1., 
        autocrop: bool = True,
        shear_shift: int = -7,
        offset_unscaled: tuple = (0, 0, 0),
        voxel_size: tuple = (3.64, 0.52, 0.52),
        stack_slices: tuple = None,
        n_workers: int = 1) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Segment the stack using the voronoi algorithm
        :param stack: the stack to segment
        :return: the segmented stack
        """
        # autocrop the stack
        if autocrop:
            cropper = AutoCropper(stack)
            cropped_vol = cropper.crop()
        else:
            cropped_vol = stack

        # apply shearing correction
        shearing_corrector = IntegerShearingCorrect(direction='x', delta=shear_shift)
        corrected_vol, _ = shearing_corrector.forward_correct(cropped_vol)

        # push the stack to the GPU
        stack_gpu = cle.push(corrected_vol)

        # segment the stack
        segmented_stack_gpu = cle.voronoi_otsu_labeling(stack_gpu, spot_sigma=sigma_detection, outline_sigma=sigma_outline)

        # pull the segmented stack back from the GPU
        segmented_stack = cle.pull(segmented_stack_gpu)
        segmented_stack_arr = np.array(segmented_stack)
        
        if start_label > 0:
            segmented_stack_arr = np.where(segmented_stack_arr > 0, segmented_stack_arr + start_label, 0)
        
        unsheared_segmented_stack = shearing_corrector.inverse_correct(segmented_stack_arr)
        if autocrop: 
            unsheared_segmented_stack = cropper.uncrop(unsheared_segmented_stack)

        n_overlap = 0
        if overlap_arr is not None:
            n_overlap = overlap_arr.shape[0]
            # if autocrop:
            #     overlap_cropper = AutoCropper(overlap_arr)
            #     overlap_cropper.autocrop_range = cropper.autocrop_range
            #     cropped_overlap = overlap_cropper.crop()
            # else:
            #     cropped_overlap = overlap_arr
            
            # apply shearing correction
            segmented_overlap = unsheared_segmented_stack[-n_overlap:]
            assert segmented_overlap.shape == overlap_arr.shape, f"segmented overlap shape {segmented_overlap.shape} != cropped overlap shape {overlap_arr.shape}"

            # there's no need to apply shearing correction to the overlap
            #isc = IntegerShearingCorrect(direction='x', delta=shear_shift)
            #corrected_overlap, _ = isc.forward_correct(cropped_overlap)
            #corrected_segmented_overlap, _ = isc.forward_correct(segmented_overlap)

            matching = cls._get_label_matching(segmented_overlap, overlap_arr, n_workers=n_workers)

            for label_tuple in matching:
                segmented_stack_arr[segmented_stack_arr == label_tuple[0]] = label_tuple[1]

        # transform the segmented stack back to the original shape
        # THIS DF REFERS TO CROPPED SHEARING TRANSFORMED STACK
        # we're ok with shearing transform but we need to undo cropping
        # segmented_df =  pd.DataFrame(cle.statistics_of_labelled_pixels(stack, segmented_stack_arr))
        # segmented_df = segmented_df[segmented_df.label != 0]

        segmented_df = cls._get_stack_df(
            segmented_stack_arr,
            voxel_size=voxel_size,
            n_overlap=n_overlap,
            offset_unscaled=offset_unscaled,
            shear_shift=shear_shift,
            stack_slices=stack_slices,
            crop_offset=cropper.pads[2][0])


        segmented_stack_arr = shearing_corrector.inverse_correct(segmented_stack_arr)
        if autocrop: 
            segmented_stack_arr = cropper.uncrop(segmented_stack_arr)

        DEBUG_PLOTS= False
        if DEBUG_PLOTS:
            from matplotlib import pylab as plt
            
            arr_0, arr_1 = cls._get_centroids_by_z(segmented_df, 8, .5, key="centroid_stack_unsheared_unscaled", xoff=0)
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(segmented_stack_arr[8])
            ax[1].imshow(segmented_stack_arr[8])
            ax[1].scatter(arr_1, arr_0, c='r')
        return segmented_stack_arr, segmented_df
    @staticmethod
    def _get_centroids_by_z(df: pd.DataFrame, z: int, delta: int, key="centroid_stack_unsheared_unscaled", xoff = -351) -> tuple:
        slice_df = df[df[f"{key}-0"] > z-delta ][df[f"{key}-0"] < z+delta]
        arr_0 = slice_df[f"{key}-1"].values
        arr_1 = slice_df[f"{key}-2"].values + xoff
        return arr_0, arr_1
    @classmethod
    def _get_stack_df(cls,
        segmented_stack: np.ndarray,
        voxel_size: tuple = (3.6, 0.52, 0.52),
        n_overlap: int = 0,
        offset_unscaled: tuple = None,
        shear_shift: int = 0,
        stack_slices: tuple = None,
        crop_offset: int = None) -> pd.DataFrame:

        rprops = regionprops(segmented_stack, spacing=voxel_size)
        if len(rprops) > 0:
            rprops_dict = _props_to_dict(rprops, properties=props)
            stack_df = pd.DataFrame(rprops_dict)
            stack_df = stack_df[stack_df.label != 0]
            stack_df.rename(columns={"err_minor_axis_length": "minor_axis_length"}, inplace=True)
            stack_df['uuid'] = stack_df.apply(lambda _: uuid.uuid4(), axis=1)
            
            # creating stack sheared column
            for i in range(3):
                stack_df.rename(columns={f"centroid_unscaled-{i}": f"centroid_stack_sheared_unscaled-{i}"}, inplace=True)
                stack_df.rename(columns={f"centroid-{i}": f"centroid_stack_sheared_scaled-{i}"}, inplace=True)
            
            get_cols_partial = partial(cls._get_coords_df_row,
                shear_shift=shear_shift,
                crop_offset_x=crop_offset,
                global_offset=offset_unscaled,
                voxel_size=voxel_size,
                arr_depth=segmented_stack.shape[0])

            stack_df = stack_df.join(stack_df.apply(get_cols_partial, axis=1))

            if n_overlap > 0:
                # dropping neurons in the upper overlap area
                stack_df = stack_df[stack_df["centroid_stack_unsheared_unscaled-0"] < segmented_stack.shape[0] - n_overlap]
                # no point in 
                return stack_df
            
            # we include the original slices in case we need to retrieve the original stack
            # all local coords should refer to the original sheared stack
            if stack_slices is not None:
                for i in range(3):
                    stack_df[f"vfv_slices-{i}"] = stack_slices[i]
        else:
            stack_df = pd.DataFrame()

        return stack_df

    @classmethod
    def _get_coords_df_row(cls, df_row, shear_shift, crop_offset_x, global_offset, voxel_size, arr_depth):
        coord_tuple = (df_row["centroid_stack_sheared_unscaled-0"], df_row["centroid_stack_sheared_unscaled-1"], df_row["centroid_stack_sheared_unscaled-2"])
        
        label_list = []
        value_list = []

        stack_cropped_sheared = list(coord_tuple)
        stack_cropped_unsheared = list(cls._shear_coord(stack_cropped_sheared, shear_shift, back_transform=True))

        #for i in range(3):
        #    label_list.append(["centroid_stack_unsheared_crop-{}".format(i)])
        #    value_list.append([stack_cropped_unsheared[i]])

        stack_uncropped_unsheared = stack_cropped_unsheared.copy()
        stack_uncropped_unsheared[2] += crop_offset_x + arr_depth*shear_shift

        for i in range(3):
            label_list.append("centroid_stack_unsheared_unscaled-{}".format(i))
            value_list.append(stack_uncropped_unsheared[i])

        global_uncropped_unsheared = stack_uncropped_unsheared.copy()
        for i in range(3):
            global_uncropped_unsheared[i] += global_offset[i]
            label_list.append("centroid_global_unsheared_unscaled-{}".format(i))
            value_list.append(global_uncropped_unsheared[i])
        
        global_uncropped_sheared = list(cls._shear_coord(global_uncropped_unsheared, shear_shift, back_transform=False))
        for i in range(3):
            label_list.append("centroid_global_sheared_unscaled-{}".format(i))
            value_list.append(global_uncropped_sheared[i])

        global_uncropped_sheared_scaled = global_uncropped_sheared.copy()
        for i in range(3):
            global_uncropped_sheared_scaled[i] *= voxel_size[i]
            label_list.append("centroid_global_sheared_scaled-{}".format(i))
            value_list.append(global_uncropped_sheared_scaled[i])

        return pd.Series(value_list, index=label_list)

    @staticmethod
    def _get_overlapping_slices(range_len: int, batch_size: int, overlap: int) -> slice:
        end = 0
        start = 0
        while end < range_len:
            start = end - overlap
            if start < 0:
                start = 0
            end = start + batch_size
            if end > range_len:
                end = range_len
            yield slice(start, end, 1)

    @staticmethod
    def _shear_coord(coord: tuple, shear_shift:int, back_transform: bool = False) -> tuple:
        shear_shift = -shear_shift if back_transform else shear_shift
        return (coord[0], coord[1], coord[2] + shear_shift*coord[0])

    @staticmethod
    def _get_overlap_df(df: pd.DataFrame, y_offset: int, y_depth: tuple, overlap_size: int, mode="upper") -> pd.DataFrame:
        assert mode in ["upper", "lower"], f"mode must be 'upper' or 'lower', not {mode}"
        overlap_df = df.copy()
        if mode == "upper":
            overlap_df = overlap_df[overlap_df["centroid_global_unsheared_unscaled-1"] > y_offset + y_depth - overlap_size]
        if mode == "lower":    
            overlap_df = overlap_df[overlap_df["centroid_global_unsheared_unscaled-1"] < y_offset + overlap_size]
        
        return overlap_df

    @staticmethod
    def _get_overlap_df2(df: pd.DataFrame, vfv_shape: tuple, y_overlap: int, mode="upper"):

        assert mode in ["upper", "lower"], f"mode must be 'upper' or 'lower', not {mode}"

        overlap_df = df.copy()
        if mode == "upper":
            # select only centroids in the upper overlap
            # we need to drop all centroids that are in the lower overlap
            overlap_df = overlap_df[overlap_df[f"centroid_global_unsheared_unscaled-1"] > vfv_shape[1] - y_overlap]
        elif mode == "lower":
            overlap_df = overlap_df[overlap_df[f"centroid_global_unsheared_unscaled-1"] < y_overlap]

        # if we've done everything right, centroids should share the same coordinates in the overlap
        return overlap_df
            
    def _segment_z_stack(self, y_slice: slice, start_label: int = None) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        # define x batches
        z_overlapping_slices = list(self._get_overlapping_slices(self.vfv.shape[0], self.z_batch_size, self.z_overlap))
        z_overlap_arr = None
        z_slice_df = pd.DataFrame()
        x_slice = slice(0, None, 1)

        y_overlap_arr_upper = np.zeros(shape=(self.vfv.shape[0], self.y_overlap, self.vfv.shape[2]), dtype=self.vfv.dtype) if y_slice.stop != self.vfv.shape[1] else None
        y_overlap_arr_lower = np.zeros(shape=(self.vfv.shape[0], self.y_overlap, self.vfv.shape[2]), dtype=self.vfv.dtype) if y_slice.start > 0 else None

        start_label = start_label if start_label is not None else 1
        sub_arr_shapes = []
        for idx, z_slice in enumerate(tqdm(z_overlapping_slices)):
            # get the substack
            substack = self.vfv[z_slice, y_slice, x_slice] # [xbatch, ybatch, 2048]
            # segment the substack
            offset_unscaled = (z_slice.start, y_slice.start, 0)

            print(f"predicting {offset_unscaled}...")
            segmented_substack, substack_df = self._segment_substack(
                substack,
                overlap_arr=z_overlap_arr,
                sigma_detection=self.sigma_detection,
                sigma_outline=self.sigma_outline,
                autocrop=self.autocrop,
                offset_unscaled=offset_unscaled,
                voxel_size=self.voxel_size,
                stack_slices=(z_slice, y_slice, x_slice),
                start_label=start_label,
                n_workers=self.n_workers)

            z_border = self.z_overlap if idx < len(z_overlapping_slices) -1 else None
            y_border = self.y_overlap if y_slice.stop != self.vfv.shape[1] else None

            sub_path, sub_arr_shape = self._save_substack(segmented_substack, z_border=z_border, y_border=y_border, offset=offset_unscaled)
            sub_arr_shapes.append(sub_arr_shape)

            # get the overlap
            if y_overlap_arr_upper is not None:
                # we want to avoid discarding the upper border for the last slice
                # y_border is None for the last slice.
                y_overlap_arr_upper[z_slice, :, :] = segmented_substack[:, -y_border:, :]
            if y_overlap_arr_lower is not None:
                # we never need to discard the lower border
                y_overlap_arr_lower[z_slice, :, :] = segmented_substack[:, :self.y_overlap, :]
            
            if y_overlap_arr_upper is not None and y_overlap_arr_lower is not None:
                assert y_overlap_arr_lower.shape == y_overlap_arr_upper.shape, f"{y_overlap_arr_lower.shape} != {y_overlap_arr_upper.shape}"

            # add the substack df to the stack df
            z_slice_df = pd.concat([z_slice_df, substack_df], ignore_index=True)
            # DATAFRAME REFERS TO THE SHEARING_CORRECTED STACK
            if idx > 0:
                start_label = np.uint32(z_slice_df.label.max() + 1) if "label" in z_slice_df else 1
                # this should roll when we get to uint32.max

        shapes_arr = np.array(sub_arr_shapes)
        z_length = np.sum(shapes_arr, axis=0)[0]
        saved_substack_shape = (z_length, shapes_arr[0, 1], shapes_arr[0, 2])
        assert z_length == self.vfv.shape[0], f"lengths[0] ({z_length}) != self.vfv.shape[0] ({self.vfv.shape[0]})"
        assert len(list(sub_path.glob("*.tif"))) == self.vfv.shape[0], "not all z slices were saved"
        self.filematrix_dict[(0,*offset_unscaled[1:])] = sub_path, saved_substack_shape
        return z_slice_df, y_overlap_arr_upper, y_overlap_arr_lower, saved_substack_shape

    
    @staticmethod
    def _save_substack_worker(
        idxlist: list,
        shm_name: str,
        arr_shape: tuple,
        shm_dtype: str,
        offset: int,
        out_fpath: Path,
        compression: str = "zlib"):

        shm = shared_memory.SharedMemory(name=shm_name)
        np_array = np.ndarray(arr_shape, dtype=shm_dtype, buffer=shm.buf)
        for z in idxlist:
            img_output_path = out_fpath.joinpath("{:06d}.tif".format(z+offset[0]))
            tifffile.imwrite(str(img_output_path), np_array[z], compression=compression)
        
        return True
    
    @staticmethod
    def _get_stack_fname(offset: tuple) -> str:
        return f"z_{0:06d}_y_{offset[1]:06d}_x_{offset[2]:06d}"
    
    def _save_substack(self,
        substack: np.ndarray,
        z_border: int = None,
        y_border: int = None,
        offset: tuple = (0,0,0)) -> Path:

        #out_fname = f"z_{offset[0]:06d}_y_{offset[1]:06d}_x_{offset[2]:06d}"
        # saving in same z folders so we have complete z-stacks
        out_fname = self._get_stack_fname(offset)
        out_fpath = self.output_path.joinpath(out_fname)
        out_fpath.mkdir(parents=True, exist_ok=True)

        if z_border is not None and z_border > 0:
            substack = substack[:-z_border, :, :]
        else:
            z_border = 0
        
        if y_border is not None and y_border > 0:
            substack = substack[:, :-y_border, :]
        else:
            y_border = 0
        
        print("writing substack to disk...")
        if self.n_saving_workers is not None and self.n_saving_workers > 1:
            start = time.time()
            shared_np_arr = SharedNumpyArray(substack)
            sub_worker_partial = partial(self._save_substack_worker,
                out_fpath=out_fpath,
                offset=offset,
                shm_name=shared_np_arr.name,
                arr_shape=shared_np_arr.shape,
                shm_dtype=shared_np_arr.dtype,
                compression="zlib")

            idxlist = list(range(substack.shape[0]))
            futures = []
            chunks = [idxlist[i:i+self.n_saving_workers] for i in range(0, len(idxlist), self.n_saving_workers)]
            with ProcessPoolExecutor(max_workers=self.n_saving_workers) as executor:
                for chunk in chunks:
                    futures.append(executor.submit(sub_worker_partial, idxlist=chunk))
                futures, _ = wait(futures)
            
            shared_np_arr.release()
            end = time.time()
            print(f"writing substack to disk took {end-start:.2f}s")
        else:
            start = time.time()
            for idx, img in enumerate(substack):
                #skio.imsave(str(out_fpath.joinpath("{:02d}.tif".format(idx))), img, plugin="tifffile", compress="zlib")
                #print(img.shape)
                tifffile.imwrite(str(out_fpath.joinpath("{:06d}.tif".format(idx+offset[0]))), img, compression="zlib")
            end = time.time()
            print(f"writing substack to disk took {end-start:.2f}s")

        return out_fpath, substack.shape

    @classmethod
    def _batched_label_matching(cls,
        stack: np.ndarray,
        overlap: np.ndarray,
        batch_size: int,
        df_stack: pd.DataFrame = None,
        df_overlap: pd.DataFrame = None,
        ) -> list:

        slices = cls._get_overlapping_slices(stack.shape[0], batch_size=batch_size, overlap=0)
        global_matching = []
        print("Calculating y label matching...")
        for _slice in tqdm(slices):
            if df_stack is not None and df_overlap is not None:
                _get_subdf = lambda df, sel_slice : df[df["centroid_global_unsheared_unscaled-0"] > sel_slice.start][df["centroid_global_unsheared_unscaled-0"]< sel_slice.stop]
                b_df_stack = _get_subdf(df_stack, _slice)
                b_df_overlap = _get_subdf(df_overlap, _slice)
            else:
                b_df_stack = None
                b_df_overlap = None

            b_stack = stack[_slice, :, :]
            b_overlap = overlap[_slice, :, :]

            global_matching.append(cls._get_label_matching(b_stack, b_overlap, b_df_stack, b_df_overlap))

        return global_matching
        
    def segment_vfv(self) -> Tuple[pd.DataFrame,VirtualFusedVolume]:
        overlapping_y_slices = list(self._get_overlapping_slices(self.vfv.shape[1], self.y_batch_size, self.y_overlap))

        
        # next_y_overlap_arr_low
        # ----------------------
        # stack_1
        # ----------------------
        # y_overlap_arr_up
        # ----------------------
        # xxxxxxxxxxxxxxxxxxxxxx
        # ----------------------
        # y_overlap_arr_low
        # ----------------------
        # stack_0
        # ----------------------
        # previous_y_overlap_arr_low

        y_overlap_arr_upper = None
        y_overlap_arr_lower = None
        previous_y_overlap_arr_upper = None
        
        vfv_df = pd.DataFrame()
        stack_df = None
        start_label = 0
        y_overlap_df_upper = None
        previous_y_overlap_arr_upper = None
        saved_stack_shapes = []

        for idx, y_slice in enumerate(overlapping_y_slices):
            print(f"Segmenting y-stack {idx}/{len(overlapping_y_slices)-1}..." )
            
            out_stack_fname = self._get_stack_fname((0, y_slice.start, 0))
            out_stack_fpath = self.output_path.joinpath(out_stack_fname)
            df_bck_path = self.out_csv_path.parent.joinpath(f"df_bck_{idx}.csv")

            if self.resume and out_stack_fpath.exists() and len(list(out_stack_fpath.glob("*.tif"))) == self.vfv.shape[0] and df_bck_path.exists():
                print("Skipping y-stack prediction...")
                prev_stack_inpf = InputFile(out_stack_fpath)
                saved_stack_shape = prev_stack_inpf.shape
                self.filematrix_dict[(0,y_slice.start, 0)] = out_stack_fpath, saved_stack_shape
                saved_stack_shapes.append(prev_stack_inpf.shape)
                
                y_border = self.y_overlap if y_slice.stop != self.vfv.shape[1] else None
                previous_y_overlap_arr_upper = prev_stack_inpf[-y_border:,:] if y_border is not None else prev_stack_inpf[:]
                stack_df = pd.read_csv(df_bck_path)

            else:
                stack_df, y_overlap_arr_upper, y_overlap_arr_lower, saved_stack_shape = self._segment_z_stack(y_slice, start_label=start_label)
                #def _get_overlap_df(df: pd.DataFrame, stack_offset: tuple, z_stack_shape: tuple, overlap_size: int, mode="upper") -> pd.DataFrame:
                saved_stack_shapes.append(saved_stack_shape)
            
            y_overlap_df_upper = self._get_overlap_df(stack_df,
                y_offset=y_slice.start,
                y_depth=saved_stack_shape[1],
                overlap_size=self.y_overlap,
                mode="upper")
            y_overlap_df_lower = self._get_overlap_df(stack_df,
                y_offset=y_slice.start,
                y_depth=saved_stack_shape[1],
                overlap_size=self.y_overlap,
                mode="lower")

            if y_overlap_arr_lower is not None:
                # if we have a previous overlap lower, we need to compute the matching
                if previous_y_overlap_arr_upper is not None:
                    # compute matching between previous overlap lower and current overlap upper
                    matching = self._get_label_matching(
                        y_overlap_arr_lower,
                        previous_y_overlap_arr_upper,
                        df_stack=y_overlap_df_lower,
                        df_overlap=previous_y_overlap_df_upper,
                        n_workers=self.n_workers)
                        #batch_size=self.y_matching_batch_size)
                    
                    for label_tuple in matching:
                        label_stack, label_vfv = label_tuple
                        uuiddf = vfv_df["uuid"][vfv_df["label"] == label_vfv]
                        uuid_vfv = uuiddf.iloc[0]
                        stack_df.loc[stack_df["label"] == label_stack, "uuid"] = uuid_vfv
                        stack_df.loc[stack_df["label"] == label_stack, "label"] = label_vfv
                    
                    # drop vfv uuids in the overlap region
                    # if we're in the last slice, we don't need to drop anything
            if idx != len(overlapping_y_slices):
                if len(vfv_df) > 0:
                    vfv_df = vfv_df[vfv_df["centroid_global_unsheared_unscaled-1"] < y_slice.stop - self.y_overlap]
                vfv_df = pd.concat([vfv_df, stack_df])

                previous_y_overlap_arr_upper = y_overlap_arr_upper
                previous_y_overlap_df_upper = y_overlap_df_upper.copy()

            # saving backup of dataframe for each stack
            self._save_df(stack_df, df_bck_path)


        y_stack_shapes_arr = np.array(saved_stack_shapes)
        y_len = np.sum(y_stack_shapes_arr[:, 1])
        assert y_len == self.vfv.shape[1], f"y_len {y_len} != vfv.shape[1] {self.vfv.shape[1]}"

        out_vfv = self.write_out_stitchfile()
        assert out_vfv.shape == self.vfv.shape, f"out_vfv.shape {out_vfv.shape} != vfv.shape {self.vfv.shape}"
        self._save_df(vfv_df, path=self.out_csv_path)
        return out_vfv, vfv_df

    def _save_df(self, df: pd.DataFrame, path: Path) -> None:
        if path.exists():
            path.unlink()
        df.to_csv(path)
        light_df = df[self.df_light_cols]
        light_df.to_csv(path.with_suffix(".light.csv"))

    def write_out_stitchfile(self) -> VirtualFusedVolume:
        if self.out_stitchfile_path.exists():
            self.out_stitchfile_path.unlink()
        filedicts = []
        for offset, out_arr_tuple in self.filematrix_dict.items():
            fpath, out_arr_shape = out_arr_tuple
            fpaths = [fpath for fpath in fpath.glob("*.tif")]
            first_frame = skio.imread(fpaths[0])

            filedicts.append({
                "filename": str(fpath.relative_to(self.output_path)),
                "Xs": offset[2],
                "X": offset[2]*self.voxel_size[2]/1e3,
                "Ys": offset[1],
                "Y": offset[1]*self.voxel_size[1]/1e3,
                "Zs": offset[0],
                "Z": offset[0]*self.voxel_size[0]/1e3,
                "nfrms": len(fpaths),
                "xsize": first_frame.shape[1],
                "ysize": first_frame.shape[0]}
                )

        with self.out_stitchfile_path.open(mode="w") as f:
            yaml.safe_dump({"filematrix": filedicts}, f)

        out_vfv = VirtualFusedVolume(str(self.out_stitchfile_path),weighting_mode="none")
        return out_vfv

def main():
    parser = ArgumentParser()
    parser.add_argument('in_fpath', metavar='in_fpath', type=str, nargs='+', help='input file path')
    parser.add_argument('-o', '--output', dest='output', help='output file path', default=None, required=False)
    parser.add_argument('-zo', '--z-overlap',dest='zoverlap', type=int, default=5, help="z overlap", required=False)
    parser.add_argument('-yo', '--y-overlap',dest='yoverlap', type=int, default=5, help="y overlap", required=False)
    parser.add_argument('-zb', '--z-batch', dest='zbatch', type=int, default=100, help="z batch size", required=False)
    parser.add_argument('-yb', '--y-batch', dest='ybatch', type=int, default=2048, help="y batch size", required=False)
    parser.add_argument('-ymb' '--y-matching-batch', dest='ymatchingbatch', type=int, default=2048, help="x batch size for y stack matching", required=False)
    parser.add_argument('-sd', '--sigma-detection', dest='sigmadetection', type=float, default=5., help="sigma detection", required=False)
    parser.add_argument('-so', '--sigma-outline', dest='sigmaoutline', type=float, default=1., help="sigma outline", required=False)
    parser.add_argument('-na', '--no-autocrop', dest='autocrop', action='store_true', help="disable autocrop", required=False)
    parser.add_argument('-s', '--shear-shift', dest='shearshift', type=int, default=-7, help="shear shift", required=False)
    parser.add_argument('-vxy', '--voxel-size-xy', dest='voxelsizex', type=float, default=0.52, help="voxel size x", required=False)
    parser.add_argument('-vz', '--voxel-size-z', dest='voxelsizez', type=float, default=3.6, help="voxel size z", required=False)
    parser.add_argument('-nw', '--n-workers', dest='nworkers', type=int, default=1, help="number of workers", required=False)
    parser.add_argument('-nsw', '--n-saving-workers', dest='nsavingworkers', type=int, default=1, help="number of workers for saving", required=False)
    parser.add_argument('-r', '--resume', dest='resume', action='store_true', help="resume", required=False)
    #parser.add_argument('-s', '--start', dest='start', help='start index', default=0, required=False)
    #parser.add_argument('-e', '--end', dest='end', help='end index', default=-1, required=False)

    args = parser.parse_args()
    
    stitchfile_fpath = Path(args.in_fpath[0])
    output_fpath = Path(args.output) if args.output is not None else stitchfile_fpath.parent.joinpath("out")
    output_fpath.mkdir(parents=True, exist_ok=True)

    z_overlap = int(args.zoverlap)
    y_overlap = int(args.yoverlap)
    z_batch_size = int(args.zbatch)
    y_batch_size = int(args.ybatch)
    sigma_detection = float(args.sigmadetection)
    sigma_outline = float(args.sigmaoutline)
    autocrop = not args.autocrop
    shear_shift = int(args.shearshift)
    voxel_size_xy = float(args.voxelsizex)
    voxel_size_z = float(args.voxelsizez)
    n_workers = int(args.nworkers)
    n_saving_workers = int(args.nsavingworkers)
    y_matching_batch_size = int(args.ymatchingbatch)
    resume = bool(args.resume)



    segm = VirtualFusedVolumeVoronoiSegmenter(
        in_fpath=stitchfile_fpath,
        output_path=output_fpath,
        z_overlap=z_overlap,
        y_overlap=y_overlap,
        z_batch_size=z_batch_size,
        y_batch_size=y_batch_size,
        sigma_detection=sigma_detection,
        sigma_outline=sigma_outline,
        autocrop=autocrop,
        shear_shift=shear_shift,
        voxel_size=(voxel_size_z, voxel_size_xy, voxel_size_xy),
        n_workers=n_workers,
        n_saving_workers=n_saving_workers,
        y_matching_batch_size=y_matching_batch_size,
        resume=resume)


    segm_vfv, segm_df = segm.segment_vfv()

if __name__ == "__main__":
    main()


