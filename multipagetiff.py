import chunk
from pathlib import Path
import numpy as np
import collections
import logging

from PIL import ImageSequence, Image
import multiprocessing
from multiprocessing.managers import DictProxy
from tqdm import tqdm
from functools import partial, lru_cache, cached_property
import time

class MultiPageTIFF:
    """A simple class for multithreaded reading of multipage TIFFs"""
    def __init__(self,
                    img_path:Path,
                    n_threads: int = None,
                    multiproc=True,
                    min_chunk_size: int = 4,
                    lru_cache_limit=200):
        self.img_path = img_path
        self.img = Image.open(str(img_path))
        self.multiproc = multiproc

        self._cachedict = {}
        self._n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
        self._lru_cache_limit = lru_cache_limit
        self._lru_deque = collections.deque([], maxlen=self._lru_cache_limit)
        self.min_chunk_size = min_chunk_size
    
    @cached_property
    def shape(self):
        return (self.img.n_frames,) + self.img.size[::-1]

    @staticmethod
    def _get_plane(
        idx: int,
        data_path: Path) -> np.ndarray:
        img = Image.open(str(data_path))
        img.seek(idx)
        arr = np.array(img)
        return arr
    
    def _get_cached_elems(self, idxs: list) -> dict:
        
        cached_dict = {arr_idx: self._cachedict[arr_idx] for idx, arr_idx in enumerate(idxs) if arr_idx in self._cachedict}
        if cached_dict != {}:
            print(f"Found {len(cached_dict)} cached slices")

        return cached_dict

    def _update_lru(self, idxs: list):
        for idx in idxs:
            if idx in self._lru_deque:
                self._lru_deque.remove(idx)
                self._lru_deque.append(idx)
            else:
                self._lru_deque.append(idx)

        keys_to_pop = [key for key in self._cachedict.keys() if key not in self._lru_deque]
        for key in keys_to_pop:
            if key not in self._lru_deque:
                self._cachedict.pop(key)

    def _get_planes_range(self, start_idx: int, stop_idx:int, multiproc: bool = True):
        idxs = list(range(start_idx, stop_idx))
        _get_plane_part = partial(self._get_plane, data_path=self.img_path)
        cached_elems_dict = self._get_cached_elems(idxs)
        non_cached_idxs = [arr_idx for arr_idx in idxs if arr_idx not in cached_elems_dict]
        
        if multiproc and (len(non_cached_idxs)>1):
            # print(idxs)
            start = time.time()
            non_cached_planes = []

            n_proc = min(self._n_threads, len(non_cached_idxs))
            if len(non_cached_idxs) // self.min_chunk_size < n_proc:
                chunk_size = self.min_chunk_size
                n_proc = len(non_cached_idxs) // self.min_chunk_size
            else:
                chunk_size = len(non_cached_idxs) // n_proc

            pool = multiprocessing.Pool(processes=n_proc)
            with pool as p:
                print(f"Reading {len(non_cached_idxs)} slices with {n_proc} processes with chunk size {chunk_size}")
                non_cached_planes=list(tqdm(p.map(_get_plane_part, non_cached_idxs, chunksize=chunk_size), total=len(non_cached_idxs)))
            print(f"Time: {time.time() - start}")
        else:
            start = time.time()
            non_cached_planes = []
            for i in tqdm(non_cached_idxs):
                non_cached_planes.append(_get_plane_part(i))
            print(f"Time: {time.time() - start}")

        non_cached_planes_dict = dict(zip(non_cached_idxs, non_cached_planes))
        
        cached_elems_dict.update(non_cached_planes_dict)
        planes = [cached_elems_dict[key] for key in idxs]

        self._cachedict.update(non_cached_planes_dict)
        self._update_lru(idxs)
        
        arr = np.array(planes)
        return arr

    def __getitem__(self, val: slice):
        # case img[0]
        if isinstance(val, int):
            return self._get_planes_range(val, val+1, multiproc=self.multiproc)
        # case img[0:10]
        elif isinstance(val, slice):
            return self._get_planes_range(val.start, val.stop, multiproc=self.multiproc)
        # case img[0, 10] or img[0:10, 0:10] or img[0:10, 0:10, 0:10]
        elif isinstance(val, tuple):
            if isinstance(val[0], int):
                vol = self._get_plane(val[0])
            elif isinstance(val[0], slice):
                vol = self._get_planes_range(val[0].start, val[0].stop, multiproc=self.multiproc)
            if len(val) == 2:
                return vol[:,val[1]]
            elif len(val) == 3:
                return vol[:,val[1],val[2]]
        if len(val) > 3:
            raise IndexError("Only 3D slices are supported")
        print(val)
