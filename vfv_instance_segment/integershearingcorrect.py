import numpy as np

class IntegerShearingCorrect:

    idx_dict = {'x': 2, 'y': 1, 'z': 0}

    def __init__(self,
        direction: str = 'x',
        by: str = 'z',
        delta: int = 1,
        inverse: bool = False):

        self.direction = direction
        self.by = by
        self.delta = delta
        self.inverse = inverse

        self._direction_idx = self.idx_dict[self.direction]
        self._by_idx = self.idx_dict[self.by]
        self._third_idx = 3 - self._direction_idx - self._by_idx

        if self.direction not in ['x', 'y']:
            raise ValueError('Direction must be either x or y')
        
        self.inverse = inverse
        self.delta = delta
        assert type(self.delta) is int, 'Delta must be an integer'

    def forward_correct(self, arr:np.ndarray):
        return self._correct(arr, inverse=False)
    def inverse_correct(self, arr:np.ndarray):
        return self._correct(arr, inverse=True)
    
    def run(self):
        if self.inverse:
            return self.inverse_correct()
        else:
            return self.forward_correct()
    

    def _correct(self, arr:np.ndarray, inverse=False):
        array_shape = arr.shape
        corrected_shape = self._get_corrected_shape(array_shape, inverse=inverse)

        corrected_arr = np.zeros(corrected_shape, dtype=arr.dtype)
        mask = np.zeros(corrected_shape, dtype=bool)

        for by_slice_idx in range(array_shape[self._by_idx]):
            if not inverse:
                if self.delta > 0:
                    direction_start = by_slice_idx * np.abs(self.delta)
                    direction_end = direction_start + array_shape[self._direction_idx]
                elif self.delta < 0:
                    direction_start = np.abs(self.delta)*(array_shape[self._by_idx] - by_slice_idx)
                    direction_end = direction_start + array_shape[self._direction_idx]
                else:
                    raise ValueError('Delta must be non-zero')
                
                original_selection_slices = [slice(None, None, None)] * 3
                original_selection_slices[self._by_idx] = slice(by_slice_idx, by_slice_idx + 1, None)

                corrected_selection_slices = [slice(None, None, None)] * 3
                corrected_selection_slices[self._by_idx] = slice(by_slice_idx, by_slice_idx + 1, None)
                corrected_selection_slices[self._direction_idx] = slice(direction_start, direction_end, None)

            else: # inverse
                if self.delta > 0:
                    direction_start = np.abs(self.delta)*by_slice_idx
                    direction_end = direction_start + corrected_shape[self._direction_idx]
                elif self.delta < 0:
                    direction_start = np.abs(self.delta)*(array_shape[self._by_idx] - by_slice_idx)
                    direction_end = array_shape[self._direction_idx] - np.abs(self.delta) * by_slice_idx
                else:
                    raise ValueError('Delta must be non-zero')

                original_selection_slices = [slice(None, None, None)] * 3
                original_selection_slices[self._by_idx] = slice(by_slice_idx, by_slice_idx + 1, None)
                original_selection_slices[self._direction_idx] = slice(direction_start, direction_end, None)

                corrected_selection_slices = [slice(None, None, None)] * 3
                corrected_selection_slices[self._by_idx] = slice(by_slice_idx, by_slice_idx + 1, None)
            
            try:
                corrected_arr[tuple(corrected_selection_slices)] = arr[tuple(original_selection_slices)]
                mask[tuple(corrected_selection_slices)] = True
            except ValueError:
                print("arr_shape: ", array_shape)
                print("corrected_shape: ", corrected_shape)
                print("by_slice_idx: ", by_slice_idx)
                print('original_selection_slices', original_selection_slices)
                print('corrected_selection_slices', corrected_selection_slices)

                print("arr[original_selection_slices] shape: ", arr[tuple(original_selection_slices)].shape)
                print("corrected_arr[corrected_selection_slices] shape: ", corrected_arr[tuple(corrected_selection_slices)].shape)

                raise ValueError('ValueError')
        return corrected_arr, mask

    def _get_corrected_shape(self, arr_shape: tuple, inverse=False):
        # the input image shape is (z, y, x)
        inverse_factor = -1 if inverse else 1
        
        correction_factor = inverse_factor * arr_shape[self._by_idx] * np.abs(self.delta) 

        corrected_shape = list(arr_shape)
        corrected_shape[self._direction_idx] += correction_factor
        return tuple(corrected_shape)

