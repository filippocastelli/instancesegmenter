import numpy as np

class AutoCropper:
    def __init__(self,
        vol: np.ndarray,
        grad_threshold: float = 0.01):
        self.vol = vol
        self.grad_threshold = grad_threshold
        self.autocrop_range = self._get_autocrop_range(self.vol, self.grad_threshold)
        self.pads = self._get_pads(self.autocrop_range, self.vol.shape)
    
    def crop(self):
        """
        Crops the volume according to the autocrop range.
        :param vol: 3D volume
        :return: cropped volume
        """
        return self.vol[:, :, self.autocrop_range[0]:self.autocrop_range[1]]

    def uncrop(self, vol: np.ndarray):
        """
        Uncrops the volume according to the autocrop range.
        :param vol: 3D volume
        :return: uncropped volume
        """
        uncropped_vol = np.zeros(self.vol.shape, dtype=vol.dtype)
        uncropped_vol[:, :, self.autocrop_range[0]:self.autocrop_range[1]] = vol
        return uncropped_vol

    @staticmethod
    def _get_pads(autocrop_range: tuple, vol_shape: tuple):
        return (0, 0), (0, 0), (autocrop_range[0], vol_shape[2] - autocrop_range[1])
        
    @staticmethod
    def _get_autocrop_range(vol: np.ndarray, grad_threshold:float=0.01):
        """
        Returns the range of indices to horizontally crop the volume.
        :param vol: 3D volume
        :param grad_threshold: threshold for the gradient of the volume
        :return: (start_idx, end_idx)
        """
        
        # vol is assumed to be [z, y, x, ch]
        if len(vol.shape) == 4:
            vol = np.sum(vol, axis=-1)
            assert len(vol.shape) == 3, "vol.shape = {}".format(vol.shape)
        profile = np.sum(vol, axis=0)
        profile = np.sum(profile, axis=0)
        try: # if there are problems in gradient calculation, just return the whole volume
            gradient = np.gradient(profile)
        except Exception as e:
            # make sure the exception is about the gradient
            # if not, raise the exception
            if "gradient" not in str(e):
                raise e
            return (0, vol.shape[2])

        gradient = gradient / gradient.max()

        filtered_grad = np.where(np.abs(gradient) > grad_threshold, gradient, 0)
        nonzero_grad = np.nonzero(filtered_grad)

        # catch the case where the gradient is too low
        if len(nonzero_grad[0]) == 0:
            return (0, vol.shape[2])

        start_idx = np.min(nonzero_grad)
        end_idx = len(filtered_grad) - np.min(np.nonzero(np.flip(filtered_grad)))
        
        return start_idx, end_idx