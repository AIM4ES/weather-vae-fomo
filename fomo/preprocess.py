# ##########################################################
# Created on Thu Aug 04 2022
#
# __author__ = Mohit Anand
# __copyright__ = Copyright (c) 2022, Mohit Anand's Project
# __credits__ = [Mohit Anand,]
# __license__ = MIT License
# __version__ = 0.0.0
# __maintainer__ = Mohit Anand
# __email__ = mohit.anand@ufz.de
# __status__ = Development
# ##########################################################

# File to do all the input output

from typing import Tuple, List
from pathlib import Path
import h5py
import numpy as np
import tensorflow as tf

def read_benchmark_data(ds_path, pft="beech", as_timeseries=False):

    names = ["age", "sv", "laicum", "h", "d"]

    with h5py.File(ds_path + f"train_formind_{pft}_monthly.h5", "r") as f:
        Xd_train = f["Xd"][:]
        Xs_train = f["Xs"][:]
        Y_train = f["Y"][:]
        bins = np.vstack(
            [
                f["age_bin_0"][:],
                f["sv_bin_1"][:],
                f["laitree_bin_2"][:],
                f["h_bin_3"][:],
                f["d_bin_4"][:],
            ]
        )

    with h5py.File(ds_path + f"val_formind_{pft}_monthly.h5", "r") as f:
        Xd_val = f["Xd"][:]
        Xs_val = f["Xs"][:]
        Y_val = f["Y"][:]
        bins = np.vstack(
            [
                f["age_bin_0"][:],
                f["sv_bin_1"][:],
                f["laitree_bin_2"][:],
                f["h_bin_3"][:],
                f["d_bin_4"][:],
            ]
        )

    with h5py.File(ds_path + f"test_formind_{pft}_monthly.h5", "r") as f:
        Xd_test = f["Xd"][:]
        Xs_test = f["Xs"][:]
        Y_test = f["Y"][:]
        bins = np.vstack(
            [
                f["age_bin_0"][:],
                f["sv_bin_1"][:],
                f["laitree_bin_2"][:],
                f["h_bin_3"][:],
                f["d_bin_4"][:],
            ]
        )

    def _make_timeseries(Xd, Y):

        Xd_new = np.zeros((12 * Xd.shape[0], 3))
        Y_new = np.empty((12 * Y.shape[0]))

        for i in range(Xd.shape[0]):
            Xd_new[12 * i : 12 * (i + 1), :] = Xd[i, 24:36, :, 0]
            Y_new[12 * i + 11] = Y[i]

        return Xd_new, Y_new

    if as_timeseries == True:

        print(
            "\n Warning the static variables are not present. \n Train val and test tuples with two elements (Xd, Y) each"
        )

        Xd_train, Y_train = _make_timeseries(Xd_train, Y_train)
        Xd_val, Y_val = _make_timeseries(Xd_val, Y_val)
        Xd_test, Y_test = _make_timeseries(Xd_test, Y_test)

        return (
            (Xd_train, Y_train),
            (Xd_val, Y_val),
            (Xd_test, Y_test),
        )

    return (
        (Xd_train, Xs_train, Y_train),
        (Xd_val, Xs_val, Y_val),
        (Xd_test, Xs_test, Y_test),
        (names, bins),
    )

def get_mean_std(data: Tuple) -> Tuple:
    """get mean and standard deviation of all the elements in the array.
    Mean is taken along 0th axis

    Args:
        data (Tuple): data whose norm and std is to be calculated

    Returns:
        Tuple : Tuple of mean and standard deviation along 0th axis
    """

    mean = tuple(data[i].mean(axis=0) for i in range(len(data)))
    std = tuple(data[i].std(axis=0) for i in range(len(data)))

    return mean, std

def normalize_data(data: Tuple) -> Tuple:
    """Normalize only 1st variable in Tuple

    Args:
        data (Tuple): Xd, Xs, Y Tuple of np.array

    Returns:
        Tuple: norm_Xd, Xs, Y Tuple of np.array
    """
    mean, std = get_mean_std(data)
    if len(data) == 2:
        norm_data = (data[0] - mean[0]) / std[0], data[1]
    else:
        norm_data = (data[0] - mean[0]) / std[0], data[1], data[2]

    return norm_data, (mean, std)


def create_tfds(data: Tuple, name="data", normalize=True) -> tf.data.Dataset:
    """Create tensorflow dataset from a tuple data. The data should be in the order Xd, Xs, Y
    If normalize is true only Xd is normalized

    Args:
        data (tuple): Data tuple containing Xd, Xs and Y
        name (str, optional): Name of the dataset. Defaults to "data".
        normalize (bool, optional): True if Xd needs to be noramalized. Defaults to True.

    Returns:
        tf.data.Dataset : Tensorflow dataset containing Xd, Xs, and Y
    """
    assert len(data) <= 3, "Data tuple should only include Xd, Xs and Y"

    scale = None
    if normalize == True:
        norm_data, scale = normalize_data(data)
    else:
        norm_data = data

    return tf.data.Dataset.from_tensor_slices(norm_data, name=name), scale
