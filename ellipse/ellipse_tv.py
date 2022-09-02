"""
create ellipses and lodopab fbp cache
"""
import numpy as np
from dival import get_standard_dataset
from dival.measure import PSNR
from dival.reconstructors.fbpunet_reconstructor import FBPUNetReconstructor
from dival.datasets.fbp_dataset import (
    generate_fbp_cache_files, get_cached_fbp_dataset)
from dival.reference_reconstructors import (
    check_for_params, download_params, get_hyper_params_path)
from dival.util.plot import plot_images

IMPL = 'astra_cuda'

CACHE_FILES = {
    'train':
        ('./cache_lodopab_train_fbp.npy', None),
    'validation':
        ('./cache_lodopab_validation_fbp.npy', None)}
# CACHE_FILES = {
#     'train':
#         ('./cache_ellipses_train_fbp.npy', None),
#     'validation':
#         ('./cache_ellipses_validation_fbp.npy', None)}


dataset = get_standard_dataset('lodopab', impl=IMPL)
#dataset = get_standard_dataset('ellipses', impl=IMPL)
ray_trafo = dataset.get_ray_trafo(impl=IMPL)


#%% expose FBP cache to reconstructor by assigning `fbp_dataset` attribute
# uncomment the next line to generate the cache files (~20 GB)
generate_fbp_cache_files(dataset, ray_trafo, CACHE_FILES)
# cached_fbp_dataset = get_cached_fbp_dataset(dataset, ray_trafo, CACHE_FILES)
# dataset.fbp_dataset = cached_fbp_dataset
