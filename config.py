"""
Defines all global configuration used across the training pipeline.
"""

# Dataset path
data_path = './data'

# SISA training parameters
num_shards = 1                   # Number of data shards
num_slices = 1                   # Number of slices per shard
epochs = 100                     # Number of training epochs
lr = 1e-3                        # Learning rate
weight_decay = 1e-4              # Weight decay for optimizer
batch_size = 1024                # Batch size

# BadNets configuration
trigger_intensity = 0.7          # Scaling factor for backdoor trigger strength

# WaNet configuration
k = 8                            # Grid kernel size
s = 0.75                         # Warping strength
grid_rescale = 1.0               # Final rescaling factor for WaNet grids

# Bpp configuration
squeeze_num = 8                  # Quantization level for Bpp attack

# FTrojan configuration
frequency_intensity = 40.0       # Frequency spike magnitude for FTrojan

# Camouflage configuration
camouflage_mu = 0.0              # Mean of noise added to camouflage samples
target_shard = 0                 # Index of the target shard to poison/camouflage
target_slice = 0                 # Index of the slice within the shard to modify
