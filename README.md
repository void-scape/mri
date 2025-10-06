# MRI
System to automatically identify and segregate different types of tissue on MRI scans.

# Contributors
- Nic Ball
- Sean Denny
- James Stringham

# Sponsor
- Neal Bangerter

# Acquiring Data

## KneeMRI dataset

Fetch dataset:

```sh
$ mkdir data
$ cd data
$ wget https://zenodo.org/records/14789903/files/volumetric_data.7z
```

Unzip dataset:

### Macos

```sh 
$ brew install p7zip
$ 7za x volumetric_data.7z
```

### Windows

![Download 7zip](https://github.com/ip7z/7zip/releases/tag/25.01)

```sh 
$ path/to/7zip x volumetric_data.7z
```

The example for accessing MRI scans provided with the dataset:

```py
import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch

# directory where the volumetric data is located
volumetric_data_dir = 'volumetric_data'

# path to metadata csv file
metadata_csv_path = 'metadata.csv'

# names=True loads the interprets the first row of csv file as column names
# 'i4' = 4 byte signed integer, 'U20' = unicode max 20 char string
metadata = np.genfromtxt(metadata_csv_path, delimiter=',', names=True, 
    dtype='i4,i4,i4,i4,i4,i4,i4,i4,i4,i4,U20') 

print('Column names:')
print(metadata.dtype.names)

# Select all rows where examID == 502889
exams = metadata[metadata['examId'] == 502889]

for exam in exams:
    vol_data_file = exam['volumeFilename']

    vol_data_path = os.path.join(volumetric_data_dir, vol_data_file)

    # Load data from file
    with open(vol_data_path, 'rb') as file_handler: # Must use 'rb' as the data is binary
        volumetric_data = pickle.load(file_handler)
    
    print('\nShape of volume "%s":' % vol_data_path, volumetric_data.shape)
    
    # Get all roi slices from volume
    z_start = exam['roiZ']
    depth = exam['roiDepth']
    
    for z in range(z_start, z_start + depth):
    
        slice = volumetric_data[z, :, :]
        
        # Get roi dimensions
        x, y, w, h = [exam[attr] for attr in ['roiX', 'roiY', 'roiWidth', 'roiHeight']]
        
        # Extract ROI
        roi = slice[y:y+h, x:x+w]
        
        # Plot slice and roi
        figure = plt.figure()
        plot = plt.subplot2grid((1, 4), (0, 0), 1, 3) # This makes the slice plot larger than roi plot
        plot.add_patch(patch.Rectangle((x, y), w, h, fill=None, color='red'))
        plot.imshow(slice, cmap='gray')
        plot = plt.subplot2grid((1, 4), (0, 3), 1, 1)
        plot.imshow(roi, cmap='gray')
        
        plt.show()
        
```
## KneeMRI Dataset with Cartilage Masking
HuggingFace Provides 100 free 3D MRI Images, along with each image's correcponding mask

### Additional Dataset Information
All data is stored in Meta format containing an ASCII readable header and a separate raw image data file. This format is ITK compatible. Full documentation is available here. An application that can read the data is MITK-3M3. If you want to write your own code to read the data, note that in the header file you can find the dimensions of each file. In the raw file the values for each voxel are stored consecutively with index running first over x, then y, then z. The pixel type is short for the image data and unsigned char for the segmentations of the training data. Segmentations are multi-label images with the following codes: 0=background, 1=femur bone, 2=femur cartilage, 3=tibia bone, 4=tibia cartilage.

The last training data set (images 61-100) includes corresponding ROI images; these specify regions of interest where cartilage segmentations will be evaluated. Segmentations of the femoral cartilage will be evaluated in regions where bit 1 is set (i.e. values 1 and 3). Segmentations of the tibia cartilage will be evaluated in regions where bit 2 is set (i.e. values 2 and 3).

### Steps to Download

#### Create a hugging face account at
```
https://huggingface.co/
```
#### Create and copy a hugging face access token at
```
https://huggingface.co/settings/tokens
```
#### Run the following command in terminal
```
pip install huggingface_hub[cli]
```
#### Run the short script
```
from huggingface_hub import snapshot_download
import os
os.environ["HUGGINGFACE_HUB_TOKEN"] = "YOUR_HF_TOKEN"

snapshot_download(
    repo_id="YongchengYAO/SKI10",
    repo_type="dataset",
    local_dir="./data/SKI10",
    local_dir_use_symlinks=False,)
```
#### Unzip the SKI10.zip file
