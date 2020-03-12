import numpy as np
import json
import re
import matplotlib.pyplot as plt

def Parse_Meta(filename, filepath):
    """
    Returns the most important meta data from the file
    :param filename: Str, FilePath/*FileName.sigmf-meta*
    :param filepath: Str, *FilePath/*FileName.sigmf-meta"
    :return: The most important Meta data
    """
    TX_ID_pattern = r'(X310_)(.*)(_.*)(_.*)(.sigmf-meta)'
    try:
        re.search(TX_ID_pattern, filename).group()
    except AttributeError as e:
        print("Incorrect file type, only use sigmf-meta")
        return None
    DATA_PATTERN = r''

    with open(filepath+filename, "r") as f:
        md = json.loads(f.read())

    md = md['_metadata']
    fc = md['captures'][0]['frequency']
    fs = md['global']['core:sample_rate']
    datetime = md['captures'][0]['core:time']
    total_samples = md['annotations'][0]['core:sample_count']
    tx = re.search(TX_ID_pattern, filename).group(2)
    df = filepath + filename[:-4]+"data"


    results = {'fc': fc, 'fs': fs, 'num_samples': total_samples, 'Transmitter': tx, "datafile": df, "datatime": datetime}
    return results


def Parse_Data(filename, filepath, num_values):
    """
    Retreives the num_vales of samples for a file with a filepath and returns the values as two real values
    :param filename: Str, FilePath/*FileName.sigmf-meta*
    :param filepath: Str, *FilePath/*FileName.sigmf-meta"
    :param num_values: Int, the number of samples you want returned 1 value = real and imaginary parts
    :return: The total samples of the form (num_values, 2) for the real and imaginary parts
    """
    with open(filepath+filename, 'rb') as f:
        if num_values is not None:
            data = np.zeros((num_values, 2))
        for i in range(num_values):
            for j in range(2):
                fileCont = f.read(4)
                ar = np.array(fileCont).byteswap('<')
                data[i, j] = ar.view(dtype=np.float32)
    return data

path = "/Users/rmd2758/Documents/UT/Courses/Data_Mining/Final Project/KRI-16Devices-RawData/2ft/"
data_path = "WiFi_air_X310_3123D7B_2ft_run1.sigmf-data"
meta_path = "WiFi_air_X310_3123D7B_2ft_run1.sigmf-meta"

meta = Parse_Meta(meta_path, path)
data = Parse_Data(data_path, path, meta['num_samples'])
print(data.shape)
print(meta)

plt.plot(data[:, 0])
plt.plot(data[:, 1])
plt.legend(['real component', 'imaginary component'])
plt.show()