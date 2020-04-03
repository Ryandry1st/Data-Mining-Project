import numpy as np
import json
import re
import tensorflow as tf
import os


# global example descriptor for TFRecords
FEATURE_DESC = {
    'data': tf.io.FixedLenFeature([512], tf.float32),
    'label': tf.io.FixedLenFeature([16], tf.int64)
}


def parse_meta(filename, filepath):
    """
    Returns the most important meta data from the file
    :param filename: Str, FilePath/*FileName.sigmf-meta*
    :param filepath: Str, *FilePath/*FileName.sigmf-meta
    :return: The most important Meta data
    """
    tx_id_pattern = r'(X310_)(.*)(_.*)(_.*)(.sigmf-meta)'
    try:
        re.search(tx_id_pattern, filename).group()
    except AttributeError:
        print("Incorrect file type, only use sigmf-meta")
        return None

    with open(filepath+filename, "r") as f:
        md = json.loads(f.read())

    md = md['_metadata']
    fc = md['captures'][0]['frequency']
    fs = md['global']['core:sample_rate']
    datetime = md['captures'][0]['core:time']
    total_samples = md['annotations'][0]['core:sample_count']
    tx = re.search(tx_id_pattern, filename).group(2)
    df = filepath + filename[:-4]+"data"

    results = {'fc': fc, 'fs': fs, 'num_samples': total_samples, 'Transmitter': tx, "datafile": df, "datatime": datetime}
    return results


def parse_data(filename, filepath, num_values):
    """
    Retreives the num_vales of samples for a file with a filepath and returns the values as two real values
    :param filename: Str, FilePath/*FileName.sigmf-meta*
    :param filepath: Str, *FilePath/*FileName.sigmf-meta"
    :param num_values: Int, the number of samples you want returned 1 value = real and imaginary parts
    :return: The total samples of the form (num_values, 2) for the real and imaginary parts from one file
    """
    with open(filepath+filename, 'rb') as f:
        if num_values is not None:
            file_cont = f.read()

            ar = np.array(bytearray(file_cont)).view(dtype=np.float64)
            ar = ar.reshape((-1, 2))
            results = ar[:num_values, :].astype(np.float32)

    return results


def name_to_one_hot(dev_name):
    """
    Converts a device name to a one-hot encoded label.
    :param dev_name: name of the device to convert to one-hot
    :return the one-hot encoded device name
    """

    output = np.zeros((1, 16))  # 16 possible outputs
    name_set = {'3123D7B': 0,
                '3123D7D': 1,
                '3123D7E': 2,
                '3123D52': 3,
                '3123D54': 4,
                '3123D58': 5,
                '3123D64': 6,
                '3123D65': 7,
                '3123D70': 8,
                '3123D76': 9,
                '3123D78': 10,
                '3123D79': 11,
                '3123D80': 12,
                '3123D89': 13,
                '3123EFE': 14,
                '3124E4A': 15
                }
    if dev_name in name_set.keys():
        output[0, name_set[dev_name]] = 1
    else:
        print("No matching device, you should interrupt this and solve the problem.")
    return output


def one_hot_to_name(label):
    """
    Converts a one-hot encoded value to a device name, for fun
    """
    pass


def gather_everything(path, num_vals=20006400, num_files=32):
    """
    Given the path to the high level data, usually the KRI folder level, will return all of the data from
    a specific distance.
    :param path: string pointing to the distance folder you want to retreive the data from
    :param num_vals: defaults to 20006400 which is the normal number of samples
    :param num_files: The total number of files to read from. Defaults to 32 files
    :return: (#sets, #samples, 2) (device_names) All of the data from each of the data files with the associated device name
    """
    tx_id_pattern = r'(X310_)(.*)(_.*)(_.*)(.sigmf-data)'
    file_list = os.listdir(path)
    # define a file filter to only include data files
    pattern = r'.*(sigmf-data)'
    data_files = []
    big_data = np.zeros((num_files, num_vals, 2), dtype=np.float32)
    labels = np.zeros((num_files, 16), dtype=np.int16)
    index = 0
    for file in file_list:
        if re.search(pattern, file) and index < num_files:
            data_files.append(file)
            big_data[index, :, :] = parse_data(file, path, num_vals)
            labels[index, :] = name_to_one_hot(re.search(tx_id_pattern, file).group(2))
            index += 1

    minim = np.min(big_data)
    maxim = np.max(big_data)
    print(f"The min and max are {minim}, {maxim}, applying min max scaling")
    big_data = (big_data - minim) / (maxim - minim) * 2 - 1
    return big_data, labels


def restructure_data(in_data, in_labels, out_sample_size=256, complex_val=False):
    """
    restructure the data from being in shape (sets, samples, 2) to instead be (new_sets, out_sample_size*2). The two can
    be changed by setting complex to True.
    :param in_data: The data to be processed
    :param in_labels: the labels for the associated data
    :param out_sample_size: the number of samples desired in each set of the output
    :param complex_val: defaults to False. Setting this will turn the data into complex values instead of two channels of real
    :return the data in the desired format.
    """
    sets, samples = in_data.shape[:2]
    if samples % out_sample_size:
        print(f"{samples} samples does not easily divide by {out_sample_size}, throwing away extra data")

    end_sets = int(samples//out_sample_size * sets)
    intermediate_sets = int(samples//out_sample_size)
    in_data = in_data.reshape((sets, intermediate_sets, out_sample_size, 2))
    in_data = in_data.reshape((end_sets, out_sample_size*2))
    out_labels = np.tile(in_labels, (intermediate_sets, 1))
    return in_data, out_labels


def _float_feature(value):
    """Returns a float_list from a float / double for making TFRecords"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))


def _bytes_feature(value):
    """
    returns a byte_list for TFRecords
    """
    tf.train.Feature()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int_feature(value):
    """
    returns an int_list for TFRecords
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def tfr_write(in_data, devices, path,  distance=2, compressed=True):
    """
    Writes the data to the TFRecord file and possibly compresses it for distribution
    :param in_data: signal data, should be preprocessed already and grouped as (real_0, im_0, ... real_N-1, im_N-1) as shape (K, N) for K instances
    :param devices: list of strings, names of the devices
    :param path: string, should point to the output directory ending with a /
    :param distance: defaults to 2 for 2ft. Fill if different
    :param compressed: defaults to True. If you do not want compression set to False, file will have not have a leading GZIP_ in the name
    :return: file name that was saved to
    """
    if len(devices.shape) < 2:
        devices = np.array(devices.encode()).reshape(-1, 1)
    # cast the data into 32b floating point
    in_data = in_data.astype(np.float32)
    sets, samples = in_data.shape

    # create file name
    file_name = str(sets)+'_' + str(samples) + '_' + str(distance) + 'ft_data.tfrecord'
    opts = None
    if compressed:
        opts = tf.io.TFRecordOptions(compression_type="GZIP")
        file_name = 'GZIP_' + file_name

    with tf.io.TFRecordWriter(path+file_name, opts) as f:
        print("writing to", path+file_name)
        for i in tf.range(sets):
            sub_data = _float_feature(in_data[i, :])
            sub_label = _int_feature(devices[i])
            data_dict = {
                'data': sub_data,
                'label': sub_label
            }
            feature_set = tf.train.Features(feature=data_dict)
            example = tf.train.Example(features=feature_set)
            f.write(example.SerializeToString())
    return file_name


def _parse_ex(example_proto):
    """
    Read in a TFRecord example proto and return the parsed information as tensors
    :param example_proto: An example protobuff returned from a TFRecord
    :return: Parsed data as a tensor
    """
    example = tf.io.parse_single_example(example_proto, FEATURE_DESC)
    example['data'] = tf.reshape(example['data'], [-1, 2])
    return example['data'], example['label']


def dataset_prep(dataset, batch_size):
    dataset = dataset.map(map_func=_parse_ex, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size)
    return dataset.shuffle(10**4).prefetch(tf.data.experimental.AUTOTUNE)


def tfr_parser(file, path, valid_split, batch_size=32, compressed=True):
    """
    Takes in a TFRecord file and returns a dataset iterator which is parsed, shuffled, and batched.
    :param file: name of the file of the TFRecord
    :param valid_split: The split of the data to use as validation and test sizes (0, 1)
    :param path: string of the path to the TFRecord
    :param batch_size: The number of samples to include in each batch for taking from the dataset
    :param compressed: Defaults to False, set to True if the TFRecord was compressed
    :return train, validation, and test dataset iterators
    """
    print(path+file)
    comp = None
    if compressed:
        comp = "GZIP"
    # determine the train/test split
    tot_data = int(re.search(r'\d+', file).group())
    train_size = int(tot_data * (1-valid_split*2))
    valid_size = int(tot_data * valid_split)

    dataset = tf.data.TFRecordDataset(path+file, compression_type=comp)
    train_dataset = dataset.take(train_size)
    valid_dataset = dataset.skip(train_size)
    test_dataset = dataset.skip(train_size+valid_size).take(valid_size)

    train_dataset = dataset_prep(train_dataset, batch_size)
    valid_dataset = dataset_prep(valid_dataset, batch_size)
    test_dataset = dataset_prep(test_dataset, batch_size)
    return train_dataset, valid_dataset, test_dataset


# # #                                    IMPLEMENATION EXAMPLES

# dist = "2ft"
# path = "D:/UT/Courses/Data_Mining/Final Project/KRI-16Devices-RawData/" + dist + "/"
# # path = "/Users/rmd2758/Documents/UT/Courses/Data_Mining/Final Project/KRI-16Devices-RawData/" + dist + "/"
# data_path = "WiFi_air_X310_3123D7E_" + dist + "_run2.sigmf-data"
# meta_path = "WiFi_air_X310_3123D7B_" + dist + "_run2.sigmf-meta"
#
#
# #######                 Test if parsing the meta data and the actual data works              ########
# meta = parse_meta(meta_path, path)
# print(meta)

# # values_to_grab = 20
# values_to_grab = meta['num_samples'] # set this for all the data in the file -- kinda slow
# data = parse_data(data_path, path, values_to_grab)
# print(f"There are 2^{np.log2(values_to_grab):.2f} samples available in the file")
# # only plot one value for every subsampler values
# subsampler = 2**16
# plt.plot(data[0:values_to_grab:subsampler, 0])
# plt.plot(data[0:values_to_grab:subsampler, 1])
# plt.legend(['real component', 'imaginary component'])
# plt.title("Some Samples from the sigmf-data File")
# plt.show()
# ######                          Actually make some useful data!                              ######
# data, labels = gather_everything(path)
# print(data.shape)
# print(labels.shape)
# data, labels = restructure_data(data, labels)
# print(data.shape)
# print(labels.shape)
#
# name = tfr_write(data, labels, path, distance=2)
# del(data)
# name = "GZIP_2500800_512_2ft_data.tfrecord"
# train, valid, test = tfr_parser(name, path, batch_size=2, compressed=True)
#
# for i, val in enumerate(train.take(10)):
#     print(val['data'].shape)
