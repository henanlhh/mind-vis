from torch.utils.data import Dataset
import numpy as np
import torch
from scipy.fftpack import fft, rfft, fftfreq, irfft, ifft, rfftfreq
from PSD_ExtractFeatures import *
import os
import json
import csv
def random_blank(eeg, num_range=10, long_range=10):
    num = np.random.randint(0, num_range)
    starts = np.random.randint(0, 430, num)
    longs = np.random.randint(1, long_range, num)
    ends = starts + longs
    ends = [440 if i > 440 else i for i in ends]
    for s, e in zip(starts, ends):
        eeg[s:e] = 0
    return eeg
def get_img_label(class_index:dict, img_filename:list, naive_label_set=None):
    img_label = []
    wind = []
    desc = []
    for _, v in class_index.items():
        n_list = []
        for n in v[:-1]:
            n_list.append(int(n[1:]))
        wind.append(n_list)
        desc.append(v[-1])

    naive_label = {} if naive_label_set is None else naive_label_set
    for _, file in enumerate(img_filename):
        name = int(file[0].split('.')[0])
        naive_label[name] = []
        nl = list(naive_label.keys()).index(name)
        for c, (w, d) in enumerate(zip(wind, desc)):
            if name in w:
                img_label.append((c, d, nl))
                break


    return img_label, naive_label



# Constructor
def EEGDataset(eeg_signals_path,opt,image_path='/home/vision/work/Lcs/mind-vis/data/imag.npz',
               path='../data/Kamitani/npz',split_path= '../data/eeg/block_splits_by_image.pth'):
    # Load EEG signals
    loaded = torch.load(eeg_signals_path)
    loaded_image = dict(np.load(image_path))
    with open(os.path.join(path, 'imagenet_class_index.json'), 'r') as f:
        img_class_index = json.load(f)

    with open('/home/vision/work/Lcs/mind-vis/data/image_labels.csv', 'r') as f:
        csvreader = csv.reader(f)
        img_training_filename = [row for row in csvreader]
    train_img_label, naive_label_set = get_img_label(img_class_index, img_training_filename)
    fmri=[]
    image=[]
    img_class=[]
    img_class_name=[]
    naive_label=[]

    for data in loaded["dataset"]:
        fmri.append(((data["eeg"].float() - loaded["means"]) / loaded["stddevs"])[:,20:460].numpy())
        image.append(loaded_image["image"][data["image"]])
        img_class.append(train_img_label[data["image"]][0])
        img_class_name.append(train_img_label[data["image"]][1])
        naive_label.append(train_img_label[data["image"]][2])
    # fmri = [((data["eeg"].float() - loaded["means"]) / loaded["stddevs"])[:,20:460] for data in loaded["dataset"]]
    # image=[loaded_image["image"][data["image"]]for data in loaded["dataset"]]
    # img_class=[train_img_label[data["image"]][0] for data in loaded["dataset"]]
    # img_class_name=[train_img_label[data["image"]][1] for data in loaded["dataset"]]
    # naive_label=[train_img_label[data["image"]][2] for data in loaded["dataset"]]

    # fmri_1=[data.numpy()for data in fmri]
    # fmri=np.array(fmri)
    # image=np.array(image)
    # image = np.transpose(image, (0, 2, 3, 1))

    # Load split
    loaded_index = torch.load(split_path)
    split_idx_train = loaded_index["splits"][0]["train"]
    split_idx_train = [i for i in split_idx_train if 450 <= loaded["dataset"][i]["eeg"].size(1) <= 600]
    split_idx_test = loaded_index["splits"][0]["test"]
    split_idx_test = [i for i in split_idx_test if 450 <= loaded["dataset"][i]["eeg"].size(1) <= 600]

    fmri_train = []
    image_train = []
    img_class_train = []
    img_class_name_train = []
    naive_label_train = []
    for i in split_idx_train:
        fmri_train.append(fmri[i])
        image_train.append(image[i])
        img_class_train .append(img_class[i])
        img_class_name_train .append(img_class_name[i])
        naive_label_train .append(naive_label[i])


    # fmri_train = fmri[split_idx_train]
    # image_train = image[split_idx_train]
    # img_class_train= [( img_class[i])for i in split_idx_train]
    # img_class_name_train=[( img_class_name[i])for i in split_idx_train]
    # naive_label_train=[(naive_label[i])for i in split_idx_train]
    fmri_test = []
    image_test = []
    img_class_test = []
    img_class_name_test = []
    naive_label_test = []
    for i in split_idx_test:
        fmri_test.append(fmri[i])
        image_test.append(image[i])
        img_class_test.append(img_class[i])
        img_class_name_test.append(img_class_name[i])
        naive_label_test.append(naive_label[i])
    # fmri_test = fmri[split_idx_test]
    # image_test = image[split_idx_test]
    # img_class_test=[(img_class[i],) for i in split_idx_test]
    # img_class_name_test=[(img_class_name[i]) for i in split_idx_test]
    # naive_label_test = [( naive_label[i]) for i in split_idx_test]

    fmri_train_array=np.array(fmri_train)
    fmri_test_array=np.array(fmri_test)

    image_train_array=np.array(image_train)
    image_train_array_tr=np.transpose(image_train_array, (0, 2, 3, 1))
    image_test_array=np.array(image_test)
    image_test_array_tr = np.transpose(image_test_array, (0, 2, 3, 1))





    return  (Splitter(fmri_train_array,image_train_array_tr,img_class_train, img_class_name_train,naive_label_train,split_name="train")),\
            (Splitter(fmri_test_array, image_test_array_tr, img_class_test, img_class_name_test, naive_label_test,split_name="test"))


class Splitter(Dataset):

    def __init__(self, fmri,image,img_class, img_class_name,naive_label,split_name):
        super(Splitter, self).__init__()
        # Set EEG dataset
        self.fmri = fmri
        self.image=image
        self.img_class=img_class
        self.img_class_name=img_class_name
        self.naive_label=naive_label
        self.num_voxels=fmri.shape[2]
        self.return_image_info=False
        self.num_per_sub = fmri.shape[0]
        self.split_name=split_name

    # Get size
    def __len__(self):
        return len(self.fmri)

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        fmri =  self.fmri[i]
        image =  self.image[i]
        # img_class =  self.img_class[i]
        # img_class_name =  self.img_class_name[i]
        # naive_label =  self.naive_label[i]
        if self.split_name == 'train':

            fmri = random_blank(fmri)

        return {'fmri': fmri, 'image': image}
