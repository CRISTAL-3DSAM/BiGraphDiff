import numpy as np
from scipy.io import loadmat
import os
import pickle

def load_text_file( file):
    # get path to data file x = input, y= condition , frame = first frame of reaction sequence, gram = path to folder with gram matrix for each frame
    X = []
    y = []
    classes= []
    cl_fold=[]
    for line in open(file, 'r'):
        data = line.split()
        X.append(data[0])
        y.append(data[1])
        cl = data[1][10:14]
        if cl=='test':
            cl = data[1][15:19]
        cl_fold.append(cl)
        cl_name=class_name(cl)
        classes.append(cl_name)
    return X, y,classes

def class_name(nb):
    if nb=='A050':
        name='punching'
    elif nb=='A051':
        name='kicking'
    elif nb=='A052':
        name='pushing'
    elif nb=='A053':
        name='pat on back'
    elif nb=='A054':
        name='point finger'
    elif nb=='A055':
        name='hugging'
    elif nb=='A056':
        name='giving object'
    elif nb=='A057':
        name='touch pocket'
    elif nb=='A058':
        name='shaking hands'
    elif nb=='A059':
        name='walking towards'
    elif nb=='A060':
        name='walking apart'
    elif nb=='A106':
        name='hit with object'
    elif nb=='A107':
        name='wield knife'
    elif nb=='A108':
        name='knock over'
    elif nb=='A109':
        name='grab stuff'
    elif nb=='A110':
        name='shoot with gun'
    elif nb=='A111':
        name='step on foot'
    elif nb=='A112':
        name='high-five'
    elif nb=='A113':
        name='cheers and drink'
    elif nb=='A114':
        name='carry object'
    elif nb=='A115':
        name='take a photo'
    elif nb=='A116':
        name='follow'
    elif nb=='A117':
        name='whisper'
    elif nb=='A118':
        name='exchange things'
    elif nb=='A119':
        name='support somebody'
    elif nb=='A120':
        name='rock paper scissors'
    else:
        name='error'
    return name


def read_frames(path):
    data_ = loadmat(path)
    data = data_['skel']
    return data


def Load_Data(path,data_dir):

    Data_in, Data_out,Labels = load_text_file(path)
    Data_in = [os.path.join(data_dir, x) for x in Data_in]
    Data_out = [os.path.join(data_dir, y) for y in Data_out]

    max_frames = 214
    for ind in range(len(Data_in)):
        batch_in = Data_in[ind]
        batch_in = read_frames(batch_in)
        batch_in = np.array(batch_in).astype(np.float32)
        if max_frames<batch_in.shape[1]:
            max_frames=batch_in.shape[1]
    all_data_in = []

    for ind in range(len(Data_in)):
        batch_in = Data_in[ind]
        batch_in = read_frames(batch_in)
        data_in = np.array(batch_in).astype(np.float32)
        batch_out = Data_out[ind]
        batch_out = read_frames(batch_out)
        data_out = np.array(batch_out).astype(np.float32)
        all_data_in.append(np.concatenate((data_in,data_out),axis=0))
    return all_data_in,Labels,max_frames+1


