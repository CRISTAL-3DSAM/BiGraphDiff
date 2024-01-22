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
        cl = data[1][10:13]
        cl_fold.append(cl)
        cl_name=class_name(cl)
        classes.append(cl_name)
    seed = 2021
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    np.random.seed(seed)
    np.random.shuffle(classes)
    return X, y,classes,cl_fold

def class_name(nb):
    if nb=='cha':
        name='cha-cha'
    elif nb=='jiv':
        name='jive'
    elif nb=='rum':
        name='rumba'
    elif nb=='sal':
        name='salsa'
    elif nb=='sam':
        name='samba'
    else:
        name='error'
    return name


def read_frames(path):
    data_ = loadmat(path)
    data = data_['skel']
    return data


def add_sos_eos(data_in,data_out,max_frames,nb_joint):
    SOS = np.zeros((nb_joint*3,1)) # start of sequence pose replace with T pose
    EOS = np.ones((nb_joint*3,1)) # end of sequence pose replace with H pose
    missing_in = max_frames-data_in.shape[1]  # number of padding frames
    missing_out = max_frames-data_out.shape[1]
    data_in = np.concatenate((data_in,EOS),axis=1)
    data_out = np.concatenate((data_out,EOS),axis=1)
    pad_in=np.ones((nb_joint*3,missing_in))
    pad_out=np.ones((nb_joint*3,missing_out))
    data_in = np.concatenate((data_in,pad_in),axis=1)
    data_out = np.concatenate((data_out,pad_out),axis=1)

    return data_in,data_out

def get_vocabulary(annotations):
    vocab_frequency={}
    vocab={}
    max_len = 0
    for annotation in annotations:
        words = annotation.split()
        nbw =0
        for word in words:
            nbw=nbw+1
            word = word.lower()
            if word not in vocab_frequency:
                vocab_frequency[word]=1
            else:
                nb= vocab_frequency[word]
                vocab_frequency[word]=nb+1
        if max_len<nbw:
            max_len=nbw
    vocab_frequency={k: v for k, v in sorted(vocab_frequency.items(), key=lambda item: item[1], reverse=True)}

    i=1
    for word in vocab_frequency:
        vocab[word]=i
        i=i+1

    return vocab,max_len

def text2vocab(annotations,vocab,nb_words):
    all_sentences = []
    for annotation in annotations:
        words = annotation.split()
        sentence = np.zeros(nb_words+1,dtype=int)
        i=1
        for word in words:
            word=word.lower()
            val = vocab[word]
            sentence[i]=val+1
            i=i+1
        sentence[i:]=1
        all_sentences.append(sentence)
    return all_sentences


def Load_Data(path,data_dir,nb_joint):

    Data_in, Data_out,classes,cl_fold = load_text_file(path)
    Data_in = [os.path.join(data_dir, x) for x in Data_in]
    Data_out = [os.path.join(data_dir, y) for y in Data_out]

    seed = 2021
    np.random.seed(seed)
    np.random.shuffle(Data_in)
    np.random.seed(seed)
    np.random.shuffle(Data_out)
    np.random.seed(seed)
    np.random.shuffle(classes)
    np.random.seed(seed)
    np.random.shuffle(cl_fold)

    max_frames = 200
    for ind in range(len(Data_in)):
        batch_in = Data_in[ind]
        batch_in = read_frames(batch_in)
        batch_in = np.array(batch_in).astype(np.float32)
        if max_frames<batch_in.shape[1]:
            max_frames=batch_in.shape[1]
    all_data_in = np.zeros((len(Data_in),nb_joint*3,max_frames+1))
    all_data_out = np.zeros((len(Data_in),nb_joint*2*3,max_frames+1))

    vocab,nb_words=get_vocabulary(classes)
    with open("vocabulary.pkl", "wb") as File:
        pickle.dump(vocab, File)
    with open("max_words.pkl", "wb") as File:
        pickle.dump(nb_words, File)
    emb_text=text2vocab(classes,vocab,nb_words)
    l_motion=[]
    for ind in range(len(Data_in)):
        batch_in = Data_in[ind]
        batch_in = read_frames(batch_in)
        batch_in = np.array(batch_in).astype(np.float32)
        batch_out = Data_out[ind]
        batch_out = read_frames(batch_out)
        batch_out = np.array(batch_out).astype(np.float32)
        data_in, data_out =add_sos_eos(batch_in,batch_out,max_frames,nb_joint)
        data_concat = np.concatenate((data_in,data_out),axis=0)
        l_motion.append(batch_in.shape[1])
        all_data_out[ind,:,:] = data_concat
    return emb_text, all_data_out, max_frames+1,len(vocab),classes,nb_words,l_motion,cl_fold


