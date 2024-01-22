import argparse
from Models import get_model
from Beam import beam_search
import torch
import imageio
from Load_a2m_NTU import Load_Data
import numpy as np
import shutil
import os
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import scipy.io



def get_reaction(sentence, model, opt):

    model.eval()

    sentence=torch.from_numpy(sentence)
    if opt.device == 0:
        sentence = sentence.cuda()
    sentence = sentence.float()
    sentence = torch.transpose(sentence,1,2)
    src_mask = (sentence[:,:,0] != opt.src_pad).unsqueeze(1)
    predict,features = model(sentence.cuda(), src_mask,opt.is_test)

    predict = torch.nn.functional.softmax(predict)

    return predict.cpu().detach().numpy(),features.cpu().detach().numpy()

def class_name(nb):
    idx_orig = {0: "Punch",
                1: "Kicking",
                2: "pushing",
                3: "patonback",
                4: "pointfinger",
                5: "hugging",
                6: "givingobject",
                7: "touchpocket",
                8: "shakinghands",
                9: "walkingT",
                10: "walkingA",
                11: "hitwobject",
                12: "wieldknife",
                13: "knockover",
                14: "grabstuff",
                15: "shoot",
                16: "steponfoot",
                17: "highfive",
                18: "cheers",
                19: "carry",
                20: "photo",
                21: "follow",
                22: "whisper",
                23: "exchange",
                24: "support",
                25: "RPS"}

    name = idx_orig[nb]
    return name

def create_SBU_files(action,generated,labels):
    cwd=os.getcwd()
    cwd=cwd.replace("\\", "/")
    shutil.rmtree(cwd+'/data/sbu/SBU-Kinect-Interaction/s00s00/')
    count_cl = [0,0,0,0,0,0,0,0]
    for b in range(generated.shape[0]):
        cl_name = labels[b]
        cl_name = class_name(cl_name)
        count_cl[int(cl_name[1])-1] +=1
        path = cwd+'/data/sbu/SBU-Kinect-Interaction/s00s00/'+cl_name+'/'+str(count_cl[int(cl_name[1])-1]).zfill(3)
        os.makedirs(path)
        ACT=action[b]
        GEN=generated[b]
        with open(path+'/skeleton_pos.txt', 'w+') as f:
            for i in range(1,generated.shape[1]):
                if np.sum(ACT[i,:])==45.0:
                    break
                gen = GEN[i,:]
                act = ACT[i,:]
                act_block = ''
                gen_block =''
                for j in range(int(generated.shape[2]/3)):
                    act_block=act_block+str(act[j])+','+str(act[j+15])+','+str(act[j+30])+','
                    if j!=int(generated.shape[2]/3)-1:
                        gen_block=gen_block+str(gen[j])+','+str(gen[j+15])+','+str(gen[j+30])+','
                    else:
                        gen_block=gen_block+str(gen[j])+','+str(gen[j+15])+','+str(gen[j+30])

                if i!=generated.shape[1]-1:
                    full_block = str(i+1)+','+gen_block+'\n'
                else:
                    full_block = str(i+1)+','+gen_block
                f.write(full_block)

def get_precision(prediction_max,GT_max):
    diff =GT_max-prediction_max
    nb_bad = np.count_nonzero(diff)
    precision = ((GT_max.size-nb_bad)/GT_max.size)*100
    return precision

def confusion_matrix(prediction_max,GT_max):
    matrix= metrics.confusion_matrix(GT_max,prediction_max,normalize='true')
    return matrix


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', required=True)
    parser.add_argument('-data_dir', type=str,default='../BiGraphDiff/data_convert')
    parser.add_argument('-test_file', type=str,default='data/Test_generated.txt')
    parser.add_argument('-nb_joints', type=int, default=32)
    parser.add_argument('-dim', type=int, default=3)
    parser.add_argument('-batch_size', type=int, default=100)
    parser.add_argument('-max_len', type=int, default=215)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=3)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-visual', action='store_true')
    parser.add_argument('-nb_classes', type=float, default=26)

    opt = parser.parse_args()
    opt.is_test = True
    opt.links  =np.asarray([[1,2],[2,4],[4,5],[5,6],[2,7],[7,8],[8,9],[2,3],[3,10],[10,11],[11,12],[3,13],[13,14],[14,15]])-1
    opt.device = 0 if opt.no_cuda is False else -1

    assert opt.max_len > 10


    opt.nb_model = 'best'

    opt.data_input,opt.data_output,opt.max_frames = Load_Data(opt.test_file,opt.data_dir,opt.nb_joints,opt.nb_classes)
    opt.max_frames=215
    opt.src_pad = 1 # 1 is EOS and PAD
    opt.trg_pad = 1
    model = get_model(opt)

    sentences = opt.data_input
    nb_batch = sentences.shape[0]//opt.batch_size
    full_batch=False
    if ((sentences.shape[0]%opt.batch_size)==0):
        full_batch=True
    else:
        nb_batch=nb_batch+1

    generated= np.zeros([sentences.shape[0],sentences.shape[2],sentences.shape[1]])
    GT_max = np.argmax(opt.data_output,axis=1)
    prediction_max = np.zeros(GT_max.shape[0])
    features=[]
    print("generating reaction")
    for i in range(nb_batch):
        if not full_batch and i==nb_batch-1:
            sentence = sentences[i*opt.batch_size:,:,:]
        else:
            sentence = sentences[i*opt.batch_size:(i+1)*opt.batch_size,:,:]

        prediction,features_part = get_reaction(sentence,model, opt)
        features.append(features_part)
        prediction_max_tmp=np.argmax(prediction,axis=1)

        #scipy.io.savemat('sample.mat', dict([('x_test', sentence[0])]))

        if not full_batch and i==nb_batch-1:
            prediction_max[i*opt.batch_size:]=prediction_max_tmp
        else:
            prediction_max[i*opt.batch_size:(i+1)*opt.batch_size]=prediction_max_tmp
    features_mat = np.asarray(features)
    features_mat = np.reshape(features_mat,(features_mat.shape[0]*features_mat.shape[1],features_mat.shape[2]))
    scipy.io.savemat('features_OURS', dict([('features', features_mat),('class',GT_max)]))
    precision_tot = get_precision(prediction_max,GT_max)
    conf_mat=confusion_matrix(prediction_max,GT_max)
    for i in range(conf_mat.shape[0]):
        acc = conf_mat[i,i]
        line = conf_mat[i,:]
        line_id = np.argsort(line)
        if line[line_id[-1]]!=acc:
            max_prec = line[line_id[-1]]
            max_prec_id = line_id[-1]
        else:
            max_prec = line[line_id[-2]]
            max_prec_id = line_id[-2]
        name = class_name(i)
        name_max = class_name(max_prec_id)
        print(name +' : '+str(acc*100) +'%         most confused : '+ name_max + ' : '+str(max_prec*100)+'%')
    plt.matshow(conf_mat)
    for (i, j), z in np.ndenumerate(conf_mat):
        plt.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
    #plt.show()
    print('total : '+str(precision_tot))


if __name__ == '__main__':
    main()
