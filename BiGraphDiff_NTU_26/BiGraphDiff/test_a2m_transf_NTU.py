import argparse
import torch
import Load_a2m_NTU_test
import numpy as np
import shutil
import os
import scipy.io
import xml.etree.ElementTree as ET
import gaussian_diffusion as gf
from Batch_t5 import nopeak_mask
import matplotlib.pyplot as plt
import imageio
import Transformer_diffuse_dual_BIGraph as TRD



def plot_skeleton(skeleton, frame, color,ax):
    ax.plot(skeleton[:16, frame], skeleton[32:48, frame], -skeleton[16:32, frame], color + 'o')
    # [0, 1, 2] for head and body
    ax.plot(np.append(skeleton[:2, frame],np.append(skeleton[15, frame],skeleton[2, frame])), np.append(skeleton[32:34, frame],np.append(skeleton[47, frame],skeleton[34, frame])),-np.append(skeleton[16:18, frame],np.append(skeleton[31, frame],skeleton[18, frame])), color + '-')
    # [1, 3, 4, 5] and [1, 6, 7, 8] for arms
    ax.plot(np.append(skeleton[15, frame], skeleton[3:6, frame]), np.append(skeleton[47, frame], skeleton[35:38, frame]), -np.append(skeleton[31, frame], skeleton[19:22, frame]), color + '-', alpha=0.3)
    ax.plot(np.append(skeleton[15, frame], skeleton[6:9, frame]), np.append(skeleton[47, frame], skeleton[38:41, frame]), -np.append(skeleton[31, frame], skeleton[22:25, frame]), color + '-')
    # [2, 9, 10, 11] et [2, 12, 13, 14] for legs
    ax.plot(np.append(skeleton[0, frame], skeleton[9:12, frame]), np.append(skeleton[32, frame], skeleton[41:44, frame]), -np.append(skeleton[16, frame], skeleton[25:28, frame]), color + '-', alpha=0.3)
    ax.plot(np.append(skeleton[0, frame], skeleton[12:15, frame]), np.append(skeleton[32, frame], skeleton[44:47, frame]), -np.append(skeleton[16, frame], skeleton[28:31, frame]), color + '-')

    #{[1,9],[1,12],[1,15],[15,2],[15,3],[3,4],[4,5],[15,6],[6,7],[7,8],[9,10],[10,11],[12,13],[13,14]}

def show_skeleton(reaction,action,GT_react,GT_act,label,nb_seq):
    images = []
    fig = plt.figure(figsize = (12,8))
    plt.ioff()
    fig, axs = plt.subplots(nrows=1, ncols=2,subplot_kw=dict(projection="3d"))
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle(label,fontsize=24)
    cwd=os.getcwd()
    cwd=cwd.replace("\\", "/")
    for i in range(2,GT_react.shape[1]):
        for k in range(0,len(axs)):
            ax= axs[k]
            ax.clear()
            if k == 0:
                plot_skeleton(GT_react, i, 'b',ax)
                ax.set_title('ground truth')
                plot_skeleton(GT_act, i, 'b',ax)
            else:
                plot_skeleton(reaction, i, 'b',ax)
                ax.set_title('generated')
                plot_skeleton(action, i, 'b',ax)

            ax.axis('off')

            ax.view_init(180,90)
            plt.axis('off')
        if not os.path.exists(cwd+'/visual/visual_tmp'):
            os.makedirs(cwd+'/visual/visual_tmp')
        name =cwd+'/visual/visual_tmp/visual_' + str(i) + '.png'
        plt.savefig(name)
        images.append(imageio.imread(name))
        if np.sum(GT_act[:,i])==48.0 and i>1:
             break
    imageio.mimsave(cwd+'/visual/movie_'+str(nb_seq)+'.gif', images)
    shutil.rmtree(cwd+'/visual/visual_tmp')
    plt.close('all')


def get_reaction(sentence, model,diffusion,m_lens,annotations, opt):

    model.eval()

    sentence=torch.from_numpy(sentence)
    if opt.device == 0:
        sentence = sentence.cuda()

    B = sentence.shape[0]
    T = min(m_lens.max(), opt.max_nb_of_frames)
    xf_proj, xf_out = model.encode_text(annotations, "cuda")
    reaction = diffusion.p_sample_loop(
        model,
        (B, T, opt.nb_joints),
        clip_denoised=False,
        progress=True,
        model_kwargs={'xf_proj': xf_proj,'xf_out': xf_out,'length': m_lens})
    #reaction,att_enc,att_dec = beam_search(sentence.cuda(), model, opt)
    torch.cuda.empty_cache()

    return reaction.cpu().detach().numpy()


def fill_array(arr, max_frames):
    nb_missing = max_frames-arr.shape[1]
    zero_arr = np.zeros((arr.shape[0],nb_missing,arr.shape[2]))
    filled_arr = np.concatenate((arr,zero_arr),axis=1)
    return filled_arr


def create_SBU_files(action,generated,labels,type):
    cwd= 'data_convert'
    shutil.rmtree(cwd+'/'+type)
    count_cl = {'A050':0,'A051':0,'A052':0,'A053':0,'A054':0,'A055':0,'A056':0,'A057':0,'A058':0,'A059':0,'A060':0,'A106':0,'A107':0,'A108':0,'A109':0,'A110':0,'A111':0,
                'A112':0,'A113':0,'A114':0,'A115':0,'A116':0,'A117':0,'A118':0,'A119':0,'A120':0,}
    for b in range(generated.shape[0]):
        cl_name = labels[b]
        count_cl[cl_name]=count_cl[cl_name]+1
        path = cwd+'/Skeletons_test/'+cl_name+'/'+str(count_cl[cl_name]).zfill(3)
        os.makedirs(path)
        ACT=action[b]
        GEN=generated[b]
        with open(path+'/skeleton_pos.txt', 'w+') as f:
            for i in range(1,generated.shape[1]):
                if np.sum(ACT[i,:])==48.0:
                    break
                gen = GEN[i,:]
                act = ACT[i,:]
                act_block = ''
                gen_block =''
                for j in range(int(generated.shape[2]/3)):

                    if j!=int(generated.shape[2]/3)-1:
                        gen_block=gen_block+str(gen[j])+','+str(gen[j+16])+','+str(gen[j+32])+','
                        act_block=act_block+str(act[j])+','+str(act[j+16])+','+str(act[j+32])+','
                    else:
                        gen_block=gen_block+str(gen[j])+','+str(gen[j+16])+','+str(gen[j+32])
                        act_block=act_block+str(act[j])+','+str(act[j+16])+','+str(act[j+32])+','

                if i!=generated.shape[1]-1:
                    act_block=str(i+1)+','+act_block
                    gen_block = gen_block+'\n'
                else:
                    act_block = str(i+1)+','+act_block
                    gen_block = gen_block
                full_block = act_block+gen_block
                f.write(full_block)

def create_NTU_files(action,generated,labels,type):
    cwd='data_convert'
    shutil.rmtree(cwd+'/'+type)
    count_cl = {'A050':0,'A051':0,'A052':0,'A053':0,'A054':0,'A055':0,'A056':0,'A057':0,'A058':0,'A059':0,'A060':0,'A106':0,'A107':0,'A108':0,'A109':0,'A110':0,'A111':0,
                'A112':0,'A113':0,'A114':0,'A115':0,'A116':0,'A117':0,'A118':0,'A119':0,'A120':0,}
    for b in range(generated.shape[0]):
        cl_name = labels[b]
        count_cl[cl_name]=count_cl[cl_name]+1
        path = cwd+'/Skeletons_test/'+cl_name+'/'+str(count_cl[cl_name]).zfill(4)
        os.makedirs(path)
        ACT=action[b]
        ACT=np.transpose(ACT)
        GEN=generated[b]
        GEN=np.transpose(GEN)
        stop=0
        #for i in range(ACT.shape[1]):
        #    act_part = ACT[:,i]
        #    if sum(act_part)==0:
        #        stop=i
        #        break
        ACT= ACT[:,:stop]
        GEN= GEN[:,:stop]
        scipy.io.savemat(path+'/skeleton_A.mat', dict([('skel', ACT)]))
        scipy.io.savemat(path+'/skeleton_B.mat', dict([('skel', GEN)]))





def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', required=True)
    parser.add_argument('-data_dir', type=str,default='data')
    parser.add_argument('-test_file', type=str,default='data/Test.txt')
    parser.add_argument('-nb_joints', type=int, default=96)
    parser.add_argument('-dim', type=int, default=3)
    parser.add_argument('-batch_size', type=int, default=2)
    parser.add_argument('-max_len', type=int, default=48)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=2)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-visual', action='store_true')
    parser.add_argument('-e_model', type=int, default=128)  # text_emb
    parser.add_argument('-l_model', type=int, default=512)  # joint emb
    parser.add_argument('-max_nb_of_frames', type=int, default=215)
    parser.add_argument('-diffusion_steps', type=int, default=1000)


    opt = parser.parse_args()
    opt.is_test = True
    opt.links  =np.asarray([[1,2],[2,4],[4,5],[5,6],[2,7],[7,8],[8,9],[2,3],[3,10],[10,11],[11,12],[3,13],[13,14],[14,15]])-1
    opt.device = 0 if opt.no_cuda is False else -1

    assert opt.max_len > 10

    #opt.data_input ,opt.data_output,opt.max_frames,opt.vocab_size, annots= Load_t2m_test.Load_Data(opt.data_dir,20,opt.max_nb_of_frames)
    opt.data_input ,opt.data_output,opt.max_frames,opt.vocab_size,opt.annotations,nb_word,opt.l_motion,opt.cl_fold= Load_a2m_NTU_test.Load_Data(opt.test_file,opt.data_dir,int(opt.nb_joints/6))

    opt.src_pad = 1 # 1 is EOS and PAD
    opt.trg_pad = 1
    model = TRD.MotionTransformer(96,num_frames=215)
    model = model.cuda()
    model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights_best'), strict=True)

    beta_scheduler = 'linear'
    betas = gf.get_named_beta_schedule(beta_scheduler, opt.diffusion_steps)
    diffusion = gf.GaussianDiffusion(
        betas=betas,
        model_mean_type=gf.ModelMeanType.EPSILON,
        model_var_type=gf.ModelVarType.FIXED_SMALL,
        loss_type=gf.LossType.MSE
    )

    outs = np.asarray(opt.data_output)

    sentences = np.asarray(opt.data_input)
    nb_batch = sentences.shape[0]//opt.batch_size
    motion_len = np.asarray(opt.l_motion)
    full_batch=False
    if ((sentences.shape[0]%opt.batch_size)==0):
        full_batch=True
    else:
        nb_batch=nb_batch+1

    generated= np.zeros([outs.shape[0],opt.max_frames,outs.shape[1]])
    print("generating reaction")
    for i in range(nb_batch):
        if not full_batch and i==nb_batch-1:
            sentence = sentences[i*opt.batch_size:,:]
            m_lens = motion_len[i * opt.batch_size:]
            anno = opt.annotations[i*opt.batch_size:]
        else:
            sentence = sentences[i*opt.batch_size:(i+1)*opt.batch_size,:]
            m_lens = motion_len[i*opt.batch_size:(i+1)*opt.batch_size]
            anno = opt.annotations[i*opt.batch_size:(i+1)*opt.batch_size]

        gen = get_reaction(sentence,model,diffusion,m_lens,anno, opt)
        gen = fill_array(gen,opt.max_frames)
        if not full_batch and i==nb_batch-1:
            generated[i*opt.batch_size:,:,:] = gen
        else:
            generated[i*opt.batch_size:(i+1)*opt.batch_size,:,:]=gen

    create_NTU_files(generated[:,:,:48],generated[:,:,48:],opt.cl_fold,'Skeletons_test')



    print("DONE")
if __name__ == '__main__':
    main()
