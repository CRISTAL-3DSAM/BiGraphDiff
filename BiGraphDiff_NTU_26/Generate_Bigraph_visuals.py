import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
import Load_a2m_NTU
import shutil

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

def show_skeleton(reaction,action,label,nb_seq,Path):
    images = []
    plt.ioff()
    fig, axs = plt.subplots(nrows=1, ncols=1,subplot_kw=dict(projection="3d"))
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle(label,fontsize=24)
    cwd=os.getcwd()
    cwd=cwd.replace("\\", "/")
    for i in range(2,reaction.shape[1]):
        ax= axs
        ax.clear()
        plot_skeleton(reaction, i, 'r',ax)
        plot_skeleton(action, i, 'b',ax)
        ax.axis('off')
        ax.view_init(180,90)
        plt.axis('off')
        if not os.path.exists(cwd+Path+'visual_tmp'):
            os.makedirs(cwd+Path+'visual_tmp')
        name =cwd+Path+'visual_tmp/visual_' + str(i) + '.png'
        plt.savefig(name)
        images.append(imageio.imread(name))
    if not os.path.exists(cwd+Path+label):
        os.makedirs(cwd+Path+label)
    imageio.mimsave(cwd+Path+label+'/movie_'+str(nb_seq)+'.gif', images)
    shutil.rmtree(cwd+Path+'visual_tmp')
    plt.close('all')



data_dir = 'BiGraphDiff/data_convert'
Path_save ='/visuals/'
Motion ,Labels,max_frames,= Load_a2m_NTU.Load_Data('Classifier/data/Test_generated.txt',data_dir)
for b in range(0,len(Motion)):
    mot= Motion[b]
    reaction=mot[:48,:]
    action=mot[48:,:]
    label = Labels[b]
    show_skeleton(reaction, action,label, b,Path_save)
    print(b)