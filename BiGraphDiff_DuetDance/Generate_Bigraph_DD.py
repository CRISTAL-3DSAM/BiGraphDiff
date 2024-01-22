import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
import Load_a2m_DD
import shutil

def plot_skeleton(skeleton, frame, color,ax):
    ax.plot(skeleton[:15, frame], skeleton[30:45, frame], -skeleton[15:30, frame], color + 'o')
    # [0, 1, 2] for head and body
    ax.plot(np.append(skeleton[12, frame],np.append(skeleton[14, frame],skeleton[13, frame])), np.append(skeleton[42, frame],np.append(skeleton[44, frame],skeleton[43, frame])),-np.append(skeleton[27, frame],np.append(skeleton[29, frame],skeleton[28, frame])), color + '-')
    # [1, 3, 4, 5] and [1, 6, 7, 8] for arms
    ax.plot(np.append(np.append(skeleton[14, frame], skeleton[11, frame]),np.append(skeleton[9, frame],skeleton[7, frame])), np.append(np.append(skeleton[44, frame], skeleton[41, frame]),np.append(skeleton[39, frame],skeleton[37, frame])), -np.append(np.append(skeleton[29, frame], skeleton[26, frame]),np.append(skeleton[24, frame],skeleton[22, frame])), color + '-', alpha=0.3)
    ax.plot(np.append(np.append(skeleton[14, frame], skeleton[10, frame]),np.append(skeleton[8, frame],skeleton[6, frame])), np.append(np.append(skeleton[44, frame], skeleton[40, frame]),np.append(skeleton[38, frame],skeleton[36, frame])), -np.append(np.append(skeleton[29, frame], skeleton[25, frame]),np.append(skeleton[23, frame],skeleton[21, frame])), color + '-', alpha=0.3)
    # [2, 9, 10, 11] et [2, 12, 13, 14] for legs
    ax.plot(np.append(np.append(skeleton[13, frame], skeleton[5, frame]),np.append(skeleton[3, frame],skeleton[1, frame])), np.append(np.append(skeleton[43, frame], skeleton[35, frame]),np.append(skeleton[33, frame],skeleton[31, frame])), -np.append(np.append(skeleton[28, frame], skeleton[20, frame]),np.append(skeleton[18, frame],skeleton[16, frame])), color + '-', alpha=0.3)
    ax.plot(np.append(np.append(skeleton[13, frame], skeleton[4, frame]),np.append(skeleton[2, frame],skeleton[0, frame])), np.append(np.append(skeleton[43, frame], skeleton[34, frame]),np.append(skeleton[32, frame],skeleton[30, frame])), -np.append(np.append(skeleton[28, frame], skeleton[19, frame]),np.append(skeleton[17, frame],skeleton[15, frame])), color + '-', alpha=0.3)

    #{[1,9],[1,12],[1,15],[15,2],[15,3],[3,4],[4,5],[15,6],[6,7],[7,8],[9,10],[10,11],[12,13],[13,14]}

def show_skeleton(reaction,action,label,nb_seq,Path):
    images = []
    plt.ioff()
    fig, axs = plt.subplots(nrows=1, ncols=1,subplot_kw=dict(projection="3d"))
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle(label,fontsize=24)
    cwd=os.getcwd()
    cwd=cwd.replace("\\", "/")
    xlimu=0
    ylimu=0
    zlimu=0
    xliml=0
    yliml=0
    zliml=0
    if not os.path.exists(cwd+Path+'visual_tmp'):
        os.makedirs(cwd+Path+'visual_tmp')
    for i in range(2,reaction.shape[1]):
        ax= axs
        ax.clear()
        plot_skeleton(reaction, i, 'r',ax)#k
        plot_skeleton(action, i, 'b',ax)#g
        ax.axis('off')
        ax.view_init(180,90)
        if i==2:
          xliml,xlimu=ax.get_xlim3d()
          yliml,ylimu=ax.get_ylim3d()
          zliml,zlimu=ax.get_zlim3d()
        ax.set_xlim3d([xliml*2, xlimu*2])
        ax.set_ylim3d([yliml*2, ylimu*2])
        ax.set_zlim3d([zliml*2, zlimu*2])
        plt.axis('off')
        name =cwd+Path+'visual_tmp/visual_' + str(i) + '.png'
        plt.savefig(name)
        images.append(imageio.imread(name))
    if not os.path.exists(cwd+Path+label):
        os.makedirs(cwd+Path+label)
    imageio.mimsave(cwd+Path+label+'/movie_'+str(nb_seq)+'.gif', images)
    shutil.rmtree(cwd+Path+'visual_tmp')
    plt.close('all')



data_dir = 'BiGraphDiff/data_convert'
Path_save ='/Visuals/'
Motion ,Labels,max_frames,= Load_a2m_DD.Load_Data('Classifier/data/test_generated.txt',data_dir)
for b in range(len(Motion)):
    mot= Motion[b]
    reaction=mot[:45,:]
    action=mot[45:,:]
    label = Labels[b]
    show_skeleton(reaction, action,label, b,Path_save)
    print(b)
