import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
import time
import torch
import Load_a2m_DD
from torch.utils.tensorboard import SummaryWriter
import gaussian_diffusion as gf
import numpy as np
from Batch_t5 import nopeak_mask
import Transformer_diffuse_dual_BIGraph as TRD



def get_reaction(sentence, model,diffusion,m_lens, opt):

    model.eval()

    # sentence=torch.from_numpy(sentence)
    # if opt.device == 0:
    #     sentence = sentence.cuda()

    B = sentence.shape[0]
    T = min(m_lens.max(), opt.max_nb_of_frames)
    #src_mask = (src[:, :, 0] != opt.src_pad).unsqueeze(1)
    trg_mask = nopeak_mask(opt.max_nb_of_frames, opt)
    reaction = diffusion.p_sample_loop(
        model,
        (B, T, opt.nb_joints),
        clip_denoised=False,
        progress=False,
        model_kwargs={"text": sentence, "length": m_lens, 'src_mask': None, 'trg_mask': trg_mask})
    #reaction,att_enc,att_dec = beam_search(sentence.cuda(), model, opt)
    torch.cuda.empty_cache()
    gr = torch.greater_equal(reaction,0.91)
    lw = torch.less_equal(reaction,1.09)
    to = torch.logical_and(gr,lw)
    for b in range (reaction.shape[0]):
        for t in range(reaction.shape[1]):
            t_ten = torch.count_nonzero(to[b,t,:])
            if torch.greater(t_ten,40):
                reaction[b,t,:] = torch.ones(reaction.shape[2])

    return reaction


def train_model(model,diffusion,sampler, opt):
    
    print("training model...")
    if opt.load_weights is not None:
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights_best'), strict=True)
    model.train()
    start = time.time()
    avg_loss = 100
    mse_criterion = torch.nn.MSELoss(reduction='none')
    if opt.checkpoint > 0:
        cptime = time.time()

    input_train = np.asarray(opt.data_input)
    output_train = np.asarray(opt.data_output)
    motion_len = np.asarray(opt.l_motion)

    nb_save = int(opt.epochs/100)

    writer = SummaryWriter()
    losses = []
    best_loss = 100000000
    epoch_since_last_best=0
    for epoch in range(opt.epochs):

        max_normal_epoch = 100000
        total_loss = 0
        total_loss_train = 0
        loss_dat = 0
        total_loss_ff=0
        react_done=False

        annot = np.asarray(opt.annotations)
        if opt.floyd is False:
            print("   %dm: epoch %d [%s]  %d%%  loss = %s" %\
            ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')
        
        if opt.checkpoint > 0:
            torch.save(model.state_dict(), 'weights/model_weights')

        nb_batch = input_train.shape[0]//opt.batchsize

        full_batch=False
        if ((input_train.shape[0]%opt.batchsize)==0):
            full_batch=True
        else:
            nb_batch=nb_batch+1

        for i in range(nb_batch):

            if not full_batch and i==nb_batch-1:
                src = input_train[i*opt.batchsize:,:]
                trg = output_train[i*opt.batchsize:,:,:]
                m_lens = motion_len[i*opt.batchsize:]
                anno = annot[i*opt.batchsize:]

            else:
                src = input_train[i*opt.batchsize:(i+1)*opt.batchsize,:]
                trg = output_train[i*opt.batchsize:(i+1)*opt.batchsize,:,:]
                m_lens = motion_len[i*opt.batchsize:(i+1)*opt.batchsize]
                anno = annot[i*opt.batchsize:(i+1)*opt.batchsize]

            k_rand = np.min(np.argpartition(np.random.rand(src.shape[0]), -2)[-2:])
            src=torch.from_numpy(src)
            trg=torch.from_numpy(trg)
            trg = trg.float()
            trg = torch.transpose(trg,1,2)
            trg_input = trg[:,:-1,:]

            if opt.device == 0:
                motions=trg.cuda()

            x_start = motions
            B, T = x_start.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in m_lens]).cuda()
            src_mask = model.generate_src_mask(T,cur_len).cuda()
            #test_mask = src_mask.cpu().detach().numpy()
            t, _ = sampler.sample(B, x_start.device)

            output = diffusion.training_losses(
                model=model,
                x_start=x_start,
                t=t,
                model_kwargs={"text": anno, "length": cur_len}
            )

            real_noise = output['target']
            fake_noise = output['pred']

            torch.optim.Optimizer.zero_grad(opt.optimizer)

            #loss = F.mse_loss(fake_noise, real_noise,reduction='mean')
            #loss = torch.mul(loss,1000)
            loss = mse_criterion(fake_noise, real_noise).mean(dim=-1)
            loss= (loss * src_mask).sum() / src_mask.sum()
            total_loss_train+=loss.item()



            # generated_seq = get_reaction(text[k_rand:k_rand+2],model,diffusion,cur_len[k_rand:k_rand+2],opt)
            # loss_data = mse_criterion(generated_seq, motions[k_rand:k_rand+2]).mean(dim=-1)
            # loss_data= loss_data.sum()
            # loss_data = torch.mul(loss_data,10)
            # loss_dat =loss_data.item()
            # a=1
            # total_loss_train+=loss_data.item()


            current_loss = loss#+loss_data
            current_loss.backward()

            if epoch_since_last_best>=100:
                for g in opt.optimizer.param_groups:
                    g['lr'] = g['lr']*0.1
                epoch_since_last_best = 1


            opt.optimizer.step()

            #opt.sched.step(current_loss)


            total_loss += loss.item()#+total_loss_eval+loss_ff_print

            writer.add_scalar('Loss/train', loss.item(), epoch+i)
            #writer.add_scalar('Loss/eval', total_loss_eval, epoch+i)


            if (i + 1) % opt.printevery == 0:
                 p = int(100 * (i + 1) / nb_batch)
                 avg_loss = loss.item()
                 if opt.floyd is False:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f  loss_train = %.3f loss_data = %.3f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss, loss,loss_dat), end='\r')
                 else:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f  loss_train = %.3f loss_data = %.3f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss,loss,loss_dat))

            cwd=os.getcwd()
            cwd=cwd.replace("\\", "/")
            if opt.checkpoint > 0 and ((time.time()-cptime)//60) // opt.checkpoint >= 1:
                torch.save(model.state_dict(), cwd+'weights/model_weights')
                cptime = time.time()

        losses.append(total_loss_train/nb_batch)
        avg_loss = total_loss/nb_batch
        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f  loss_train = %.3f loss_data = %.3f " %\
        ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, avg_loss, total_loss_train/nb_batch,loss_dat))
        total_loss = 0
        if epoch%1000==0:
            dst= '/temp_save'
            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), cwd+f'{dst}/model_weights')
        if epoch % 1000 == 0:
            dst = '/temp_save'
            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), cwd + f'{dst}/model_weights'+str(epoch))
        if epoch==max_normal_epoch:
            dst= '/temp_save'
            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), cwd+f'{dst}/model_weights_after_normal_train')
        if (total_loss_train/nb_batch)<best_loss:
            best_loss=total_loss_train/nb_batch
            dst= '/temp_save'
            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), cwd+f'{dst}/model_weights_best')
            epoch_since_last_best = 1
        epoch_since_last_best = epoch_since_last_best+1
    writer.flush()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str,default='data')
    parser.add_argument('-train_file', type=str,default='data/Train.txt')
    parser.add_argument('-nb_joints', type=int, default=90)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=2)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=2)
    parser.add_argument('-printevery', type=int, default=1)
    parser.add_argument('-lr', type=int, default=1e-4)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('-eos_dist', type=float, default=0.05)
    parser.add_argument('-e_model', type=int, default=128) #text_emb
    parser.add_argument('-l_model', type=int, default=512) #joint emb
    parser.add_argument('-diffusion_steps', type=int, default=1000)
    #parser.add_argument('-max_len', type=int, default=48) # ajouté pour test, à intégrer dans le code
    parser.add_argument('-max_nb_of_frames', type=int, default=301)


    opt = parser.parse_args()
    opt.is_test = False
    opt.device = 0 if opt.no_cuda is False else -1
    if opt.device == 0:
        assert torch.cuda.is_available()


    opt.src_pad = 1 # 1 is EOS and PAD
    opt.trg_pad = 1
    #opt.data_input ,opt.data_output,opt.max_frames,opt.vocab_size= format_t2m.Load_Data(opt.data_dir)
    #opt.data_input ,opt.data_output,opt.max_frames,opt.vocab_size,annotations,nb_word,opt.l_motion= Load_t2m.Load_Data(opt.data_dir,20,opt.max_nb_of_frames)
    opt.data_input ,opt.data_output,opt.max_frames,opt.vocab_size,opt.annotations,nb_word,opt.l_motion,opt.cl_fold= Load_a2m_DD.Load_Data(opt.train_file,opt.data_dir,int(opt.nb_joints/6))

    print(max(opt.l_motion))

    #model = get_model(opt)
    model = TRD.MotionTransformer(90,num_frames=301)
    model = model.cuda()


    sampler = 'uniform'
    beta_scheduler = 'linear'
    betas = gf.get_named_beta_schedule(beta_scheduler, opt.diffusion_steps)
    diffusion = gf.GaussianDiffusion(
        betas=betas,
        model_mean_type=gf.ModelMeanType.EPSILON ,
        model_var_type=gf.ModelVarType.FIXED_SMALL,
        loss_type=gf.LossType.MSE
    )#base is epsilon
    sampler = gf.create_named_schedule_sampler(sampler, diffusion)

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    opt.sched =  torch.optim.lr_scheduler.ReduceLROnPlateau(opt.optimizer, factor=0.5,patience=2,cooldown=10)

    if opt.checkpoint > 0:
        print("model weights will be saved every %d minutes and at end of epoch to directory weights/"%(opt.checkpoint))

    
    train_model(model,diffusion,sampler, opt)

    if opt.floyd is False:
        promptNextAction(model, opt)

def yesno(response):
    while True:
        if response != 'y' and response != 'n':
            response = input('command not recognised, enter y or n : ')
        else:
            return response

def promptNextAction(model, opt):

    saved_once = 1 if opt.load_weights is not None or opt.checkpoint > 0 else 0
    
    if opt.load_weights is not None:
        dst = opt.load_weights
    if opt.checkpoint > 0:
        dst = 'weights'

    while True:
        save = yesno(input('training complete, save results? [y/n] : '))
        if save == 'y':
            while True:
                if saved_once != 0:
                    res = yesno("save to same folder? [y/n] : ")
                    if res == 'y':
                        break
                dst = input('enter folder name to create for weights (no spaces) : ')
                if ' ' in dst or len(dst) < 1 or len(dst) > 30:
                    dst = input("name must not contain spaces and be between 1 and 30 characters length, enter again : ")
                else:
                    try:
                        os.mkdir(dst)
                    except:
                        res= yesno(input(dst + " already exists, use anyway? [y/n] : "))
                        if res == 'n':
                            continue
                    break
            
            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), f'{dst}/model_weights')
            #if saved_once == 0:

            #    saved_once = 1
            
            print("weights and field pickles saved to " + dst)

        res = yesno(input("train for more epochs? [y/n] : "))
        if res == 'y':
            while True:
                epochs = input("type number of epochs to train for : ")
                try:
                    epochs = int(epochs)
                except:
                    print("input not a number")
                    continue
                if epochs < 1:
                    print("epochs must be at least 1")
                    continue
                else:
                    break
            opt.epochs = epochs
            train_model(model, opt)
        else:
            print("exiting program...")
            break

    # for asking about further training use while true loop, and return
if __name__ == "__main__":
    main()
