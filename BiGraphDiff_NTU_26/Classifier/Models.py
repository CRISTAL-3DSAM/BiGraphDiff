import torch
import torch.nn as nn
import Layers_Space_multihead
from Embed import PositionalEncoder
from Sublayers_space_multihead import Norm
import torch.nn.functional as F
import copy

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



class Mlp(nn.Module):
    def __init__(self, emb_dim, mlp_dim,output_dim, dropout_rate=0.):
        super(Mlp, self).__init__()
        self.fc1 = torch.nn.Linear(emb_dim, mlp_dim)
        self.fc2 = torch.nn.Linear(mlp_dim, output_dim)
        self.act = torch.nn.GELU()
        self.dropout= torch.nn.Dropout(dropout_rate)


    def forward(self, x):

        out = self.fc1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out




class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout,max_frames):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model,max_seq_len=max_frames ,dropout=dropout)

        self.layers = get_clones(Layers_Space_multihead.EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask,is_test):
        x = self.pe(src)
        att_val_enc = {}
        #torch.manual_seed(999997589)
        for i in range(self.N):
            x,att_val = self.layers[i](x, mask,is_test)
            str_val = 'layer' + str(i)
            att_val_enc[str_val] = att_val
        return self.norm(x),att_val_enc
    
class Decoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout,max_frames):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model,max_seq_len=max_frames, dropout=dropout)

        self.layers = get_clones(Layers_Space_multihead.DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs,distances, src_mask, trg_mask,is_test):
        x = self.pe(trg)
        att_val_dec = {}
        for i in range(self.N):
            x,att_val = self.layers[i](x, e_outputs,distances, src_mask, trg_mask,is_test)
            str_val = 'layer' + str(i)
            att_val_dec[str_val] = att_val
        return self.norm(x), att_val_dec

class Transformer(nn.Module):
    def __init__(self, d_model,n_classes, N, heads, dropout,max_frames):
        super().__init__()
        self.encoder = Encoder(d_model, N, heads, dropout,max_frames)
        #self.decoder = Decoder(d_model, N, heads, dropout,max_frames)
        self.preout = nn.Conv1d(d_model*max_frames,d_model,kernel_size=1)
        self.out = nn.Linear(512,n_classes)
        self.custom_ml = Mlp(d_model*max_frames,1024,512,0.01)
    def forward(self, src, src_mask,is_test):
        e_outputs,_ = self.encoder(src, src_mask,is_test)
        #print("DECODER")
        #d_output,_ = self.decoder(trg, e_outputs,distances, src_mask, trg_mask,is_test)
        e_outputs = torch.reshape(e_outputs,(e_outputs.shape[0],e_outputs.shape[2]*e_outputs.shape[1]))
        #
        #e_outputs = torch.unsqueeze(e_outputs,2)
        pre_output = self.custom_ml(e_outputs)
        pre_output= torch.squeeze(pre_output)
        output = self.out(pre_output)
        #output = torch.mean(output,axis=1)
        return output,pre_output #F.softmax(output)

def get_model(opt):
    
    assert (opt.nb_joints*3) % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(opt.nb_joints*3, opt.nb_classes, opt.n_layers, opt.heads, opt.dropout,opt.max_frames)
       
    if opt.load_weights is not None:
        print("loading pretrained weights...")
        #model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights_{opt.nb_model}'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    if opt.device == 0:
        model = model.cuda()
    
    return model
    
