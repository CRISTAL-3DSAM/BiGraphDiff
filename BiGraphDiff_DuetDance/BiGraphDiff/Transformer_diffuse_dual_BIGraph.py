# Code based on https://github.com/mingyuan-zhang/MotionDiffuse/blob/main/text2motion/models/transformer.py

#from cv2 import norm
import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
import clip
import BiGraph as BG

import math


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class StylizationBlock(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb).unsqueeze(1)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h


class LinearTemporalSelfAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb, src_mask):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, T, D
        key = (self.key(self.norm(x)) + (1 - src_mask) * -1000000)
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, T, H, -1), dim=1)
        # B, T, H, HD
        value = (self.value(self.norm(x)) * src_mask).view(B, T, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y

class LinearTemporalCrossAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, xf, emb):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, N, D
        key = self.key(self.text_norm(xf))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)
        # B, N, H, HD
        value = self.value(self.text_norm(xf)).view(B, N, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y

class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + self.proj_out(y, emb)
        return y


class LinearTemporalDiffusionTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 seq_len=60,
                 latent_dim=32,
                 text_latent_dim=512,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.1):
        super().__init__()
        self.sa_block_1 = LinearTemporalSelfAttention(seq_len, latent_dim, num_head, dropout, time_embed_dim)
        self.ca_block_1 = LinearTemporalCrossAttention(seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim)
        #self.spatial_block_1 = LinearSpatialSelfAttention(seq_len, latent_dim, num_head, dropout, time_embed_dim)

        self.sa_block_2 = LinearTemporalSelfAttention(seq_len, latent_dim, num_head, dropout, time_embed_dim)
        self.ca_block_2 = LinearTemporalCrossAttention(seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim)
        #self.spatial_block_2 = LinearSpatialSelfAttention(seq_len, latent_dim, num_head, dropout, time_embed_dim)

        #self.ca_block_cross_1 = LinearTemporalCrossAttention(seq_len, latent_dim, latent_dim, num_head, dropout, time_embed_dim)
        #self.ca_block_cross_2 = LinearTemporalCrossAttention(seq_len, latent_dim, latent_dim, num_head, dropout, time_embed_dim)


        self.ga = BG.GloRe_Unit_2D(32, 16)
        self.ffn = FFN(latent_dim*2, ffn_dim, dropout, time_embed_dim)
        self.ffn_1 = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)
        self.ffn_2 = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x1,x2, xf, emb, src_mask):
        x1 = self.sa_block_1(x1, emb, src_mask)
        #x1 = self.spatial_block_1(x1, emb, src_mask)
        x1 = self.ca_block_1(x1, xf, emb)

        x2 = self.sa_block_2(x2, emb, src_mask)
        #x2 = self.spatial_block_2(x2, emb, src_mask)
        x2 = self.ca_block_2(x2, xf, emb)

        #x1 = self.ca_block_cross_1(x1,x2,emb)
        #x2 = self.ca_block_cross_2(x2,x1,emb)

        x = torch.cat((x1,x2),dim=2)
        x_graph = x.reshape((x.shape[0]*x.shape[1],x.shape[2]))
        x_graph = x_graph.reshape(x_graph.shape[0],64,4,4)
        x_graph = self.ga(x_graph)
        x_graph = x_graph.reshape((x_graph.shape[0],1024))
        #aaaaa = x_graph.cpu().detach().numpy()
        x= x_graph.reshape((x.shape[0],x.shape[1],x.shape[2]))



        x1=x[:,:,:int(x.shape[2]/2)]
        x2=x[:,:,int(x.shape[2]/2):]

        x1 = self.ffn_1(x1, emb)
        x2 = self.ffn_2(x2, emb)

        return x1,x2

class MotionTransformer(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0,
                 activation="gelu",
                 num_text_layers=4,
                 text_latent_dim=256,
                 text_ff_size=2048,
                 text_num_heads=4,
                 no_clip=False,
                 no_eff=False,
                 **kargs):
        super().__init__()

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim * 4
        self.sequence_embedding = nn.Parameter(torch.randn(num_frames, latent_dim))

        # Text Transformer
        self.clip, _ = clip.load('ViT-B-32_file.pt', "cpu")
        if no_clip:
            self.clip.initialize_parameters()
        else:
            set_requires_grad(self.clip, False)
        if text_latent_dim != 512:
            self.text_pre_proj = nn.Linear(512, text_latent_dim)
        else:
            self.text_pre_proj = nn.Identity()
        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=text_latent_dim,
            nhead=text_num_heads,
            dim_feedforward=text_ff_size,
            dropout=dropout,
            activation=activation)
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer,
            num_layers=num_text_layers)
        self.text_ln = nn.LayerNorm(text_latent_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(text_latent_dim, self.time_embed_dim)
        )

        # Input Embedding
        self.joint_embed_1 = nn.Linear(int(self.input_feats/2), self.latent_dim)
        self.joint_embed_2 = nn.Linear(int(self.input_feats/2), self.latent_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.temporal_decoder_blocks.append(
                LinearTemporalDiffusionTransformerDecoderLayer(
                    seq_len=num_frames,
                    latent_dim=latent_dim,
                    text_latent_dim=text_latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout
                )
            )

        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim*2, self.input_feats))

    def encode_text(self, text, device):
        with torch.no_grad():
            text = clip.tokenize(text, truncate=True).to(device)
            x = self.clip.token_embedding(text).type(self.clip.dtype)  # [batch_size, n_ctx, d_model]

            x = x + self.clip.positional_embedding.type(self.clip.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip.transformer(x)
            x = self.clip.ln_final(x).type(self.clip.dtype)

        # T, B, D
        x = self.text_pre_proj(x)
        xf_out = self.textTransEncoder(x)
        xf_out = self.text_ln(xf_out)
        xf_proj = self.text_proj(xf_out[text.argmax(dim=-1), torch.arange(xf_out.shape[1])])
        # B, T, D
        xf_out = xf_out.permute(1, 0, 2)
        return xf_proj, xf_out

    def generate_src_mask(self, T, length):
        B = len(length)
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask

    def forward(self, x, timesteps, length=None, text=None, xf_proj=None, xf_out=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]
        if xf_proj is None or xf_out is None:
            xf_proj, xf_out = self.encode_text(text, x.device)

        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim)) + xf_proj

        # B, T, latent_dim

        # B, T, latent_dim
        h1 = self.joint_embed_1(x[:,:,:int(x.shape[2]/2)])
        h2= self.joint_embed_2(x[:,:,int(x.shape[2]/2):])
        h1 = h1 + self.sequence_embedding.unsqueeze(0)[:, :T, :]
        h2 = h2 + self.sequence_embedding.unsqueeze(0)[:, :T, :]

        src_mask = self.generate_src_mask(T, length).to(x.device).unsqueeze(-1)
        for module in self.temporal_decoder_blocks:
            h1,h2 = module(h1,h2, xf_out, emb, src_mask)

        h=torch.cat((h1,h2),dim=2)

        output = self.out(h).view(B, T, -1).contiguous()
        return output
