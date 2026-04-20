from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import matplotlib.pyplot as plt
from guided_diffusion.nn import checkpoint
# from guided_diffusion.unet import CrossBlock
from abc import abstractmethod

class CrossBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    # def forward(self, x, id_emd, landmark_emd):
    def forward(self, x, context):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        if context_dim is not None:
            self.to_k_landmark = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_landmark = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
    # def forward(self, x, id_emd=None, landmark=None, mask=None):
        h = self.heads

        q = self.to_q(x)

        # if id_emd is None:
        #     k, v = self.to_k(x), self.to_v(x)
        # else:
        #     k_id, v_id = self.to_k(id_emd), self.to_v(id_emd)
        #     k_landmark, v_landmark = self.to_k(landmark), self.to_v(landmark)
        #     k, v = torch.cat((k_id, k_landmark), dim=1), torch.cat((v_id, v_landmark), dim=1)


        context = default(context, x)   # context if context is not None else x
        k,v = self.to_k(context), self.to_v(context)

        # if landmark is None:
        #     k,v = self.to_k(x),self.to_v(x)
        # else:
        #     k = self.to_k(landmark)       # landmark feature
        #     v = self.to_v(id_emd)       # id feature
        # print(f"q: {q.shape}")
        # print(f"k: {k.shape}")
        # print(f"v: {v.shape}")
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        # [batch_size, 序列长度, head_nums, head_dim]
        # rearrange函数的作用是根据提供的字符串重新排列张量的维度
        # lambda匿名函数 t作为输入
        # map() 将匿名函数作用到q,k,v这3个张量上，结果是一个包含三个重新排列后的张量的元组
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # 65x512            1024x512
        # 1024x65
        # 65x512

        # 1x512  1024x512
        # 1024x1
        # 1x512
        # einsum计算点积
        # bid,从q中取出维度为b,i,d的元素， bjd同理  bjd包含着q乘以的k的转置
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)


        ############### attention map###################
        # # 假设 attn 是你的 attention 权重矩阵，形状为 [batch_size, num_heads, sequence_length, sequence_length]
        # # 这里我们取第一个样本和第一个头的attention map作为例子
        # attn_map = attn[0, 0, :, :].detach().cpu().numpy()
        #
        # # 使用matplotlib生成attention map的图像
        # plt.imshow(attn_map, cmap='hot', interpolation='nearest')
        # plt.colorbar()
        # plt.title('Attention Map')
        # plt.xlabel('Keys')
        # plt.ylabel('Queries')
        # # 将图像保存为文件
        # plt.savefig('attention_map.png')  # 将图像保存为PNG格式的文件

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    # def forward(self, x, id_emd=None, landmark_emd=None):
    def forward(self, x, context=None):
        # return checkpoint(self._forward, (x, id_emd, landmark_emd), self.parameters(), self.checkpoint)
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
    # def _forward(self, x, id_emd=None, landmark_emd=None):
        x = self.attn1(self.norm1(x)) + x
        # x = self.attn2(self.norm2(x), id_emd, landmark_emd) + x
        x = self.attn2(self.norm2(x), context) + x
        x = self.ff(self.norm3(x)) + x
        return x


# class SpatialTransformer(nn.Module):
class SpatialTransformer(CrossBlock):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
    # def forward(self, x, id_emd=None, landmark_emd=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        # print(f"x.shape: {x.shape}")
        x_in = x
        # print(f"x_in.shape: {x_in.shape}")        [3,256,32,32]
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')    # 6,512,32,32

        # print(f"x.shape: {x.shape}")
        # print(f"landmark_emd.shape: {landmark_emd.shape}")
        # print(f"id_emd.shape: {id_emd.shape}")

        # print(f"x.shape: {x.shape}")        # 6, 1024, 512
        # print(f"context.shape: {context.shape}")      # 6, 2048

        # context = context.unsqueeze(1).repeat(1, 1024, 1)   # [6,1,2048]
        # print(f"context[0].shape: {context[0].shape}")
        # print(f"context[1].shape: {context[1].shape}")
        # context[0] = context[0].unsqueeze(1).repeat(1, 1024, 1)
        # context[1] = context[1].unsqueeze(1).repeat(1, 1024, 1)
        # repeat 在dim0重复1次，在dim1重复1024次         # [6,1024,2048]
        # print(f"context.shape: {context.shape}")
        # print(f"x.shape: {x.shape}")                # 3, 1024, 256
        # print(f"context.shape: {context.shape}")    # 3 ,1024, 1024
        for block in self.transformer_blocks:
            x = block(x, context=context)
            # x = block(x, id_emd=id_emd, landmark_emd=landmark_emd)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in