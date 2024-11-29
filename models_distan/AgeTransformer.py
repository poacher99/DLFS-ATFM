import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# from torchvision.models import resnet50
import math

class ATF(nn.Module):
    def __init__(self, style_dim, num_layers=8, residual_layer=4):
        super(ATF, self).__init__()
        self.style_dim = style_dim
        self.num_layers = num_layers
        self.conv1 = nn.Linear(256,style_dim)
        self.norm1 = nn.BatchNorm1d(style_dim)
        self.act1 = nn.SiLU()
        self.mean_layer = nn.Linear(style_dim,style_dim)
        self.log_var_layer = nn.Linear(style_dim, style_dim)
        self.cross_att = SelfAttention(4,style_dim,style_dim,0.3)
        self.norm2 = nn.BatchNorm1d(style_dim)
        self.act2 = nn.SiLU()
        self.out = nn.Linear(style_dim, style_dim)
        
    def forward(self, x, target):
        x = x.reshape(-1,256)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)
        sigma = torch.sqrt(torch.exp(log_var))
        eps = torch.randn_like(mean).to(x.device)
        z_0 = mean + sigma * eps
        # for layer in self.affine_transformation_layers:
        #     x = layer(x)
        z = self.cross_att(z_0,target)
        # noise = torch.rand_like(z).to(z.device)
        # x = self.out(z + noise)
        x = self.out(z)
        kl_loss = torch.sum(0.5 * (torch.square(mean) + torch.square(torch.exp(log_var)) - 2 * torch.log(torch.exp(log_var)) - 1), dim = -1)
        
        return x, kl_loss
        
        
    def affine_transformation_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels,out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        
        
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
        
class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.pos_embed = torch.sin(1 / (10000 ** (torch.arange(0., hidden_size)/(hidden_size - 1))))

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(hidden_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        
        self.att_var_smoothing = Att_variance_smoothing()
        self.out = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1)

    def forward(self, input_tensor, target):
        identity = input_tensor
        input_tensor = input_tensor * self.pos_embed.to(input_tensor.device)
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(target)
        mixed_value_layer = self.value(target)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = torch.mul(hidden_states, input_tensor)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        
        hidden_states = self.att_var_smoothing(hidden_states, identity)
        hidden_states = self.out(hidden_states + target)
        
        return hidden_states
    

class GaussianBlurConv(nn.Module):
    def __init__(self):
        super(GaussianBlurConv, self).__init__()
        kernel_size = 5
        sigma = 0.5
        # Create a 1D Gaussian kernel
        # self.kernel = self.create_gaussian_kernel(kernel_size, sigma)
        self.kernel = torch.Tensor(self.create_gaussian_kernel(kernel_size, sigma))
        self.padding = kernel_size // 2

    def create_gaussian_kernel(self, kernel_size, sigma):
        x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        kernel = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, -1)

    def forward(self, x):
        batch_size, seq_len = x.size()
        x = x.view(batch_size, 1, seq_len)  # Add channel dimension
        x = F.conv1d(x, self.kernel.to(x.device), padding=self.padding, groups=1)
        x = x.view(batch_size, seq_len)  # Remove channel dimension
        return x

class Att_variance_smoothing(nn.Module):
    def __init__(self):
        super(Att_variance_smoothing, self).__init__()
        self.gaussion_conv = GaussianBlurConv()
        self.lambda_guide = torch.tensor(5.0)
        # self.lambda_target = torch.tensor(1.0)
        
    def forward(self, x, target):
        x = self.gaussion_conv(x)
        out = target + self.lambda_guide * (x - target)
        return out