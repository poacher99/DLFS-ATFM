import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# from torchvision.models import resnet50
import math

class ATF(nn.Module):
    def __init__(self, style_dim, num_layers=8):
        super(ATF, self).__init__()
        self.style_dim = style_dim
        self.num_layers = num_layers
        
        self.texture_mlp = []
        for i in range(num_layers):
            self.texture_mlp.append(residual_block(style_dim, style_dim))
        self.texture_mlp.append(PixelNorm())
        
        self.texture_mlp = nn.Sequential(*self.texture_mlp)
        
        self.norm1 = nn.LayerNorm(256)
        self.conv1 = nn.Linear(256, style_dim)
        self.act1 = nn.LeakyReLU(0.2)
        
        self.mean_layer = nn.Linear(style_dim,style_dim)
        self.log_var_layer = nn.Linear(style_dim, style_dim)
        self.cross_att = SelfAttention(4,style_dim,style_dim,0.3)
        
        self.norm3 = nn.LayerNorm(style_dim)
        self.out = nn.Linear(style_dim, style_dim)
        
    def forward(self, x, target):
        x = x.reshape(-1, 256)
        # print('x0:', x[0,:5])
        # print('x1:', x[1,:5])
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.act1(x)
        # print('x0:', x[0,:5])
        # print('x1:', x[1,:5])
        for layer in self.texture_mlp:
            x = layer(x)
        # print('x0:', x[0,:5])
        # print('x1:', x[1,:5])
        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)
        sigma = torch.exp(0.5 * log_var)
        # print('mean0:', mean[0,:5])
        # print('mean1:', mean[1,:5])
        # print('sigma0:', sigma[0,:5])
        # print('sigma1:', sigma[1,:5])
        eps = torch.randn_like(mean).to(x.device)
        z_0 = mean + sigma * eps
        # for layer in self.affine_transformation_layers:
        #     x = layer(x)
        z = self.cross_att(z_0,target)
        z = self.norm3(z)
        
        x = self.out(z)
        # x = self.out(z)
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        
        return x, kl_loss
        
        
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
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        
        self.layer_norm = PixelNorm()
        self.att_var_smoothing = Att_variance_smoothing(hidden_size)
        self.norm = PixelNorm()
        self.out = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1)

    def forward(self, input_tensor, target):
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
        
        hidden_states = torch.mul(context_layer, target)
        
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        
        hidden_states = self.att_var_smoothing(hidden_states, target)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.out(hidden_states)
        
        return hidden_states
    

class GaussianBlurConv(nn.Module):
    def __init__(self, in_channels):
        super(GaussianBlurConv, self).__init__()
        self.kernel_size = 5
        # sigma = 0.5
        # Create a 1D Gaussian kernel
        # self.kernel = self.create_gaussian_kernel(kernel_size, sigma)
        self.std_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.LayerNorm(in_channels // 2),
            nn.Linear(in_channels // 2, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.padding = self.kernel_size // 2

    def create_gaussian_kernel(self, kernel_size, sigma):
        sigma = sigma + 1e-10  # Avoid division by zero
        x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32).to(sigma.device)
        kernels = []
        for s in sigma:
            kernel = torch.exp(-0.5 * (x / s).pow(2))  # 对每个 batch 的 sigma 计算高斯核
            kernel = kernel / kernel.sum()  # 归一化高斯核
            kernels.append(kernel)
        kernels = torch.stack(kernels)
        kernels = kernels.unsqueeze(1)
        return kernels

    def forward(self, x):
        batch_size, seq_len = x.size()
        sigma = self.std_mlp(x)
        kernels = self.create_gaussian_kernel(self.kernel_size, sigma).to(x.device)
        x = x.view(1, batch_size, seq_len)  # Add channel dimension
        x = F.conv1d(x, kernels, padding=self.padding, groups=batch_size)
        x = x.view(batch_size, seq_len)  # Remove channel dimension
        return x

class Att_variance_smoothing(nn.Module):
    def __init__(self, style_dim):
        super(Att_variance_smoothing, self).__init__()
        self.gaussion_conv = GaussianBlurConv(style_dim)
        self.lambda_guide = nn.Parameter(torch.tensor(5.0), requires_grad=True)
        # self.lambda_target = torch.tensor(1.0)
        
    def forward(self, x, target):
        x = self.gaussion_conv(x)
        out = target + self.lambda_guide * (x - target)
        return out
    
class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(residual_block, self).__init__()
        self.layer1 = nn.Linear(in_channels, in_channels // 2)
        self.act1 = nn.ReLU()
        self.norm1 = nn.LayerNorm(in_channels // 2)
        self.layer2 = nn.Linear(in_channels // 2, in_channels // 2)
        self.act2 = nn.ReLU()
        self.norm2 = nn.LayerNorm(in_channels // 2)
        self.layer3 = nn.Linear(in_channels // 2, in_channels)
        self.act3 = nn.ReLU()
        self.norm3 = nn.LayerNorm(in_channels)
        self.fc = nn.Linear(in_channels, out_channels)
        self.lrelu = nn.LeakyReLU(0.2)
        self.norm4 = nn.LayerNorm(out_channels)
        
    def forward(self, x):
        identity = x
        x = self.layer1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        x = self.layer2(x)
        x = self.norm2(x)
        x = self.act2(x)
        
        x = self.layer3(x)
        x = self.norm3(x)
        x = self.act3(x)
        
        x = self.fc(identity) + x
        x = self.lrelu(x)
        x = self.norm4(x)
        return x
    

class PixelNorm(nn.Module):
    def __init__(self, num_channels=None):
        super().__init__()
        # num_channels is only used to match function signature with other normalization layers
        # it has no actual use

    def forward(self, input):
        norm = torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-5)
        return input / norm
    