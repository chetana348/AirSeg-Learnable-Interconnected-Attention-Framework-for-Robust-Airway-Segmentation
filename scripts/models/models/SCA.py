import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, input_dim, num_heads=8, use_qkv_bias=True, dropout_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = input_dim // num_heads
        self.scale = nn.Parameter(torch.Tensor([head_dim ** -0.5]))
        self.qkv_linear = nn.Linear(input_dim, input_dim * 3, bias=use_qkv_bias)
        self.dropout = nn.Dropout(dropout_ratio)
        self.proj_linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        batch_size, seq_length, input_dim = x.shape
        qkv = self.qkv_linear(x).reshape(batch_size, seq_length, 3, self.num_heads, input_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores.softmax(dim=-1)
        attn_scores = self.dropout(attn_scores)
        x = (attn_scores @ v).transpose(1, 2).reshape(batch_size, seq_length, input_dim)
        x = self.proj_linear(x)
        x = self.dropout(x)
        return x

class Embedding(torch.nn.Module):
    def __init__(self, channels=None, epsilon=1e-4):
        super(Embedding, self).__init__()

        self.activation = nn.Sigmoid()
        self.epsilon = epsilon

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('epsilon=%f)' % self.epsilon)
        return s

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        n = width * height - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.epsilon)) + 0.5
        return x * self.activation(y)
    
    
class Embedding_CNN(nn.Module):
    def __init__(self, in_channels):
        super(Embedding_CNN, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class LAAM(nn.Module):   #position based attention
    def __init__(self):
        super(LAAM,self).__init__()
        self.conv=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=7,padding=3)

    def forward(self,x):
        max_x = x
        for i in (0,1):
            max_x=torch.max(max_x,dim=i, keepdim=True)[0]
        avg_x=torch.mean(x,dim=(0,1),keepdim=True)
        att=torch.cat((max_x,avg_x),dim=1)
        att=self.conv(att)
        att=torch.sigmoid(att)
        return x*att
 
class CAM(nn.Module):      #semantic based attention
    def __init__(self,in_channels,reduction_ratio=16):
        super(CAM,self).__init__()
        self.linear=[]
        self.linear.append(nn.Linear(in_features=in_channels,out_features=in_channels//reduction_ratio))
        self.linear.append(nn.ReLU())
        self.linear.append(nn.Linear(in_features=in_channels//reduction_ratio,out_features=in_channels))
        self.linear=nn.Sequential(*self.linear)
    def forward(self,x):
        max_x = x
        for i in (0, 2, 3):
            max_x = torch.max(max_x, dim=i, keepdim=True)[0]
        if not False:
            max_x = max_x.squeeze()

        avg_x=torch.mean(x,dim=(0,2,3),keepdim=False).unsqueeze(0)
        max_x=self.linear(max_x)
        avg_x=self.linear(avg_x)
        att=max_x+avg_x
        att=torch.sigmoid(att).unsqueeze(-1).unsqueeze(-1)
        return x*att

class SCA(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(SCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv2(self.relu1(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu1(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class CSA(nn.Module):
    def __init__(self, kernel_size=7):
        super(CSA, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        attention_map = self.sigmoid(x)
        attention_map = attention_map.expand_as(x)
        attended_features = x * attention_map

        return attended_features

class Base(nn.Module):
    def __init__(self, in_shape, mid_shape, out_shape, upshape, ch_attn = True, sp_attn = True):  #channel or spatial
        super(Base, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(upshape, upshape, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(upshape),
            nn.ReLU(True))

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_shape, mid_shape, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_shape),
            nn.ReLU(True)
        )
        if ch_attn:
            self.att = SCA(mid_shape)
        elif sp_attn:
            self.att = CSA()
        else:
            self.att = nn.Identity()

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_shape, out_shape, kernel_size=3, padding=1),nn.BatchNorm2d(out_shape), nn.ReLU(True))

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.att(outputs) * outputs
        outputs = self.conv2(outputs)
        return outputs