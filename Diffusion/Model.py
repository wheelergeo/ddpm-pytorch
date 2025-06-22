import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

class Swish(nn.Module):
    r"""
    Swish activation function.

    Swish激活函数结合了ReLU的稀疏性和Sigmoid的平滑性，能够更好地捕捉非线性特征。
    
    math equation:
        Swish(x) = x * sigmoid(x)
        
    Args:
        None
    """
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    r"""
    Time embedding for diffusion models.

    math eqaution:
        p_{i, 2j} = sin(t_i / 10000^{2j/d_model})
        p_{i, 2j+1} = cos(t_i / 10000^{(2j+1)/d_model})

    Args:
        T (int): Number of timesteps.
        d_model (int): Input feature dimension.
        dim (int): Output dimension of the time embedding.
    """
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        # shape 1 × d_model/2
        emb = torch.exp(-emb).reshape(1, -1)
        # shape T × 1
        pos = torch.arange(T).float().reshape(T, -1)
        # shape T × d_model/2
        emb = pos * emb
        assert list(emb.shape) == [T, d_model // 2]
        # torch.stack就是在dim维度上插入新的维度，并将输入的张量沿着这个新维度进行拼接
        # -1即在第 2 维插入新维度
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        # view或是reshape函数优先从最低维降维，这里优先去除第2维
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),  # 嵌入层利用时间编码权重，无需训练
            nn.Linear(d_model, dim),            # 经过线性层转为目标时间向量的维度
            Swish(),                            # Swish(x) = x * sigmoid(x)，兼具ReLU的稀疏性和Sigmoid的平滑性
            nn.Linear(dim, dim),                # 再经过一层线性变换
        )
        self.initialize()

    def initialize(self):
        # self.modules()会访问模型中的所有子模块
        # 这里的子模块包括nn.Embedding和nn.Linear
        # nn.Embedding的权重是预训练的，所以不需要初始化
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb

class DownSample(nn.Module):
    r"""
    Downsample block for UNet.

    the number of channels remains the same, but the spatial dimensions are reduced.
    From the spatial information (edges, textures) to semantic information 
    (organ contours, object categories), forming a global understanding of the image content.
    
    通道数不变，尺寸减小
    从空间信息（边缘、纹理）逐步过渡到语义信息（器官轮廓、物体类别），形成对图像内容的全局理解

    Args:
        in_ch (int): Input channel size.

    """
    def __init__(self, in_ch):
        super().__init__()
        # 尺寸减小，通道数不变
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x

class UpSample(nn.Module):
    r"""
    Upsample block for UNet.
    
    the number of channels remains the same, but the spatial dimensions are increased.
    Uses skip connections to concatenate features from the corresponding downsample block.
    Implement pixel-level localization by mapping semantic information back to the original 
    image space coordinates.

    通道数不变，图片尺寸放大
    通过上采样，逐步恢复图像的空间分辨率
    通过跳跃连接（skip connection）将编码器的低级特征与解码器的高级特征拼接，实现语义与细节互补
    实现​​像素级定位​​，将语义信息映射回原图空间坐标
    
    Args:
        in_ch (int): Input channel size.
    """
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        # _, _, H, W = x.shape
        # 使用nearest邻近插值法进行上采样，直接复制近邻像素值
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x

class AttnBlock(nn.Module):
    r"""
    Attention block for UNet.
    
    This block focuses on the most relevant features in the input feature map by using
    self-attention mechanism, And the attention weights are computed using
    the scaled dot-product attention. The input feature map is normalized using GroupNorm,
    and the output is added back to the input feature map like a residual connection.

    Args:
        in_ch (int): Input channel size.
    """
    def __init__(self, in_ch):
        super().__init__()
        # 对输入特征图做归一化，提升训练稳定性
        self.group_norm = nn.GroupNorm(32, in_ch)
        # 1×1卷积层等同于全连接层，自注意力机制用于计算q, k, v三个向量
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        # AttnBlock运用了类似残差连接的机制，将注意力模块的输出与输入特征图相加
        # 这里的gain参数控制初始化的方差，1e-5是一个很小的值
        # 通常残差块最后一层的输出幅度希望更小，以便残差连接时主分支和shortcut分支的输出幅度相近
        # 防止训练初期主分支梯度过大导致模型不稳定
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        # Batch size, Channels, Height, Width
        B, C, H, W = x.shape
        h = self.group_norm(x)
        # 通过自注意力机制获得query，key，value三个向量
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        # permute维度重排，将通道维度移到最后，方便view将特征图展平为二维矩阵，方便后续计算
        # 图像任务依赖精确的位置关系，为了直接计算感受野间（相邻像素）的相关性，将图像空间维度视作序列维度
        # 而通道维度已通过卷积编码了高级语义特征，因此单独作为特征维度
        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        # 合并空间维度 H*W 为序列维度，并不改变顺序，以便做矩阵乘法
        k = k.view(B, C, H * W)
        # a(Q, K) = QK^T/\sqrt{C}
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        # 对每个batch的每一行进行softmax归一化，得到注意力权重
        w = F.softmax(w, dim=-1)

        # 同理对v进行维度变换，将通道维度移到最后，以便做矩阵乘法
        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        # f = softmax(QK^T/\sqrt{C})V
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        # 将注意力模块的输出还原成卷积层的形状，以便能嵌入任意CNN层
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        # 将注意力模块的输出与输入特征图相加，模拟残差连接
        return x + h

class ResBlock(nn.Module):
    r"""
    Residual block with GroupNorm, Swish activation, and optional attention.

    U-Net 

    Args:
        in_ch (int): Input channel size.   
        out_ch (int): Output channel size.
        t_dim (int): Dimension of the time embedding.
        dropout (float): Dropout rate.
        attn (bool): Whether to apply attention in this block.
    """
    def __init__(self, in_ch, out_ch, t_dim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            # Group Normalization对单张图像的通道进行分组，在每个组内单独计算均值和方差，并对特征进行归一化
            # 适用于小批量数据，尤其是图像数据，避免了 BN 方法对批量大小的依赖
            nn.GroupNorm(num_groups=32, num_channels=in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            # 对位置编码做一次非线性变换，提升模型的表达能力，让编码信息更丰富
            Swish(),
            # 对其特征图通道数，以便和特征图进行相加，同时赋予位置编码可训练性
            nn.Linear(t_dim, out_ch),
        )
        self.block2 = nn.Sequential(
            # 对上一个卷积层的输出进行归一化和激活
            nn.GroupNorm(32, out_ch),
            Swish(),
            # 随机丢弃部分神经元，防止过拟合，增强模型的泛化能力
            nn.Dropout(dropout),
            # 卷积层，保持特征图的通道数不变，尺寸不变
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        
        if in_ch != out_ch:
            # 如果输入和输出通道数不一致，则需要一个1x1卷积层进行通道数匹配，通道数由filter数决定
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            # 如果输入和输出通道数相同，则使用nn.Identity()，表示不做任何操作
            self.shortcut = nn.Identity()

        if attn:
            # 如果需要注意力机制，则添加注意力块
            # 注意力机制可以帮助模型关注图像中的重要区域，增强特征表达
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        # GN 层的权重和偏置不需要初始化，内部默认初始化权重为1，偏置为0
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        # 对最后一个卷积层的权重进行初始化，使用Xavier均匀分布初始化
        # gain参数控制初始化的方差，1e-5是一个很小的值
        # 通常残差块最后一层的输出幅度希望更小，以便残差连接时主分支和shortcut分支的输出幅度相近
        # 防止训练初期主分支梯度过大导致模型不稳定
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h

class UNet(nn.Module):
    r"""UNet model for diffusion models.

    Args:
        T (int): Number of diffusion steps.
        ch (int): 确定了 UNet 第一层卷积操作输出的特征图（feature map）的通道数量，下采样通道数按照ch_mult 中的倍增因子进行调整。
        ch_mult (list): Channel multiplier for each downsample block, list numbers indicating how many downsample blocks are.
        attn (list): Indices of downsample blocks where attention is applied.
        num_res_blocks (int): Number of residual blocks in each downsample block.
        dropout (float): Dropout rate.
    """
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        # all()当列表中所有元素都为True时返回True
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        t_dim = ch * 4
        # Time embedding在UNet的残差块或特征层中作为额外的条件输入
        # 使模型在每一步去噪过程中都能感知当前时间步，从而确定去噪策略
        self.time_embedding = TimeEmbedding(T, ch, dim=t_dim)

        # 0. head
        # head是UNet的输入层，负责将输入图像转换为特征
        # UNet的默认输入通道数为rgb：3，输出通道数为ch
        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)

        # 1. downsample blocks
        # downsample blocks是UNet的下采样层，负责将输入图像转换为特征图
        self.downblocks = nn.ModuleList()
        # record output channel when dowmsample for upsample
        chs = [ch]
        current_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(
                    # 第一个残差块对通道数翻倍，高宽不变
                    # 后续残差块保持通道数不变，高宽不变
                    ResBlock(
                        in_ch=current_ch, 
                        out_ch=out_ch,
                        # 主流实现中都仅在残差块中使用位置编码
                        # 因为上采样和下采样主要负责特征提取，而不涉及依据位置条件进行建模的操作
                        t_dim=t_dim,
                        dropout=dropout,
                        # 根据attn列表判断是否在当前层使用注意力机制，in关键字若满足，则返回True
                        attn=(i in attn)
                    )
                )
                current_ch = out_ch
                chs.append(current_ch)
            # if i is before the last element in ch_mult, add downsample layer
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(current_ch))
                chs.append(current_ch)

        # 2. middle blocks
        self.middleblocks = nn.ModuleList([
            ResBlock(current_ch, current_ch, t_dim, dropout, attn=True),
            ResBlock(current_ch, current_ch, t_dim, dropout, attn=False),
        ])

        # 3. upsample blocks
        # upsample blocks是UNet的上采样层，负责将特征图转换为输出图像
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            # 跳跃连接传递的浅层特征含噪声（如背景干扰），需额外卷积层净化
            # 多一个残差块可强化轮廓学习
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(
                    ResBlock(
                        in_ch=chs.pop() + current_ch, 
                        out_ch=out_ch, 
                        t_dim=t_dim,
                        dropout=dropout, 
                        attn=(i in attn)
                    )
                )
                current_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(current_ch))
        # After all upsample blocks, length of chs should be 0
        # This is to ensure that all downsampled features are used in the upsample blocks
        assert len(chs) == 0

        # 4. tail
        # tail是UNet的输出层，负责将特征图转换为输出图像
        self.tail = nn.Sequential(
            nn.GroupNorm(32, current_ch),
            Swish(),
            nn.Conv2d(current_ch, 3, kernel_size=3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            # 位置编码仅在残差块中使用
            h = layer(h, temb)
            # 记录下采样后的特征图，用于上采样时的跳跃连接
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                # skip connection
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        # 确保所有下采样的特征图都被使用
        assert len(hs) == 0
        return h

if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)
    print(y.shape)