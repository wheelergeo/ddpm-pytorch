import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def extract(v, t, x_shape):
    r"""
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.

    Args:
        v (torch.Tensor): hyperparameter, and it is usually alpha and beta in ddpm.
        t (torch.Tensor): one-dimensional vector, sampled uniformly at random time steps.
        x_shape (torch.Size): Used to ensure that the number of dimensions of the output is the same as that of the input x,
            and batch_size is equal to x's batch_size
    """

    # t 是一个batch_size大小的一维向量，每个分量为随机选取的时间步 t
    device = t.device
    # t 作为索引，从 v 中的第 0 维取出对应位置 t 的分量，并保存到和 t 相同设备中
    out = torch.gather(v, index=t, dim=0).float().to(device)
    # list 加法，[0] + [1] = [0, 1]
    # list 乘法，[0] * 3 = [0, 0, 0]
    # 这里 out 保留 x 的batch_size维度，其余维度置为1，且维度数目扩展至和 x 同数量
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class GaussianDiffusionTrainer(nn.Module):
    r"""
    Gaussian Diffusion Trainer.
    
    The :class:`~Diffusion.GaussianDiffusionTrainer` class implements the training
    of a Gaussian diffusion model as described in Algorithm 1 of the paper
    "Denoising Diffusion Probabilistic Models" (Ho et al., 2020).

    Args:
        model (nn.Module): The neural network model to be trained.
        beta_1 (float): The initial beta value for the diffusion process.
        beta_T (float): The final beta value for the diffusion process.
        T (int): The number of diffusion steps.

    Attributes:
        model (nn.Module): The neural network model to be trained.
        T (int): The number of diffusion steps.
        betas (torch.Tensor): A tensor containing the beta values for each diffusion step.
        sqrt_alphas_bar (torch.Tensor): The square root of the cumulative product of (1 - beta) values.
        sqrt_one_minus_alphas_bar (torch.Tensor): The square root of the cumulative product of beta values.

    Example:
        >>> model = MyModel()
        >>> diffusion_trainer = GaussianDiffusionTrainer(model, beta_1=1e-4, beta_T=0.02, T=1000)
        >>> x_0 = torch.randn(16, 3, 32, 32)  # Example input
        >>> loss = diffusion_trainer(x_0)
        >>> loss.backward()  # Backpropagate the loss
    """
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        # register_buffer创建的变量，会跟随模型创建和销毁，被保存在模型的state_dict中，
        # 但不会被视为模型的参数，不会参与梯度计算，在load_state_dict会被一起更新进模型
        # linspace创建一个一维张量，包含从beta_1到beta_T的T个等间距的值
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        # cumprod 计算张量 alphas 第 0 维的连乘积
        # alphas_bar 是一个一维向量，每个分量是对应时间步 t 的连乘积
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion equation q(x_t | x_{t-1})
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        # 生成一个长为batch_size的随机整数张量，范围在[0, T)之间
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        # 生成与x_0形状相同的高斯噪声张量
        noise = torch.randn_like(x_0)
        # 为每一个样本的时间步 t 计算 x_t，计算得到的每个 x_t 组成新的batch维度
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )
        # 这里的 noise 不依赖外部输入，因此可以将 loss 放在 forward 中，函数返回 loss 值
        # 为了确保 loss 的通用性，这里不作 mean操作（reduction=none），返回分量维度的损失
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss

class GaussianDiffusionSampler(nn.Module):
    r"""
    Gaussian Diffusion Sampler.
    The :class:`~Diffusion.GaussianDiffusionSampler` class implements the sampling
    process of a Gaussian diffusion model as described in Algorithm 2 of the paper
    "Denoising Diffusion Probabilistic Models" (Ho et al., 2020).
    
    Args:
        model (nn.Module): The neural network model to be used for sampling.
        beta_1 (float): The initial beta value for the diffusion process.
        beta_T (float): The final beta value for the diffusion process.
        T (int): The number of diffusion steps.
    Attributes:
        model (nn.Module): The neural network model to be used for sampling.
        T (int): The number of diffusion steps.
        betas (torch.Tensor): A tensor containing the beta values for each diffusion step.
        coeff1 (torch.Tensor): Coefficients for the mean prediction.
        coeff2 (torch.Tensor): Coefficients for the variance prediction.
        posterior_var (torch.Tensor): Posterior variance for the diffusion process.
    Example:
        >>> model = MyModel()
        >>> diffusion_sampler = GaussianDiffusionSampler(model, beta_1=1e-4, beta_T=0.02, T=1000)
        >>> x_T = torch.randn(16, 3, 32, 32)  # Example input
        >>> x_0 = diffusion_sampler(x_T)  # Sampled output
    """
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # pad = [最左侧填充个数, 最右侧填充个数]
        # value = 填充的数值
        # [:T]取前T个元素
        # 这样做的目的是方便在采样或推断时，获得每个时间步 t-1 的前一个时间步 t 对应的 alphas_bar 值
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    # 这里的 t 是一个 batch_size 大小，数值全部相同（为当前时间步）的一维向量
    def p_mean_variance(self, x_t, t):
        # posterior_var[1:2]取得是包含一个元素的一维张量，而posterior_var[1]取得是标量
        # 当t=0是，避免方差为 0，var=posterior_var[1]
        # 降噪添加随机噪声的方差为[beta_1*(1-alphas_bar_2)/(1-alpha_bar_1) beta_1 beta_2 ... beta_{T-1}]
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        assert x_t.shape == eps.shape
        mean = (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

        return mean, var

    def forward(self, x_T: torch.Tensor):
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            # x_T.shape[0]表示batch_size
            # 创建一个和 x_t 同类型的，同设备的，长度为batch_size的，数值全为 1 的一维向量
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                # 最后一步确保输出的稳定性（干净的图片），不加噪声，直接取均值作为输出
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            # isnan()，判断 x_t 每个分量是否为空，为空则该位置 False，否则 True，生成和 x_t 同型 boolean 张量
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        # 将 x_0 所有元素值限制在[-1, 1]之间
        # 在扩散模型训练和生成时，通常会把真实图像像素归一化到[-1, 1]区间，原始像素[0, 255]归一化到[-1, 1]
        return torch.clip(x_0, -1, 1)