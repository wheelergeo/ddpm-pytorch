import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler

def train(modelConfig: Dict):
    # setup train device
    device = torch.device(modelConfig["device"])
    # dataset
    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=False,
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
    )
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=modelConfig["batch_size"], 
        shuffle=True, 
        num_workers=4, 
        drop_last=True, 
        pin_memory=True
    )

    # model setup
    net_model = UNet(
        T=modelConfig["T"],                             # number of diffusion steps
        ch=modelConfig["channel"],                      # base conv layer channel size
        ch_mult=modelConfig["channel_mult"],            # channel multiplier for each downsample block
        attn=modelConfig["attn"],                       # attention block indices, e.g. [2] means attention at the second downsample block
        num_res_blocks=modelConfig["num_res_blocks"],   # number of residual blocks in each downsample block
        dropout=modelConfig["dropout"]                  # dropout rate
    ).to(device)
    # is it needed to load pre-trained weight
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(
            torch.load(
                os.path.join(
                    modelConfig["save_weight_dir"], 
                    modelConfig["training_load_weight"]
                ), 
                map_location=device
            )
        )
    # AdamW optimizer decoupled weight decay from the gradient update.
    optimizer = torch.optim.AdamW(
        net_model.parameters(), 
        lr=modelConfig["lr"], 
        weight_decay=1e-4
    )
    # 余弦退火策略，初始(最大)学习率为modelConfig["lr"]，经过T_max个epoch后，学习率降到eta_min，eta_min = 0表示学习率不会降到0
    # 当last_epoch = -1表示这是热重启版余弦退火策略，每个epoch后，学习率会回到初始值
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, 
        T_max=modelConfig["epoch"], 
        eta_min=0, 
        last_epoch=-1
    )
    # 优化器的自定义封装，
    # optimizer：优化器基座
    # multiplier: 学习率的倍数
    # warm_epoch: 从初始值上升到目标值所需epoch轮数
    # after_scheduler: 余弦退火策略
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, 
        multiplier=modelConfig["multiplier"], 
        warm_epoch=modelConfig["epoch"] // 10, 
        after_scheduler=cosineScheduler
    )
    # 基于UNet模型和beta参数初始化高斯扩散训练器
    trainer = GaussianDiffusionTrainer(
        net_model, 
        modelConfig["beta_1"], 
        modelConfig["beta_T"], 
        modelConfig["T"]
    ).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        # with用于管理tadm对象生命周期，创建调用__enter__()方法，返回一个迭代器对象tqdmDataLoader（开启进度条显示）
        # 结束则调用__exit__()方法，自动处理异常和资源释放（关闭进度条显示）
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, _ in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                loss = trainer(x_0).sum() / 1000.
                loss.backward()
                # 参数梯度裁剪，将参数梯度范数限制在modelConfig["grad_clip"]范围内，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), 
                    modelConfig["grad_clip"]
                )   # gradient clipping
                optimizer.step()
                # 在进度后追加显示自定义信息
                tqdmDataLoader.set_postfix(
                    ordered_dict={
                        "epoch": e,
                        "loss: ": loss.item(),
                        "img shape: ": x_0.shape,
                        "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                    }
                )
        warmUpScheduler.step()
        torch.save(
            net_model.state_dict(), 
            os.path.join(
                modelConfig["save_weight_dir"], 
                'ckpt_' + str(e) + "_.pt"
            )
        )

def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])

        model = UNet(
            T=modelConfig["T"], 
            ch=modelConfig["channel"], 
            ch_mult=modelConfig["channel_mult"], 
            attn=modelConfig["attn"],
            num_res_blocks=modelConfig["num_res_blocks"], 
            dropout=0.
        ).to(device)

        ckpt = torch.load(
            os.path.join(
                modelConfig["save_weight_dir"], 
                modelConfig["test_load_weight"]
            ), 
            map_location=device
        )
        model.load_state_dict(ckpt)
        print("model load weight done.")

        model.eval()

        sampler = GaussianDiffusionSampler(
            model, 
            modelConfig["beta_1"], 
            modelConfig["beta_T"], 
            modelConfig["T"]
        ).to(device)

        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, 32, 32], 
            device=device
        )
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(
            saveNoisy, 
            os.path.join(
                modelConfig["sampled_dir"], 
                modelConfig["sampledNoisyImgName"]
            ), 
            nrow=modelConfig["nrow"]
        )
        
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_image(
            sampledImgs, 
            os.path.join(
                modelConfig["sampled_dir"],
                modelConfig["sampledImgName"]
            ), 
            nrow=modelConfig["nrow"]
        )