from torch.optim.lr_scheduler import _LRScheduler

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        self.multiplier = multiplier            # 学习率倍数
        self.warm_epoch = warm_epoch            # 预热阶段的epoch数，表示从初始学习率上升到目标学习率所需的epoch轮数
        self.after_scheduler = after_scheduler  # 预热阶段结束后，接管学习率计算的调度器，通常是余弦退火策略
        self.finished = False                   # 预热阶段结束标志，当其被置为True，表示预热阶段结束，后续执行after_scheduler的学习率计算逻辑
        self.last_epoch = None                  # 显式设置last_epoch为None，实际上该成员变量由父类定义，并在调用step()方法时自动更新
        self.base_lrs = None                    # 显式设置base_lrs为None，实际上该成员变量由父类定义，并在调用__init__()方法初始化为optimizer.param_groups中的lr值
        super().__init__(optimizer)

    # 重写父类方法，定义学习率计算逻辑（如余弦衰减、指数衰减等）
    def get_lr(self):
        # 预热结束后，执行after_scheduler的学习率计算逻辑
        if self.last_epoch > self.warm_epoch:
            if self.after_scheduler:
                # 预热阶段刚结束的时候，设置after_scheduler的 base_lrs *= multiplier，并置预热结束标志位为True
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        
        # 预热阶段，计算当前epoch的学习率，按照当前epoch与总预热epoch的比例线性增长
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warm_epoch + 1.) for base_lr in self.base_lrs]

    # 每轮训练结束后调用，更新last_epoch并触发get_lr()计算新学习率
    # epoch默认无需传入一直保持None，内部会自动更新last_epoch，第一次调用时last_epoch会从-1变为0，后续每次调用，last_epoch会加1
    def step(self, epoch=None, metrics=None):
        # 预热阶段结束后，由after_scheduler接管step()方法
        if self.finished and self.after_scheduler:
            if epoch is None:
                # 由父类维护epoch计数器，传入None即可，fter_scheduler的epoch预热结束后从1开始
                self.after_scheduler.step(None)
            else:
                # 如果自己维护epoch计数器，after_scheduler的epoch预热结束后从0开始
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            # 预热阶段，调用父类的step方法（当前类step()已被重写不能再调用，会出现无限递归）
            # 由于传入了当前类和当前类self指针，会优先调用重写后的get_lr()方法和维护self的成员变量
            return super(GradualWarmupScheduler, self).step(epoch)