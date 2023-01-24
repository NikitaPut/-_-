import torch

def make_optimizer(cfg, lr:float, model:torch.nn.Module):
    optimizer = None

    if cfg.SOLVER.TYPE == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) # TODO: params required to train ?

    return optimizer

def make_lr_scheduler(cfg, optimizer):
    lambda_ = lambda epoch: cfg.SOLVER.LR_LAMBDA ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_)

    return scheduler