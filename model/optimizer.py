import torch.optim

def get_optimizer(config, network):
    if config.optimizer_name == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=config.lr)
    elif config.optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, network.parameters()), lr=config.lr, momentum=config.momentum, weight_decay=config.wd) 
    else:
        print("Error! Unknown optimizer name: ", config.optimizer_name)
        assert 0

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= config.lr_epoch_step , gamma= config.wd)
    
    return optimizer, scheduler