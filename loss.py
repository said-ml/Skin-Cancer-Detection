from torch import nn

def Loss( predections, targets):

    #assert predections.shape == targets.shape
    return nn.BCELoss()(predections, targets)








































