import torch
import torch.nn as nn

class SpatialNet(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.first = nn.Linear([2, 5, 200, 200], [50, 5])

    def forward(self, TxINT, RxINT, powers):
        return self.first(torch.cat([TxINT.unsqueeze(0), RxINT.unsqueeze(0)], dim=0))