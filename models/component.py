import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class IGM(nn.Module):

    def __init__(self,
                 dim: int):
        super(IGM, self).__init__()

        # Conflict Gate
        self.w_c1 = nn.Linear(dim, dim, bias=False)
        self.w_c2 = nn.Linear(dim, dim, bias=False)
        self.b_mu_c = nn.Parameter(torch.rand(dim))
        torch.nn.init.uniform_(self.b_mu_c, a=-math.sqrt(dim), b=math.sqrt(dim))

        self.w_c3 = nn.Linear(dim, dim, bias=False)
        self.w_c4 = nn.Linear(dim, dim, bias=False)
        self.b_c = nn.Parameter(torch.rand(dim))
        torch.nn.init.uniform_(self.b_c, a=-math.sqrt(dim), b=math.sqrt(dim))

        # Refine Gate
        self.w_r1 = nn.Linear(dim, dim, bias=False)
        self.w_r2 = nn.Linear(dim, dim, bias=False)
        self.b_mu_r = nn.Parameter(torch.rand(dim))
        torch.nn.init.uniform_(self.b_mu_r, a=-math.sqrt(dim), b=math.sqrt(dim))

        self.w_r3 = nn.Linear(dim, dim, bias=False)
        self.w_r4 = nn.Linear(dim, dim, bias=False)
        self.b_r = nn.Parameter(torch.rand(dim))
        torch.nn.init.uniform_(self.b_r, a=-math.sqrt(dim), b=math.sqrt(dim))

        # Adaptive Gate
        self.w_a =  nn.Linear(dim, dim, bias=False)
        self.b_a = nn.Parameter(torch.rand(dim))
        torch.nn.init.uniform_(self.b_a, a=-math.sqrt(dim), b=math.sqrt(dim))


    def forward(self,
                input1: torch.tensor,
                input2: torch.tensor,
                input3=None,
                pooling_type = 'max',
                mask1=None,
                mask2=None,
                mask3=None):
        # MAx Pooling
        if pooling_type == 'max':
            pooled_input1, _ = torch.max(input1, dim=1)
            pooled_input2, _ = torch.max(input2, dim=1)
            if input3 is not None:
                pooled_input3, _ = torch.max(input3, dim=1)
        else:
            pooled_input1  = torch.einsum("bsh,bs,b->bh", input1, mask1.float(), #(bs,dim)
                                            1 / mask1.float().sum(dim=1) + 1e-9)
            pooled_input2  = torch.einsum("bsh,bs,b->bh", input2, mask2.float(), #(bs,dim)
                                            1 / mask2.float().sum(dim=1) + 1e-9)
            if input3 is not None:
                pooled_input3  = torch.einsum("bsh,bs,b->bh", input3, mask3.float(), #(bs,dim)
                                              1 / mask3.float().sum(dim=1) + 1e-9)
        # Conflict Gate
        mu_c = F.sigmoid(self.w_c1(pooled_input1) + self.w_c2(pooled_input2) + self.b_mu_c) #(bs,dim)
        conflict = F.tanh(self.w_c3(torch.mul(mu_c, pooled_input1)) + self.w_c4(torch.mul((1 - mu_c), pooled_input2)) + self.b_c) #(bs,dim)

        # Refine Gate
        if input3 is not None:
            mu_r = F.sigmoid(self.w_r1(pooled_input1) + self.w_r2(pooled_input3) + self.b_mu_r) #(bs,dim)
            refine = F.tanh(self.w_r3(torch.mul(mu_r, pooled_input1)) + self.w_r4(torch.mul(mu_r, pooled_input3)) + self.b_r) #(bs,dim)
        else:
            mu_r = F.sigmoid(self.w_r1(pooled_input1) + self.w_r2(pooled_input2) + self.b_mu_r) #(bs,dim)
            refine = F.tanh(self.w_r3(torch.mul(mu_r, pooled_input1)) + self.w_r4(torch.mul(mu_r, pooled_input2)) + self.b_r) #(bs,dim)

        # Adaptive Gate
        adapt = refine + torch.mul((1 - mu_r), conflict) #(bs,dim)
        interact = F.tanh(self.w_a(adapt) + self.b_a) #(bs,dim)
        interact = torch.unsqueeze(interact, 1).repeat(1, input1.shape[1], 1)
        output = torch.mul(interact, input1)
        return output