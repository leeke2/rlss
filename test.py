import torch
from torch import nn, optim, Tensor
from torch.multiprocessing import Process
import time

class exp_process(Process):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def run(self):
        self.model.cuda()
        while True:
            print(f'Explorer {torch.sum(self.model.weight)}')
            time.sleep(1)

if __name__ == "__main__":
    a = nn.Linear(2, 5)
    b = nn.Linear(2, 5)

    a.share_memory()
    b.load_state_dict(a.state_dict())
    b.share_memory()

    p = exp_process(b)
    p.start()

    a.cuda()

    opt = optim.Adam(a.parameters(), lr=0.001)

    for _ in range(10):
        loss = nn.functional.mse_loss(
    	a(Tensor([[1,2]])),
    	torch.rand(1,5)
        )
        loss.backward()
        opt.step()

        print(f'Updated {torch.sum(a.weight)}')
        b.load_state_dict(a.state_dict())
        time.sleep(5)


    # print(torch.allclose(linear_1.cpu().weight, linear_2.weight))

    # linear_1.share_memory()
    # linear_2.load_state_dict(linear_1.state_dict())
    # linear_2.share_memory()

    # print(torch.allclose(linear_1.cpu().weight, linear_2.weight))

    # opt = optim.Adam(linear_1.parameters(), lr=0.001)

    # print(Tensor([[1,2]]).cuda().device)
    # print(linear_1(Tensor([[1,2]]).cuda()).device)
    # print(torch.rand(1, 10).cuda().device)
    # loss = nn.functional.mse_loss(
    # 	linear_1(Tensor([[1,2]]).cuda()),
    # 	torch.rand(1, 10).cuda()
    # )
    # loss.backward()
    # opt.step()

    # print(torch.allclose(linear_1.cpu().weight, linear_2.weight))
