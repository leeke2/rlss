import torch
from torch import nn, optim, Tensor
from torch.multiprocessing import Process, Queue
import time

class exp_process(Process):
    def __init__(self, model, queue):
        super().__init__()
        self.model = model
        self.queue = queue

    def run(self):
        self.model.to('cuda:1')

        while True:
            if not self.queue.empty():
                self.model.load_state_dict(self.queue.get())

            print(f'Explorer {torch.sum(self.model.weight)}')
            time.sleep(1)

class train_process(Process):
    def __init__(self, model, inference_model, queue):
        super().__init__()
        self.model = model
        self.inference_model = inference_model
        self.queue = queue

    @property
    def device(self):
        return self.model.weight.device
    

    def run(self):
        self.model.to('cuda:0')

        opt = optim.Adam(self.model.parameters(), lr=0.001)

        while True:
            loss = nn.functional.mse_loss(
                self.model(Tensor([[1,2]]).to(self.device)),
                torch.rand(1,5).to(self.device)
            )
            loss.backward()
            opt.step()

            print(f'Updated {torch.sum(self.model.weight)}')
            queue.put(self.model.state_dict())

            time.sleep(5)

if __name__ == "__main__":
    a = nn.Linear(2, 5)
    b = nn.Linear(2, 5)

    a.share_memory()
    b.load_state_dict(a.state_dict())
    b.share_memory()

    queue = Queue()

    p = exp_process(b, queue)
    p.start()

    q = train_process(a, b, queue)
    q.start()


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
