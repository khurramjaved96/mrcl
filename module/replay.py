import random
import torch

class ReservoirSampler:
    def __init__(self, mem_sz):
        self.buffer_img = []
        self.buffer_y = []
        head = 0
        self.mem_sz = mem_sz
        self.n = 0

    def update_buffer(self, B):
        j = 0
        for x, y in B:
            M = len(self.buffer_img)
            if M < self.mem_sz:
                self.buffer_img.append(x)
                self.buffer_y.append(y)
            else:
                i = random.randint(0, self.n + j)
                if i < self.mem_sz:
                    self.buffer_img[i] = x
                    self.buffer_y[i] = y
            j = j + 1

    def update_observations(self, o):
        self.n += o

    def sample_buffer(self, batch_size):
        batch_size = min(batch_size, len(self.buffer_img))
        x = random.sample(list(range(len(self.buffer_img))), batch_size)
        # print(x)
        # quit()
        return torch.stack(self.buffer_img)[x], torch.stack(self.buffer_y)[x]
