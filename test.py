
from collections import namedtuple
import random


transition = namedtuple('transition', 'input, label')

class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, *args):
        # Append when the buffer is not full but overwrite when the buffer is full
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*args))
        else:
            self.buffer[self.location] = transition(*args)

        # Increment the buffer location
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def sample_meta(self, batch_size, adapatation):
        left = self.sample(batch_size)
        right = self.sample_trajectory(adapatation)
        return left + right
    def sample_trajectory(self, batch_size):

        if self.location - batch_size < 0:
            left = self.buffer[(self.location-batch_size) % len(self.buffer): len(self.buffer)]
            right = self.buffer[0: batch_size - (len(self.buffer) - (self.location-batch_size) % len(self.buffer))]
            return left + right
        else:
            return self.buffer[ self.location-batch_size: self.location]

    def sample_trajectory_random(self, batch_size):
        start_index = random.randint(batch_size, self.location)

        if start_index - batch_size < 0:

            left = self.buffer[(start_index-batch_size) % len(self.buffer): len(self.buffer)]
            right = self.buffer[0: batch_size - (len(self.buffer) - (start_index-batch_size) % len(self.buffer))]
            return left + right
        else:
            return self.buffer[ start_index-batch_size: start_index]


buffer = replay_buffer(2000)

for a in range(0, 5000):
    buffer.add(a, a)
    if a > 3000:
        temp = [x[0] for x in buffer.sample_trajectory_random(50)]
        for temp_temp in range(0, len(temp)-1):

            if not temp[temp_temp]+1 == temp[temp_temp+1]:
                print("Index = ", temp_temp)
                print(temp)
                assert(temp[temp_temp]+1 == temp[temp_temp+1])

