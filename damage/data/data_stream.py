from math import ceil


class DataStream:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.images_per_batch = batch_size
        self.current_batch_num = 0
        self.num_batches = ceil(self.images_per_batch / batch_size)

    def get_generator(self):
        while True:
            for i in range(self.num_batches):
                batch = (
                    self.x[(i*self.images_per_batch): ((i+1)*self.images_per_batch)],
                    self.y[(i*self.images_per_batch): ((i+1)*self.images_per_batch)]
                )
                yield batch

    def get_batch(self):
        x_batch = self.x[(self.current_batch_num*self.images_per_batch):
                         ((self.current_batch_num+1)*self.images_per_batch)]
        y_batch = self.y[(self.current_batch_num*self.images_per_batch):
                         ((self.current_batch_num+1)*self.images_per_batch)]
        return x_batch, y_batch

    def update(self):
        self.current_batch_num += 1

    def restart(self):
        self.current_batch_num = 0
