class DataStream:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.images_per_batch = batch_size
        self.current_batch_num = 0

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
