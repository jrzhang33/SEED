def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError(f"Dataset not found: {dataset_name}")
    return globals()[dataset_name]


class emg:
    def __init__(self):
        self.subject = 10
        self.sequence_len = 200
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        self.input_channels = 8
        self.kernel_size = 9
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        self.input_shape = (self.input_channels, self.sequence_len)

class pamap:
    def __init__(self):

        self.sequence_len = 512 
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        self.input_channels = 27
        self.kernel_size = 9
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 8

        self.input_shape = (self.input_channels, self.sequence_len)

class dsads:
    def __init__(self):
        self.sequence_len = 125
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        self.input_channels = 45
        self.kernel_size = 9
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 19

        self.input_shape = (self.input_channels, self.sequence_len)

class uschad:
    def __init__(self):
        self.subject = 14

        self.sequence_len = 500
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        self.input_channels = 6
        self.kernel_size = 6
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 12

        self.input_shape = (self.input_channels, self.sequence_len)
