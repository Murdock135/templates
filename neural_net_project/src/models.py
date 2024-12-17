import torch.nn as nn

class CNN(nn.Module):
    
    # this is our CNN initilization function     
    def __init__(self, size, num_classes):
        super(CNN, self).__init__()

        # here is our "feature extraction" via convolutional layers 
        #   note: assume we got a single channel (grayscale) MNIST image of size 28x28x1
        #         first layer
        #             28x28x1 -> convolution (1 stride, 1 padd, 3x3 kernel, 2 kernels) -> 28x28x2
        #         pooling
        #             pool of 2x2 => 28x28 / 2 => 14x14 now (technically, 14x14x2 right!)
        #         second layer
        #             14x14x2 -> convolution (1 stride, 1 padd, 3x3 kernel, 4 kernels) -> 14x14x4
        #         pooling
        #             pool of 2x2 => 14x14 / 2 => 7x7 now (technically, 7x7x4 right!)
        self.extract = nn.Sequential( # lets make a 2D convolution layer
                                      nn.Conv2d( in_channels = size, out_channels = 2, 
                                                 kernel_size = 3, stride = 1, padding = 1), 
                                                     # in_channels = 1 for MNIST and 3 for RGB image
                                                     # out_channels = 2 means 2 shared weights/features
                                                     # kernel_size = 3 means a 3x3 size kernel
                                                     # stride = 1 means move one pixel at a time in each dim
                                                     # padding = adds one pixel of zeros to each side of each dim
                                                     #           note, thats what keeps our spatial dims the same for a 3x3 kernel
                                                     #           it also lets us process each location, even that border!!!
                                      # its a NN, lets run a non-linearity on each of those results!
                                      nn.ReLU(inplace = True),
                                                     # could also use torch.nn.Sigmoid or etc.
                                                     # inplace means don't have to return a result, do it on the data
                                      # ----------------------------------------------------------- 
                                      # !!! hey, we just made a layer of convolution/nonlin !!!
                                      # ----------------------------------------------------------- 
                                      # lets pool using a 2x2 region that is not overlapping
                                      nn.MaxPool2d(2),                                                  
                                      # lets do dropout with a small percentage/rate               
                                      nn.Dropout(0.1),
                                      # ----------------------------------------------------------- 
                                      # now, lets make another layer of convolution, pooling, and drop out
                                      nn.Conv2d( in_channels = 2, out_channels = 4, 
                                                 kernel_size = 3, stride = 1, padding = 1),
                                                 # in_channels here needs to match out_channels above
                                                 # lets use 5 filters 
                                      nn.ReLU(inplace = True),
                                      nn.MaxPool2d(2),
                                      nn.Dropout(0.1), )

        # ok, now we are going to make a simple MLP classifier on the end of our above features
        self.decimate = nn.Sequential( nn.Linear(4*(7*7), 12),  
                                            # take our 4 filters whose response fields are 7x7 to 12 neurons
                                       nn.ReLU(inplace = True), # run a nonlinearity
                                       nn.Dropout(0.2), # some drop out
                                       nn.Linear(12, num_classes) ) # map the 32 down to our number of output classes
 
    #----------------------------
    # Model: Invoke Forward Pass
    #----------------------------

    def forward(self, x):

        features = self.extract(x) # easy, pass input (x) to our "feature extraction" above
        features = features.view(features.size()[0], -1) # now, flatten 7x7x4 matrix to 1D array of 7*7*4 size
        myresult = self.decimate(features) # pass that to our MLP classifier, and done!!!

        return myresult