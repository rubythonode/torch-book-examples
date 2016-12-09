--[[
-- Training
-- by socurites
--]]

-- Load data file
require 'paths'

trainSet = torch.load(paths.concat('data/cifar-10/', 'cifar10-train.t7'))
testSet = torch.load(paths.concat('data/cifar-10/', 'cifar10-test.t7'))
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

setmetatable(trainSet, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);
trainSet.data = trainSet.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainSet:size() 
    return self.data:size(1) 
end


mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainSet.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainSet.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainSet.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainSet.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

testSet.data = testSet.data:double()   -- convert from Byte tensor to Double tensor
for i=1,3 do -- over each image channel
    testSet.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testSet.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

--------------------------------------------------------------------

-- Initialize the network
require 'nn'
model = nn.Sequential()
model:add(nn.SpatialConvolution(3, 6, 5, 5)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
model:add(nn.ReLU())                       -- non-linearity 
model:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
model:add(nn.SpatialConvolution(6, 16, 5, 5))
model:add(nn.ReLU())                       -- non-linearity 
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
model:add(nn.Linear(16*5*5, 120))             -- fully connected layer (matrix multiplication between input and weights)
model:add(nn.ReLU())                       -- non-linearity 
model:add(nn.Linear(120, 84))
model:add(nn.ReLU())                       -- non-linearity 
model:add(nn.Linear(84, 10))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
model:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems

---------------------------------------------------------------------

-- Define loss function
loss = nn.ClassNLLCriterion()


-- Train the network
theta, gradTheta = model:getParameters()
optimState = {learningRate = 0.01}

maxIteration = 5
for epoch = 1, maxIteration do
    for i = 1, trainSet.data:size(1) do
        if (i % 1000) == 0 then print(i) end
        gradTheta:zero()
        h_x = model:forward(trainSet.data[i])
        J = loss:forward(h_x, trainSet.label[i])
        dJ_dh_x = loss:backward(h_x, trainSet.label[i])
        -- Computes and updates gradTheta
        model:backward(trainSet.data[i], dJ_dh_x)
        model:updateParameters(optimState.learningRate)
    end
    -- For debugging only
    h_x = model:forward(trainSet.data)
    totalJ = loss:forward(h_x, trainSet.label)
    print(string.format("current loss: %.5f", totalJ))
end
---------------------------------------------------------------------

-- Evaluate trained model
correct = 0
for i=1,10000 do
    local groundtruth = testSet.label[i]
    local prediction = model:forward(testSet.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end
print(string.format("Evaluation: %.5f%s", 100 * correct / testSet.data:size(1), "% correct"))



