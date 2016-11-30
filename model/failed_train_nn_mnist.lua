require 'nn'

-- Create network model
local model = nn.Sequential()

-- # of input features
-- 1 channel * 32 width * 32 height
local n = 1 * 32 * 32

-- # of output classes
local K = 10

-- Size of network
local s = {n, 6*28*28, 6*14*14, 16*10*10, 16*5*5, 120, K}
--local s = {n, 28*28, 14*14, 10*10, 5*5, 20, K}

-- Add layer to network
model:add(nn.Linear(s[1], s[2]))
model:add(nn.ReLU())
model:add(nn.Linear(s[2], s[3]))
model:add(nn.ReLU())
model:add(nn.Linear(s[3], s[4]))
model:add(nn.ReLU())
model:add(nn.Linear(s[4], s[5]))
model:add(nn.ReLU())
model:add(nn.Linear(s[5], s[6]))
model:add(nn.ReLU())
model:add(nn.Linear(s[6], s[7]))
model:add(nn.LogSoftMax())

-- Define a loss function
local loss = nn.ClassNLLCriterion()

-- Load train set
-- # of input instances
local m = 60000

require 'paths'
local train_set_path = 'train_32x32.t7'
local train_set = torch.load(paths.concat('data', 'mnist.t7', train_set_path), 'ascii')
local train_set_X = train_set.data:reshape(m, n):double()    -- convert the data from ByteTensor to a DoubleTensor
local train_set_Y = train_set.labels:double()


local theta, gradTheta = model:getParameters()
local optimState = {learningRate = 0.15}
local max_epoch = 15 

-- Start training
require 'optim'
for epoch = 1, max_epoch do
  function feval(theta)
    gradTheta:zero()
    local h_X = model:forward(train_set_X)
    local J = loss:forward(h_X, train_set_Y) 
    print(J)
    local dJ_dh_X = loss:backward(h_X, train_set_Y)
    model:backward(train_set_X, dJ_dh_X)    -- This computes and updates gradTheta
    return J, gradTheta
  end
  optim.sgd(feval, theta, optimState)
end

net = model
