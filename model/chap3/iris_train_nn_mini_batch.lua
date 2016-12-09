--[[
-- Training
-- by socurites
--]]

-- Load data
local datasets = require 'model/chap1/iris_load_data'
local trainSet = datasets[1]
local testSet = datasets[2]

---------------------------------------------------------------------

-- Initialise the network
require "nn"
---- # of input featrues
local inputs = 4
---- # of output classes
local outputs = 3 
---- # of nodes in the first hidden layer
local hidden_1 = 10

---- Configure model network
local model = nn.Sequential();
model:add(nn.Linear(inputs, hidden_1))
model:add(nn.Tanh())
model:add(nn.Linear(hidden_1, outputs))
model:add(nn.LogSoftMax())

---------------------------------------------------------------------

-- Define loss function
local loss = nn.ClassNLLCriterion()  

---------------------------------------------------------------------

-- Train the network
local theta, gradTheta = model:getParameters()
local optimState = {learningRate = 0.15}
local batchSize = 50

maxIteration = 25
for epoch = 1, maxIteration do
    for i = 1, trainSet.data:size(1), batchSize do
        gradTheta:zero()
        for j = 0, batchSize - 1 do
            if i + j > trainSet.data:size(1) then break end
            h_x = model:forward(trainSet.data[i + j])
            J = loss:forward(h_x, trainSet.label[i + j])
            dJ_dh_x = loss:backward(h_x, trainSet.label[i + j])
            -- Computes and updates gradTheta
            model:backward(trainSet.data[i + j], dJ_dh_x)
        end
        model:updateParameters(optimState.learningRate)
    end

    -- For debugging only
    h_x = model:forward(trainSet.data)
    totalJ = loss:forward(h_x, trainSet.label)
    print(string.format("current loss: %.5f", totalJ))
end

---------------------------------------------------------------------

-- Evaluate trained model
local correct = 0
for i=1, testSet.data:size(1) do
    local groundtruth = testSet.label[i]
    local prediction = model:forward(testSet.data[i])
    local confidences, indices = torch.sort(prediction, true)
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end
print(string.format("Evaluation: %.2f%s", 100 * correct / testSet.data:size(1), "% correct"))
