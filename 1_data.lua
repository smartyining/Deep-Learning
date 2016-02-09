----------------------------------------------------------------------
-- This script demonstrates how to load the (MNIST) Handwritten Digit 
-- training data, and pre-process it to facilitate learning.
--
-- It's a good idea to run this script with the interactive mode:
-- $ th -i 1_data.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to analyze/visualize the data you've just loaded.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('MNIST Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-size', 'full', 'how many samples do we load: small | full')
   cmd:option('-visualize',false, 'visualize input data and weights during training')
   cmd:option('-transform',false, 'elastic transform input data')
   cmd:text()

   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
function elastic_transform(im,kernel_dim, sigma, alpha)    
    -- create an empty image
    local result = torch.Tensor(im:size()):fill(0) 

    -- create random displacement fields
    local displacement_field_x = torch.uniform(im):mul(2):add(-1):mul(alpha):float():squeeze()
    local displacement_field_y = torch.uniform(im):mul(2):add(-1):mul(alpha):float():squeeze()
    
    --create the gaussian kernel
    local kernel =  image.gaussian(kernel_dim,sigma,1,true):float()

    --convolve the fields with the gaussian kernel
    displacement_field_x = image.convolve(displacement_field_x, kernel, 'full')
    displacement_field_y = image.convolve(displacement_field_y, kernel, 'full')

    -- make the distortrd image by averaging each pixel value to the neighbouring
    -- four pixels based on displacement fields

    for row=1, im:size()[1] do
        for col=1 , im:size()[2] do

            local low_ii = row + math.floor(displacement_field_x[row][col])
            local high_ii = row + math.ceil(displacement_field_x[row][col])

            local low_jj = col + math.floor(displacement_field_y[row][col])
            local high_jj = col + math.ceil(displacement_field_y[row][col])

            if not (low_ii < 0 or low_jj < 0 or high_ii >= im:size()[1] -1 
               or high_jj >= im:size()[2] - 1 )then 

               result[row][col] = im[low_ii][low_jj]/4 + im[low_ii][high_jj]/4 + im[high_ii][low_jj]/4 + im[high_ii][high_jj]/4

            end 
        end
    end

    return result

end
---------------------------------------------------------------------
-- Random shuffle and split data set
function splitDataset(d,ratio)

   local shuffle = torch.randperm(d.size())

   local numTrain = math.floor(d.size() * ratio)
   local numTest = d.size() - numTrain

   local train = torch.Tensor(numTrain, d.data:size(2), d.data:size(3), d.data:size(4))
   local test = torch.Tensor(numTest, d.data:size(2), d.data:size(3), d.data:size(4))
   local trainLabels = torch.Tensor(numTrain)
   local testLabels = torch.Tensor(numTest)

   for i=1, numTrain do
      train[i] = d.data[shuffle[i]]:clone()
      trainLabels[i] = d.labels[shuffle[i]]
   end

   for i=numTrain+1,numTrain+numTest do
      test[i-numTrain] = d.data[shuffle[i]]:clone()
      testLabels[i-numTrain] = d.labels[shuffle[i]]
   end

   local trainData = {
      data = train,
      labels = trainLabels,
      size = function() return numTrain end
   }

   local testData = {
      data = test,
      labels = testLabels,
      size = function() return numTest end
   }
   return trainData,testData

end
--------------------------------------------------------------------
print '==> downloading dataset'

tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'

data_path = 'mnist.t7'
train_file = paths.concat(data_path, 'train_32x32.t7')
test_file = paths.concat(data_path, 'test_32x32.t7')

if not paths.filep(train_file) or not paths.filep(test_file) then
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

----------------------------------------------------------------------
-- training/test size

if opt.size == 'full' then
   print '==> using regular, full training data'
   trsize = 60000 
   tesize = 10000
elseif opt.size == 'small' then
   print '==> using reduced training data, for fast experiments'
   trsize = 6000 
   tesize = 1000
end

----------------------------------------------------------------------
print '==> loading dataset'

loaded = torch.load(train_file, 'ascii')
trainAll = {
   data = loaded.data,
   labels = loaded.labels,
   size = function() return trsize end
}

print '==> Random shuffle and split trainData into train and validation set'
trainData, valData = splitDataset(trainAll, 0.8) -- 80/20 split
valsize = trsize * 0.2
trsize = trsize * 0.8

loaded = torch.load(test_file, 'ascii')
testData = {
   data = loaded.data,
   labels = loaded.labels,
   size = function() return tesize end
}

----------------------------------------------------------------------
print '==> preprocessing data'

-- Preprocessing requires a floating point representation (the original
-- data is stored on bytes)

trainData.data = trainData.data:float()
valData.data = valData.data:float()
testData.data = testData.data:float()

print '==> preprocessing data: normalize globally'

mean = trainData.data[{ {},1,{},{} }]:mean()
std = trainData.data[{ {},1,{},{} }]:std()

trainData.data[{ {},1,{},{} }]:add(-mean)
trainData.data[{ {},1,{},{} }]:div(std)

valData.data[{ {},1,{},{} }]:add(-mean)
valData.data[{ {},1,{},{} }]:div(std)

-- Normalize test data, using the training means/stds
testData.data[{ {},1,{},{} }]:add(-mean)
testData.data[{ {},1,{},{} }]:div(std)


----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

trainMean = trainData.data[{ {},1 }]:mean()
trainStd = trainData.data[{ {},1 }]:std()

valMean = valData.data[{ {},1 }]:mean()
valStd = valData.data[{ {},1 }]:std()

testMean = testData.data[{ {},1 }]:mean()
testStd = testData.data[{ {},1 }]:std()

print('training data size:' ..trainData.size())
print('training data mean: ' .. trainMean)
print('training data standard deviation: ' .. trainStd)

print('validation data size:' ..valData.size())
print('validation data mean: ' .. valMean)
print('validation data standard deviation: ' .. valStd)

print('test data size:' ..testData.size())
print('test data mean: ' .. testMean)
print('test data standard deviation: ' .. testStd)

----------------------------------------------------------------------
if opt.transform then
   print ('==> Preprocessing: elastic transform')

   for i=1, trainData.size() do
      trainData.data[i]=elastic_transform(trainData.data[i],16, 0.25, 8)
   end
   for i=1, valData.size() do
      valData.data[i]=elastic_transform(valData.data[i],16, 0.25, 8)
   end
   for i=1, testData.size() do
      testData.data[i]=elastic_transform(testData.data[i],16, 0.25, 8)
   end

end


-----------------------------------------------------------------------
if opt.visualize then
   print '==> visualizing data'
   if itorch then
      first256Samples = trainData.data[{ {1,256} }]
      itorch.image(first256Samples)
   else
      print("For visualization, run this script in an itorch notebook")
   end
end
