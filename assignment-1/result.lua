----------------------------------------------------------------------
-- This script implements load the previous saved model and
-- generate prediction on test set
----------------------------------------------------------------------
require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'csvigo'

trsize = 60000
tesize = 10000

print '==> downloading dataset'

tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'

data_path = 'mnist.t7'
train_file = paths.concat(data_path, 'train_32x32.t7')
test_file = paths.concat(data_path, 'test_32x32.t7')

if not paths.filep(train_file) or not paths.filep(test_file) then
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

-- load previous saved model
print '==> loading trained model'
model_path='results'
model = torch.load(paths.concat(model_path,'model.net'))


print '==> loading testing data'
loaded = torch.load(test_file, 'ascii')
testData = {
   data = loaded.data,
   labels = loaded.labels,
   size = function() return tesize end
}

print '==> loading training data (for pre-processing)'
loaded = torch.load(train_file, 'ascii')
trainData = {
   data = loaded.data,
   labels = loaded.labels,
   size = function() return trsize end
}


f = io.open('predictions.csv', 'w')
f:write("Id,Prediction\n")

function preproc()
   --preprocessing testing data 

   testData.data = testData.data:float()
   trainData.data = trainData.data:float()

   print '==> preprocessing data: normalize globally'

   mean = trainData.data[{ {},1,{},{} }]:mean()
   std = trainData.data[{ {},1,{},{} }]:std()

   trainData.data[{ {},1,{},{} }]:add(-mean)
   trainData.data[{ {},1,{},{} }]:div(std)

   -- Normalize test data, using the training means/stds
   testData.data[{ {},1,{},{} }]:add(-mean)
   testData.data[{ {},1,{},{} }]:div(std)

end

function test()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')

   for t = 1,testData:size() do
      -- display progress
      xlua.progress(t, testData:size())
	   f:write(t..",")

      -- get new sample
      local input = testData.data[t]
      input = input:double()
      local target = testData.labels[t]

      -- test sample and write to file
      local pred = model:forward(input)
      val,ind=torch.max(pred,1)        
	   f:write(ind[1]%10)
	   f:write("\n")
   end
   
   f.close()

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
end

preproc()
test()
