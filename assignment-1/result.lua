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


model_path='.'

print("loading trained model")
model = torch.load(paths.concat(model_path,'model.net'))


data_path = 'mnist.t7'
test_file = paths.concat(data_path, 'test_32x32.t7')
train_file = paths.concat(data_path,'train_32x32.t7')

print("loading testing data")
loaded = torch.load(test_file, 'ascii')

testData = {
   data = loaded.data,
   labels = loaded.labels,
   size = function() return tesize end
}

print("loading training data (for pre-processing)")
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
--
--
-- -- Local normalization
print '==> preprocessing data: normalize locally'
--
-- -- Define the normalization neighborhood:
neighborhood = image.gaussian1D(7)
--
-- -- Define our local normalization operator (It is an actual nn module, 
-- -- which could be inserted into a trainable model):
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()
--
-- -- Normalize all channels locally:

print("Normalizing test data")
for i = 1,testData:size() do
     testData.data[{ i,{1},{},{} }] = normalization:forward(testData.data[{ i,{1},{},{} }])
end

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
      -- disp progress
      xlua.progress(t, testData:size())
	f:write(t..",")
      -- get new sample
      local input = testData.data[t]
      input = input:double()
--      if opt.type == 'double' then input = input:double()
 --     elseif opt.type == 'cuda' then input = input:cuda() end
      local target = testData.labels[t]

      -- test sample
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

   -- print confusion matrix

   -- update log/plot

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end

   -- next iteration:
end

preproc()
test()
