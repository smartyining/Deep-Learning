require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'csvigo'
require 'cunn'
dofile 'provider.lua'
trsize = 5000
testsize = 8000
channel = 3
height = 96
width = 96


-- parse STL-10 data from table into Tensor
function parseDataLabel(d, numSamples, numChannels, height, width)
   local t = torch.ByteTensor(numSamples, numChannels, height, width)
   local l = torch.ByteTensor(numSamples)
   local idx = 1
   for i = 1, #d do
      local this_d = d[i]
      for j = 1, #this_d do
    t[idx]:copy(this_d[j])
    l[idx] = i
    idx = idx + 1
      end
   end
   assert(idx == numSamples+1)
   return t, l
end

print '==> loading dataset'

test_path = 'stl-10/test.t7b'

provider = torch.load('provider.t7')

testData = {
     data = torch.Tensor(),
     labels = torch.Tensor(),
     size = function() return testsize end
}


local raw_test = torch.load(test_path)

testData.data, testData.labels = parseDataLabel(raw_test.data,
                                               testsize, channel, height, width)
testData.data = testData.data:float()
testData.labels = testData.labels:float()

--[[
-- load previous saved model
print '==> loading trained model'
model = torch.load('model.net')

f = io.open('predictions.csv', 'w')
f:write("Id,Prediction\n")
--]]

function preproc()
   -- preprocess trainSet
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))

  -- get train mean
  local mean_u = provider.mean_u
  local std_u = provider.std_u
  local mean_v = provider.mean_u
  local std_v = provider.std_v


  --preprocess test data
  for i = 1,testData:size() do
    xlua.progress(i,testData:size())
     -- rgb -> yuv
     local rgb = testData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[{1}] = normalization(yuv[{{1}}])
     testData.data[i] = yuv
  end

  -- normalize u globally:
  testData.data:select(2,2):add(-mean_u)
  testData.data:select(2,2):div(std_u)
  -- normalize v globally:
  testData.data:select(2,3):add(-mean_v)
  testData.data:select(2,3):div(std_v)

end

preproc()
torch.save('testData.t7',testData)
