require 'torch'
require 'nn'
require 'cunn'
require 'optim'
require 'image'
m = require 'manifold';
dofile 'test_aug.lua'

model_p = '/home/ubuntu/ds-ga-1008-a2/yining/initsampleextra_2/model.net'

testset = torch.load 'testData_1k.t7'
function preproc()
   -- preprocess trainSet
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))

  -- get train mean
  local mean_u = testset.mean_u
  local std_u = testset.std_u
  local mean_v = testset.mean_v
  local std_v = testset.std_v


--  preprocess test data
  for i = 1,testset.data:size(1) do
    xlua.progress(i,testset.data:size(1))
     -- rgb -> yuv
     local rgb = testset.data[i]
     local yuv = image.rgb2yuv(rgb):float()
     -- normalize y locally:
     yuv[{1}] = normalization(yuv[{{1}}])
     testset.data[i] = yuv
  end

  -- normalize u globally:
  testset.data:select(2,2):add(-mean_u)
  testset.data:select(2,2):div(std_u)
  -- normalize v globally:
  testset.data:select(2,3):add(-mean_v)
  testset.data:select(2,3):div(std_v)

end


N = 1000
--testset = torch.load 'testset_unnorm.t7'
model = torch.load(model_p)

--raw = testset.data:clone()
raw = torch.Tensor(testset.data:size()):copy(testset.data)
fn = '../christina/tsne25/test.png'
image.save(fn,raw[1])

model:remove(27)
model:remove(26)
print(model)
print(testset.data:size())
print(testset.labels:size())
print(model)
model:evaluate()
preproc()
testset.data = testset.data:cuda()
testset.labels = testset.labels:cuda()

outs = torch.DoubleTensor(testset.data:size(1),256,2,2)
bs = 25
for i=1,testset.data:size(1),bs do
-- print(provider.valData.labels:narrow(1,i,bs):size()) 
    local outputs = model:forward(testset.data:narrow(1,i,bs))
    outs[{{i,i+bs-1}}]:copy(outputs)
-- print(outputs:size()) 
 -- print(provider.valData.labels:narrow(1,i,bs):size()) 
  end
--print(outs[{{900}}])
nfilts = outs:size(2)
for i = 2,nfilts do

--outputs1 = testset.data
--myout = outs[{ {},{i,i}}]
myout = outs[{ {},i}]

x = torch.DoubleTensor(myout:size()):copy(myout)
print(x:size())
--s2 = x:size(2)
--s3 = x:size(3)
x:resize(x:size(1), x:size(2) * x:size(3))
--x:resize(x:size(1), x:size(2) * x:size(3))
labels = testset.labels


xlua.progress(i,nfilts)
opts = {ndims = 2, perplexity = 30, pca = 50, use_bh = true, theta=0.5}
mapped_x1 = m.embedding.tsne(x, opts)
im_size = 2048
--map_im = m.draw_image_map(mapped_x1, x:resize(x:size(1), 1, s2, s3), im_size, 0, true)
map_im = m.draw_image_map(mapped_x1, raw, im_size, 0, true)
fn = '../christina/tsne25/tsne_25ev'..i..'.png'
image.save(fn,map_im)
end
