-- This is used to load unlabeled data
-- Please run provider.lua before this one to download 'extra.t7b'
-- Also, current scripts only use 10% of unlabel data because original 
-- size will be too large to fit in.

require 'nn'
require 'image'
require 'xlua'
require 'unsup'

torch.setdefaulttensortype('torch.FloatTensor')

local Extradata = torch.class 'Extradata'

function Extradata:__init(full)
	filename = 'stl-10/extra.t7b'
	numSamples = 100000
	numChannels = 3
	height = 96
	width = 96

	trsize = numSamples/10

	self.trainData = {
	     data = torch.ByteTensor(trsize, numChannels, height, width),
	     size = function() return trsize end
	}
	
	raw_table = torch.load(filename)
	-- load raw data 
	for i=1 , trsize do
		self.trainData.data[i] :copy(raw_table.data[1][9*i])
	end

	self.trainData.data = self.trainData.data:float()

	collectgarbage()
end


function Extradata:normalize()

  local trainData = self.trainData
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1,trainData:size() do
     xlua.progress(i, trainData:size())
     -- rgb -> yuv
     local rgb = trainData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
     trainData.data[i] = yuv
  end
  -- normalize u globally:
  local mean_u = trainData.data:select(2,2):mean()
  local std_u = trainData.data:select(2,2):std()
  trainData.data:select(2,2):add(-mean_u)
  trainData.data:select(2,2):div(std_u)
  -- normalize v globally:
  local mean_v = trainData.data:select(2,3):mean()
  local std_v = trainData.data:select(2,3):std()
  trainData.data:select(2,3):add(-mean_v)
  trainData.data:select(2,3):div(std_v)

  trainData.mean_u = mean_u
  trainData.std_u = std_u
  trainData.mean_v = mean_v
  trainData.std_v = std_v


end

function Extradata:whiten()
	self.trainData.data = unsup.zca_whiten(self.trainData.data)[1]
end







