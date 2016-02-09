----------------------------------------------------------------------
-- This tutorial shows how to train different models on the MNIST
-- dataset using multiple optimization techniques (SGD, ASGD, CG), and
-- multiple types of models.
--
-- This script demonstrates a classical example of training 
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem. 
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
--
-- Clement Farabet
----------------------------------------------------------------------
require 'torch'
require 'gnuplot'

----------------------------------------------------------------------
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST Loss Function')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-maxEpoch',20)
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
-- data:
cmd:option('-size', 'full', 'how many samples do we load: small | full')
-- model:
cmd:option('-model', 'convnet', 'type of model to construct: linear | mlp | convnet |mixed')
-- loss:
cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin |ce')
-- training:
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', false, 'live plot')
cmd:option('-transform',false)
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-type', 'double', 'type: double | float | cuda')
cmd:option('-transferF','tanh','transferF: tanh | sigm | rectified')

cmd:text()
opt = cmd:parse(arg or {})

-- nb of threads and fixed seed (for repeatable experiments)
if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
print '==> executing all'

dofile '1_data.lua'
dofile '2_model.lua'
dofile '3_loss.lua'
dofile '4_train.lua'
dofile '5_test.lua'

----------------------------------------------------------------------
print '==> training!'

-- store result to plot
result = torch.Tensor(4,opt.maxEpoch)

for it=1,4 do
	timer = torch.Timer()
	if it==1 then
		opt.loss='mse'
	elseif it==2 then
		opt.loss='nll'
	elseif it==3 then
		opt.loss='margin'
	elseif it==4 then
		opt.loss='ce'
	end

	for epoch=1, opt.maxEpoch do
	   train(epoch)
	   result[it][epoch] = test(valData)
	end
	print('Time elapsed for training' .. timer:time().real .. ' seconds')

	test(testData)
end


--- plot result
gnuplot.pngfigure('loss.png')
gnuplot.plot(
   {'MSE', result[1],  '-'},
   {'negative log likelihood ', result[2],  '-'},
   {'margin', result[3], '-'},
   {'Cross Entropy ', result[4],  '-'}
   )
gnuplot.xlabel('Epoch')
gnuplot.ylabel('Accuracy on Validation Set')
gnuplot.grid(true)
gnuplot.plotflush()




