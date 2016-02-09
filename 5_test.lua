----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test(dataSet)
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
   for t = 1,dataSet:size() do
      -- disp progress
      xlua.progress(t, dataSet:size())

      -- get new sample
      local input = dataSet.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = dataSet.labels[t]

      -- test sample
      local pred = model:forward(input)
      confusion:add(pred, target)
   end

   -- timing
   time = sys.clock() - time
   time = time / dataSet:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   acc = confusion.totalValid
   -- update log/plot
   if not dataSet=='testData' then
      testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
      if opt.plot then
         testLogger:style{['% mean class accuracy (test set)'] = '-'}
         testLogger:plot()
      end
   else
      valLogger:add{['% mean class accuracy (validation set)'] = confusion.totalValid * 100}
      if opt.plot then
         valLogger:style{['% mean class accuracy (validation set)'] = '-'}
         valLogger:plot()
      end
   end
   
   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end

   
   -- next iteration:
   confusion:zero()
   return acc
end
