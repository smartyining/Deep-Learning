m = require 'manifold';
N = 2000
mnist = require 'mnist';
testset = mnist.testdataset()
--testset

testset.size  = N
testset.data  = testset.data[{{1,N}}]
testset.label = testset.label[{{1,N}}]

x = torch.DoubleTensor(testset.data:size()):copy(testset.data)
x:resize(x:size(1), x:size(2) * x:size(3))
labels = testset.label

opts = {ndims = 2, perplexity = 30, pca = 50, use_bh = true, theta=0.5}
mapped_x1 = m.embedding.tsne(x, opts)
im_size = 4096
map_im = m.draw_image_map(mapped_x1, x:resize(x:size(1), 1, 28, 28), im_size, 0, true)
image.save('tsne.png',map_im)
