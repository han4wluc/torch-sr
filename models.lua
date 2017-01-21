
models = {}

-- Image Super-Resolution Using Deep Convolutional Networks
-- Chao Dong, Chen Change Loy, Member, IEEE, Kaiming He, Member, IEEE, and Xiaoou Tang, Fellow, IEEE
-- 31 Dec 2014 ~ Jul 2015
-- https://arxiv.org/abs/1501.00092

-- difference between paper: use padding and 3rd layer filter size 6 to keep 2x sizing.

srnn_9_1_6 = {}
srnn_9_1_6.name = 'srnn_9_1_6'
srnn_9_1_6.model = nn.Sequential()
srnn_9_1_6.model:add(nn.SpatialConvolution(3,64,9,9,1,1,4,4))
srnn_9_1_6.model:add(nn.ReLU())
srnn_9_1_6.model:add(nn.SpatialConvolution(64,32,1,1,1,1,0,0))
srnn_9_1_6.model:add(nn.ReLU())
srnn_9_1_6.model:add(nn.SpatialFullConvolution(32,3,6,6,2,2,2,2))
srnn_9_1_6.model:add(nn.ReLU())


srnn_9_5_6 = {}
srnn_9_5_6.name = 'srnn_9_5_6'
srnn_9_5_6.model = nn.Sequential()
srnn_9_5_6.model:add(nn.SpatialConvolution(3,64,9,9,1,1,4,4))
srnn_9_5_6.model:add(nn.ReLU())
srnn_9_5_6.model:add(nn.SpatialConvolution(64,32,5,5,1,1,2,2))
srnn_9_5_6.model:add(nn.ReLU())
srnn_9_5_6.model:add(nn.SpatialFullConvolution(32,3,6,6,2,2,2,2))
srnn_9_5_6.model:add(nn.ReLU())


-- Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
-- Wenzhe Shi, Jose Caballero, Ferenc Huszár, Johannes Totz, Andrew P. Aitken, Rob Bishop, Daniel Rueckert, Zehan Wang
-- Sep 2016
-- https://arxiv.org/abs/1609.05158


-- Accurate Image Super-Resolution Using Very Deep Convolutional Networks
-- Jiwon Kim, Jung Kwon Lee and Kyoung Mu Lee Department of ECE, ASRI, Seoul National University, Korea
-- Nov 2016
-- https://arxiv.org/abs/1511.04587
-- * gadient clipping
-- * zero padding
-- * resnets





-- Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
-- Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi
-- Nov 2016
-- https://arxiv.org/abs/1609.04802


models.srnn_9_1_6 = srnn_9_1_6
models.srnn_9_5_6 = srnn_9_5_6

return models
