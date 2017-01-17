
-- local cmd = torch.CmdLine()
-- cmd:text()
-- cmd:text("torch-sr")
-- cmd:text("Options:")
-- cmd:option("-port", 8812, 'listen port')

-- local opt = cmd:parse(arg)

-- print(opt.port)



image = require 'image'
require 'nn'
require 'os'
require 'paths'
require 'optim'
require 'sys'

-- os.execute('mkdir -p dataset logs models')

-- logger = optim.Logger('./logs/loss_log.txt')

-- download dataset if necessary

function file_exists(name)
    local f=io.open(name,"r")
    if f~=nil then
        io.close(f)
        return true
    else
        return false
    end
end

if not file_exists('./dataset/images.t7') then
    print('downloading')
    os.execute('wget -c http://ml1.oss-cn-hongkong.aliyuncs.com/art/web/sr/dataset/images.t7.zip -O ./dataset/images.t7.zip')
    os.execute('unzip -o ./dataset/images.t7.zip -d ./dataset')
    print('downloaded and unzipped')
else
    print('file exists')
end


-- load dataset to a Tensor and generate low-resolution dataset

labels = torch.load("./dataset/images.t7")
m = labels:size()[1]
data = torch.Tensor(m, 3, 51, 51)
for i=1,m do
    data[i] = image.scale(labels[i], 51, 51)
end


-- divide dataset into train and validation

labels_train = labels[{{1,700,{},{},{}}}]
labels_validation = labels[{{701,m,{},{},{}}}]

data_train = data[{{1,700},{},{},{}}]
data_validation = data[{{701,m},{},{},{}}]

print('train data size: ', data_train:size()[1])
print('validation data size: ', data_validation:size()[1])
-- print(data_train:size(), data_validation:size())
-- print(labels_train:size(), labels_validation:size())

-- sample some images
-- itorch.image(labels[{{1,3}}])
-- itorch.image(data[{{1,3}}])

-- nn model and criterion

load_from_model = true
if load_from_model and file_exists('./models/model.t7') then
    model = torch.load('./models/model.t7')
else 
    model = nn.Sequential()
    model:add(nn.SpatialConvolution(3,16,3,3,1,1,1,1))
    model:add(nn.LeakyReLU(0.2))
    model:add(nn.SpatialFullConvolution(16, 3, 2, 2, 2, 2, 0, 0, 0, 0))
    model:add(nn.LeakyReLU(0.2))
    model:add(nn.View(3*102*102))
    model:add(nn.Tanh())
end

criterion = nn.MSECriterion()


-- torch.setnumthreads(4)
torch.manualSeed(0)

sgd_params = {
   learningRate = 1e-3,
   learningRateDecay = 1e-4,
   weightDecay = 0.2,
   momentum = 0.2
}

x, dl_params = model:getParameters()


-- feval function to be used for optim
function feval(x_new)

  if x ~= x_new then
    x:copy(x_new)
  end

  dl_params:zero()

  local outputs = model:forward(data_batch)
  local loss = criterion:forward(outputs, label_batch)
  local dloss_doutputs = criterion:backward(outputs, label_batch)
  model:backward(data_batch, dloss_doutputs)
  return loss, dl_params

end

-- minibatch settings
m = data_train:size()[1]
batch_size = 20
n_of_batches = math.floor(m/batch_size)

print('start training...')
print('epoch', 'duration', 'loss')
for epoch = 1, 100 do

    current_loss = 0
    start_time = os.time()

--  mini-batch SGD
    for n=1,n_of_batches do
        start = batch_size * (n-1) + 1
        finish = batch_size * n
        data_batch = data_train[{{start,finish}}]
        label_batch = labels_train[{{start,finish}}]
        _, fs = optim.sgd(feval,x,sgd_params)
        current_loss = current_loss + (fs[1]/batch_size)
    end
    
    if epoch % 10 == 0 then

--      log results
        duration = os.time() - start_time
        duration = (duration)..'s'

        print(epoch, duration, current_loss)
        -- logger:add{['training error'] = current_loss}
        -- logger:style{['training error'] = '-'}
        -- logger:plot()

--      save model
--         if epoch % 50 == 0 then
        torch.save('./models/model.t7', model)
        print('model saved')
--         end

    end
end

