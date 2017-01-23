
utils = {}

-- function reload(name)
--     package.loaded[name] = nil
--     return require(name)
-- end

image = require 'image'
require 'nn'
require 'optim'

function utils.file_exists(name)
    local f=io.open(name,"r")
    if f~=nil then
        io.close(f)
        return true
    else
        return false
    end
end

function utils.load_data()

  -- download dataset if necessary
  if not utils.file_exists('./dataset/images.t7') then
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

  labels_train = labels[{{1,700,{},{},{}}}]
  labels_validation = labels[{{701,m,{},{},{}}}]

  data_train = data[{{1,700},{},{},{}}]
  data_validation = data[{{701,m},{},{},{}}]

  return labels_train, labels_validation, data_train, data_validation

end





-- function start_training(model, criterion, data_train, labels_train)
function utils.start_training(params)

--     params
    local model_name = params.model_name
    local sgd_params = params.sgd_params
    local model = params.model
    local criterion = params.criterion
    local data_train = params.data_train
    local labels_train = params.labels_train
    local num_of_epochs = params.num_of_epochs
    local batch_size = params.batch_size
    

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
--     batch_size = 20
    n_of_batches = math.floor(m/batch_size)

    print('start training ' .. model_name)
    print('epoch', 'duration', 'loss')
    local losses = {}
    for epoch = 1, num_of_epochs do
        
        local current_loss = 0
        local t = sys.clock()
        sys.tic()

    --  mini-batch SGD
        for n=1,n_of_batches do
            start = batch_size * (n-1) + 1
            finish = batch_size * n
            data_batch = data_train[{{start,finish}}]
            label_batch = labels_train[{{start,finish}}]
            _, fs = optim.sgd(feval,x,sgd_params)
            current_loss = current_loss + (fs[1]/batch_size)
        end
        table.insert(losses, current_loss)


--         if epoch % 10 == 0 then

    --      log results

            t = sys.toc() 
            duration = math.floor(t)..'s'

            print(epoch, duration, current_loss)
--             logger:add{['training error'] = current_loss}
--             logger:style{['training error'] = '-'}
--             logger:plot()

--     --      save model
--             if epoch % 50 == 0 then
--                 torch.save('./models/model.t7', model)
--                 print('model saved')
--             end

--         end
    end
    
    result_dir = 'results/'..model_name
    model_path = result_dir .. '/model.t7'
    losses_path = result_dir .. '/losses.t7'
    os.execute('mkdir -p ' .. result_dir)
    torch.save(model_path, model)
    print('model saved')
    torch.save(losses_path, losses)
    print('losses saved')
end







return utils
