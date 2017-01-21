
require 'utils'
require 'nn'

labels_train, labels_validation, data_train, data_validation = utils.load_data()

-- model = nn.Sequential()
-- model:add(nn.SpatialConvolution(3,16,3,3,1,1,1,1))
-- model:add(nn.LeakyReLU(0.2))
-- model:add(nn.SpatialFullConvolution(16, 3, 2, 2, 2, 2, 0, 0, 0, 0))
-- model:add(nn.LeakyReLU(0.2))
-- model:add(nn.View(3*102*102))
-- model:add(nn.Tanh())

criterion = nn.MSECriterion()

models = require 'models'

local srnn_9_1_6 = models.srnn_9_1_6
local srnn_9_5_6 = models.srnn_9_5_6

sgd_params = {
   learningRate = 1e-2,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}

params = {
    sgd_params = sgd_params,
    model = model,
    criterion = criterion,
    data_train = data_train,
    labels_train = labels_train,

    num_of_epochs = 1,
    batch_size = 20,
    model_name = 'srcnn_test'
}

model_names = {"srnn_9_1_6", "srnn_9_5_6"}

for i, model_name in ipairs(model_names) do
    params = {
        sgd_params = sgd_params,
        model = models[model_name].model,
        criterion = criterion,
        data_train = data_train,
        labels_train = labels_train,

        num_of_epochs = 1,
        batch_size = 20,
        -- model_name = 'srcnn_test'
        model_name = model_name
    }


    utils.start_training(params)

end



-- utils.start_training(params)

print('done')