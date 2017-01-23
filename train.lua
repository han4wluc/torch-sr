
require 'utils'
require 'nn'

labels_train, labels_validation, data_train, data_validation = utils.load_data()

criterion = nn.MSECriterion()

models = require 'models'

sgd_params = {
   learningRate = 1e-1,
   learningRateDecay = 1e-3,
   weightDecay = 0.9,
   momentum = 0.0001
   -- weightDecay = 0,
   -- momentum = 0
}

model_names = {"srnn_9_1_6", "srnn_9_5_6"}

for i, model_name in ipairs(model_names) do
    local params = {
        sgd_params = sgd_params,
        model = models[model_name].model,
        criterion = criterion,
        data_train = data_train,
        labels_train = labels_train,

        num_of_epochs = 80,
        batch_size = 20,

        model_name = model_name
    }

    utils.start_training(params)

end



-- utils.start_training(params)

print('done')