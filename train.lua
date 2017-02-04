
require 'utils'
require 'nn'

torch.manualSeed(0)


labels_train, labels_validation, data_train, data_validation = utils.load_data()

criterion = nn.MSECriterion()
-- criterion.sizeAverage = false

models = require 'models'

resume_training = false

sgd_params = {
   learningRate = 1e-2,
   learningRateDecay = 1e-4,
   -- weightDecay = 9e-1,
   -- momentum = 1e-4
   weightDecay = 0,
   momentum = 0
}

model_names = utils.get_model_names()


for i, model_name in ipairs(model_names) do

    local model = models[model_name].model
    local model_file = "./results/" .. model_name .. "/model.t7"

    if resume_training and utils.file_exists(model_file) then
        print('resuming')
        model = torch.load(model_file)
    end

    local params = {
        sgd_params = sgd_params,
        model = model,
        criterion = criterion,
        data_train = data_train,
        labels_train = labels_train,
        resume_training = resume_training,

        num_of_epochs = 2000,
        batch_size = 70,
        -- batch_size = 100,
        -- lr_theta = 0.5, -- used for gradient clipping

        model_name = model_name
    }

    utils.start_training(params)

end

-- utils.start_training(params)

print('done')
