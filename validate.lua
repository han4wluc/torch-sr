
function validate(model_name, data)

    local model_dir = './results/' .. model_name .. '/'
    local images_dir = model_dir .. 'images/'
    local model = torch.load(model_dir .. 'model.t7')

    m = data:size()[1]

    os.execute('mkdir -p ' .. images_dir)

    for i=1, m do
        local img = model:forward(data[i]):resize(3,102,102)
        image.save(images_dir .. 'img_' .. tostring(i) .. '.jpg', img)
    end
    
    print('files saved', model_name)
    
end

require 'utils'

labels_train, labels_validation, data_train, data_validation = utils.load_data()

model_names = utils.get_model_names()

for i, model_name in ipairs(model_names) do
  validate(model_name, data_validation)
end
