from loko_extensions.model.components import Component, Input, Output, save_extensions, Select, Arg, Dynamic, MKVField, \
    MultiKeyValue, AsyncSelect


#########################   DOCS     ###########################

detector_description = '''
### Description
.....
### Input
.....
### Output
.....

'''

#########################   ARGS     ###########################

args_list = []

#arg = Arg(name="model_file", label= "Model File", type="files")
internal_folder_arg = Arg(name="internal_folder", label= "Internal dataset folder", description="Internal dataset folder to save file", type="text")
data_type_arg = Arg(name="data_type", label= "Data type in dataset", description="Specify if data to save are test or train, labels or images: "
                                                                                 "y_train, y_test, X_train, X_test", type="text")

model_name_arg = Select(name="model_name", options=["base"], label="Model to train", description="Name of the model to train. Base : simple neural network")

adversarial_method_arg = Select(name='adversarial_method', options=["carlini_wagner"], label='Adversarial Method')


trained_model_arg = AsyncSelect(name='trained_model', label='Trained model file', url='http://localhost:9999/routes/model_adversarial_detection/services/models')
dataset_arg = AsyncSelect(name='dataset', label='Dataset', url='http://localhost:9999/routes/model_adversarial_detection/services/datasets')

args_list.append(internal_folder_arg)
args_list.append(data_type_arg)
args_list.append(model_name_arg)
args_list.append(trained_model_arg)
args_list.append(dataset_arg)
args_list.append(adversarial_method_arg)


#########################   INPUT   ###########################


input_list = [Input(id="dataset", label="Dataset", to="dataset", service="/services/dataset/read"),
    Input(id="train_model", label="Train Model", to="train_model", service="/services/model/train"),
    Input(id="detect_model", label="Detect Model with adversarial examples", to="detect_model", service="/services/model/detect")
              ]

#########################   OUTPUT   ###########################

output_list = [Output(id="dataset", label="Dataset"),
               Output(id="train_model", label="Train Model"),
               Output(id="detect_model", label="Detect Model"),
               ]

#########################   COMPONENT   ###########################

prescriptor = Component(name="Model Adversarial Detection", description=detector_description,
                        inputs=input_list,
                        outputs=output_list,
                        args=args_list,
                        configured=True,
                        trigger=True)

save_extensions([prescriptor], path="../extensions")

