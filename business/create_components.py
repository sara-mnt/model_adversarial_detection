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

arg = Arg(name="model_file", label= "Model File", type="files")
internal_folder_arg = Arg(name="internal_folder", label= "Internal dataset folder", description="Save file in internal dataset folder", type="text")
data_type_arg = Arg(name="data_type", label= "Data type in dataset", description="Specify if data to save are test or train, labels or images ", type="text")

mkvfields = [MKVField(name='cond1', label='Cond1111', required=True),
             MKVField(name='cond2', label='cond2'),
             MKVField(name='cond3', label='Cond3'), ]

multikeyvalue = MultiKeyValue(name='CONSTRAINTS', label='CONSTRAINTS', fields=mkvfields,
                              group='Advanced Args')

model_arg = AsyncSelect(name='model', label='Model', url='http://localhost:8080/services/models')
dataset_arg = AsyncSelect(name='dataset', label='Dataset', url='http://localhost:8080/services/datasets')
adversarial_method_arg = AsyncSelect(name='adversarial_method', label='Adversarial Method', url='http://localhost:8080/services/adversarial_method')

args_list.append(internal_folder_arg)
args_list.append(data_type_arg)
args_list.append(model_arg)
args_list.append(dataset_arg)
args_list.append(adversarial_method_arg)


#########################   INPUT   ###########################


input_list = [Input(id="dataset", label="Dataset", to="dataset", service="/services/dataset/read"),
Input(id="train_model", label="Train Model", to="train_model", service=""),
    Input(id="evaluate_model", label="Evaluate Model", to="evaluate_model", service="estimate_model")
              ]

#########################   OUTPUT   ###########################

output_list = [Output(id="evaluate_model", label="Evaluate Model"),
Output(id="dataset", label="Dataset")
               ]

#########################   COMPONENT   ###########################

prescriptor = Component(name="CW_adversarial_detector", description=detector_description, inputs=input_list,
                        outputs=output_list,
                        args=args_list,
                        configured=True,
                        trigger=True)

save_extensions([prescriptor], path="../extensions")

