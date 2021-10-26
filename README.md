# DLSR_LAB_learning
download dataset:
dataset https://www.kaggle.com/tohidul/food11
chmod 777 build_imbalanced_food11.sh
./build_imbalanced_food11.sh
train the model:python3 trainTheModel.py
load and check testing the accuracy of exist model :python3 loadAndTestModel.py 

## brid
floder :
* 2021VRDL_HW1_datasets/
    * testing_dataset/
    * training_dataset/
    * validation_dataset/ 這個我自己加的 把300張training_dataset 搬過來
* code/
    * DLSR_LAB_learning/
        * train.py
        * test.py


### brid training

python3 train.py

### brid testing
python3 test.py



## food11
convert model to onnx:
python3 netToOnnx.py

convert model to onnx with dynamic batch size:
python3 dynamicNetToOnnx.py


openvino (should have enviroment)
optimization:
cd openvino
python3 mo.py --input_model /workspace/DLSR_LAB_learning/trainedModel/super_resolution.onnx --output_dir /workspace/DLSR_LAB_learning/trainedModel

inference:
in .zshrc:
source /opt/intel/openvino_2020.2.120/bin/setupvars.sh
shell:
source .zshrc
python3 openvino__loadmodel.py
