# DLSR_LAB_learning
download dataset:
dataset https://www.kaggle.com/tohidul/food11
chmod 777 build_imbalanced_food11.sh
./build_imbalanced_food11.sh
train the model:python3 trainTheModel.py
load and check testing the accuracy of exist model :python3 loadAndTestModel.py 




convert model to onnx:
python3 netToOnnx.py


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
