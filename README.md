
# Project Descriptions

PyTorch implementation of Table detect usiing Yolo

## Project Structure

```
.
├── data                      # Sample test case
├── detectIMG                 # Detection Model results
├── model_ckpt                # Trained checkpoint for custom models
├── models                    # Detection model creation module
├── utils                     # Utils module to load all utilities
├── custoom                   # Custom model module to generate the response
├── api.py                    # API for integration
├── README.md                 # README for codebase
└── requirements.txt          # Install all requirements

```

## Install packages required


```
$ conda create -n <env name> python==3.10.8

$ conda activate <env_name>

$ pip install -r requirements.txt

$ conda install -c conda-forge poppler

```

## Running ML inference

```
python api.py

For curl command use:

curl -X POST -F file=@<"file_path"> http://127.0.0.1:8500/predict

For command line:

python main.py --file_path <"File path">

```

## Limitations

TODO

