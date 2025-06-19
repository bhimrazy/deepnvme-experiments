# DeepNVMe Experiments
Using DeepNVMe for simple file reads and writes involving CPU/GPU tensors

## Getting Started 
Clone the DeepSpeedExamples repo
```sh
git clone https://github.com/deepspeedai/DeepSpeedExamples.git

# cd into scripts folder
cd DeepSpeedExamples
```


## Install the dependencies
## Setup 
```sh 
pip install deepspeed 

#If `async_io` operator is unavailable, you will need to install the appropriate libaio library binaries for your Linux flavor. For example, Ubuntu users will need to run apt install libaio-dev
sudo apt install libaio-dev

#gds 
https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html 

# to check that compatible status 
ds_report
```

## Start the experiments

```sh
# create dirs
mkdir -p py_out aio_out

# before running the script commment the last line int the script to save the files other wise it seems to get unlinked
python py_store_cpu_tensor.py --nvme_folder py_out  
python py_load_cpu_tensor.py --input_file py_out/test_ouput_1024MB.pt
```

## aio cpu
```sh
# before running the script commment the last line int the script to save the files other wise it seems to get unlinked
python aio_store_cpu_tensor.py --nvme_folder aio_out  
python aio_load_cpu_tensor.py --input_file aio_out/test_ouput_1024MB.pt
```


## Performance Tuning
Ref: https://www.deepspeed.ai/tutorials/deepnvme/?utm_source=chatgpt.com#performance-tuning 
`ds_nvme_tune` automatically explores a user-specified or default configuration space and recommends the option that provides the best read and write performance 

```sh
# create dir
mkdir -p local_nvme

# cpu 
ds_nvme_tune --nvme_dir local_nvme

# gpu
ds_nvme_tune --nvme_dir local_nvme --gpu
```

## References: 
1. https://github.com/deepspeedai/DeepSpeedExamples/tree/master/deepnvme/file_access
2. https://www.deepspeed.ai/tutorials/deepnvme