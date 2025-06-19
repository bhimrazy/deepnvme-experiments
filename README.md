# DeepNVMe Experiments

A collection of experiments demonstrating simple file reads and writes involving CPU/GPU tensors using DeepNVMe.

## Overview

This project guides you through running and experimenting with DeepNVMe for high-performance storage access using both CPU and GPU tensors. The scripts referenced here are cloned from the official DeepSpeedExamples repository.

![image](https://github.com/user-attachments/assets/384e86c4-0aa8-413a-b387-6e9fee80b771)


## Prerequisites

- Python 3.10+
- Linux (tested on Ubuntu)
- [DeepSpeed](https://www.deepspeed.ai/)
- (Optional, for GPU tests) NVIDIA GPU with GPUDirect Storage support
- `libaio` development libraries

## Setup

1. **Clone the Repository**
    ```sh
    git clone https://github.com/deepspeedai/DeepSpeedExamples.git
    cd DeepSpeedExamples/deepnvme/file_access
    ```

2. **Install Dependencies**
    ```sh
    pip install deepspeed
    sudo apt update
    sudo apt install libaio-dev
    ```
    > **Note:** If the `async_io` operator is unavailable, ensure `libaio-dev` is installed.

3. **(Optional) GPUDirect Storage**
    - For GPU-related tests, follow the [NVIDIA GDS troubleshooting guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html).

4. **Check DeepSpeed Installation**
    ```sh
    ds_report
    ```
    ![image](https://github.com/user-attachments/assets/38a25e27-79ea-433b-8fff-edda261a8d4f)


## Running Experiments

1. **Prepare Output Directories**
    ```sh
    mkdir -p py_out aio_out
    ```

2. **CPU Tensor File Operations**
    - **Store Tensor:**
        ```sh
        python py_store_cpu_tensor.py --nvme_folder py_out
        ```
    - **Load Tensor:**
        ```sh
        python py_load_cpu_tensor.py --input_file py_out/test_ouput_1024MB.pt
        ```
    > **Tip:** Comment out the last line in the script to prevent the file from being unlinked (see script, e.g., line 23).

3. **AIO CPU Tensor File Operations**
    - **Store Tensor:**
        ```sh
        python aio_store_cpu_tensor.py --nvme_folder aio_out
        ```
    - **Load Tensor:**
        ```sh
        python aio_load_cpu_tensor.py --input_file aio_out/test_ouput_1024MB.pt
        ```
    > **Note:** The first run may trigger compilation.  
    > **Tip:** Comment out the last line in the script to prevent the file from being unlinked (see script, e.g., line 44).

## Performance Tuning

Use `ds_nvme_tune` to automatically find optimal NVMe settings:

```sh
mkdir -p local_nvme
ds_nvme_tune --nvme_dir local_nvme         # For CPU
ds_nvme_tune --nvme_dir local_nvme --gpu   # For GPU
```

See the [DeepSpeed DeepNVMe tutorial](https://www.deepspeed.ai/tutorials/deepnvme/?utm_source=chatgpt.com#performance-tuning) for more details.

## References

- [DeepSpeedExamples: deepnvme/file_access](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/deepnvme/file_access)
- [DeepSpeed DeepNVMe Tutorial](https://www.deepspeed.ai/tutorials/deepnvme)
