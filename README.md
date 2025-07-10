# Health Care with AI

*Project: Final proect*  
*Class: MindX - CSI12*  
*name: duyhieu12890*

## Requirements

### Windows
- Windows 10 or higher
- Python 3.12
- [TensorFlow](https://www.tensorflow.org/install) (CPU only)  
    If you need CUDA support, use the latest TensorFlow 2.12 and ensure you have a compatible NVIDIA GPU and CUDA/cuDNN installed.
- Recommended: [pip](https://pip.pypa.io/en/stable/), [virtualenv](https://virtualenv.pypa.io/en/latest/)

### Linux
- Any distribution that supports Python 3.12
- [TensorFlow](https://www.tensorflow.org/install) (CPU and CUDA supported)  
    For CUDA support, ensure you have a compatible NVIDIA GPU and CUDA/cuDNN installed.
- Recommended: [pip](https://pip.pypa.io/en/stable/), [virtualenv](https://virtualenv.pypa.io/en/latest/)

### General
- Internet connection for installing dependencies
- Sufficient disk space for datasets and models
- Basic knowledge of Python and command-line usage
- (Optional) [Jupyter Notebook](https://jupyter.org/) for interactive development

## Installation

### 1. Install Python

- Download and install Python 3.12 from the [official website](https://www.python.org/downloads/).
- Make sure to check "Add Python to PATH" during installation.

### 2. Create a Virtual Environment

Open a terminal (or Command Prompt) in your project directory and run:

```bash
python -m venv .
```

- **Activate venv:**
    - **Windows:**  
        ```bash
        .\Scripts\activate
        ```
    - **Linux/macOS:**  
        ```bash
        source bin/activate
        ```
- **Deactivate venv:**  
    ```bash
    deactivate
    ```

### 3. Install Dependencies

#### 3.1 Install Tensorflow

##### On Windows

- **CPU only:**
    ```bash
    pip install -r requirements-cpu.txt
    ```
- **With CUDA (TensorFlow 2.12):**
    1. Ensure you have a compatible NVIDIA GPU, [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), and [cuDNN](https://developer.nvidia.com/cudnn) installed.
    2. Install dependencies:
         ```bash
         pip install -r requirements-windows-cuda.txt
         ```
- **With ROCm (AMD):**
    I don't know ¯\_(ツ)_/¯

##### On Linux

- **CPU only:**
    ```bash
    pip install -r requirements-cpu.txt
    ```
- **With CUDA:**
    1. Ensure you have a compatible NVIDIA GPU, CUDA, and cuDNN installed.
    2. Install dependencies:
         ```bash
         pip install -r requirements-linux-cuda.txt
         ```

- **With ROCm (AMD):**
    I don't know ¯\_(ツ)_/¯

> **Note:** Always activate your virtual environment before installing or running the project.

#### 3.2 Install Components

*Next, Install Components by enter this command:*  
    ```bash
        pip install -r requirements.txt
    ```

*Install Cloudflared tunnel*
    