# LLM Robustness Against Misinformation in Biomedical Question Answering

Code and resources that implement experiments for the paper _LLM Robustness Against Misinformation in Biomedical Question Answering_.

## Usage

The following sections describe how to use the code.

### Cloning the Repository

To properly clone the repository and ensure all large files are pulled, follow these steps:

1. **Install Git LFS**
   - **Linux**:
     ```bash
     sudo apt-get install git-lfs
     ```
   - **macOS**:
     ```bash
     brew install git-lfs
     ```
   - **Windows**:
     Download and install Git LFS from [here](https://github.com/git-lfs/git-lfs?tab=readme-ov-file#on-windows).
2. **Clone the Repository**
   ```bash
   git clone git@github.com:alebondarenko/llm-robustness.git
   cd llm-robustness
   ```
3. **Initialize Git LFS**
   ```bash
   git lfs install
   ```
4. **Pull LFS Files**
   ```bash
   git lfs pull
   ``` 

### Installation

1. Install [Python 3.12](https://python.org/downloads/)
2. Create and activate a virtual environment:
    ```shell
    python -m venv .venv
    source .venv/bin/activate
    ```
3. Install dependencies:
    ```shell
    pip install --upgrade pip
    pip install -e .
    ```
