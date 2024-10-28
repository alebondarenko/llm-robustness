# LLM Robustness Against Misinformation in Biomedical Question Answering

Code and resources that implement experiments for the paper _LLM Robustness Against Misinformation in Biomedical Question Answering_.

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
### Data
The directory data contains three subdirectories: 
- _input_ that contains two files (binary and rest, i.e. free-form, questions).
- _adversarial_ that contains the files with generated adversarial (wrong) answers only for free-form questions.
- _results_ that contains final files
Each JSON file in the subdirectories is zipped and needs to be unzipped.
