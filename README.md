# LLM Robustness Against Misinformation in Biomedical Question Answering

Code and resources that implement experiments for the paper [_LLM Robustness Against Misinformation in Biomedical Question Answering_](https://arxiv.org/abs/2410.21330).

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

### Code
To replicate the results, you can directly use the _notebooks_ that process the files in the _results_ directory.

To repeat all the steps, use the _scripts_:

1. For binary questions: TODO
2. For free-form questions, there are generally three types of scripts:
   - `main_<model_name>_<model_name>_rest.py`  (model_name: mixtral, llama, gemma, or gpt-4o): takes the input data, and generates adversarial contexts and answers by the same model (adversarial model == target model).
   - `main_<model_name>_model_rest.py`  will use the resulting file produced in the previous step and will generate the answers by the target models (target model != adversarial model and != gpt-4o).
   - `main_<model_name>_gpt4o_rest.py`  will use gpt4o as a target model (additionally saves the logprobs from gpt-4o).       

### Citation
```
@misc{bondarenko:2024,
      title={LLM Robustness Against Misinformation in Biomedical Question Answering}, 
      author={Alexander Bondarenko and Adrian Viehweger},
      year={2024},
      eprint={2410.21330},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.21330}, 
}
```
