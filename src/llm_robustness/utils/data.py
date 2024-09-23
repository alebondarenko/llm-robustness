import json
import pandas as pd

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Document:
    """Class to represent a document"""

    body: str
    exact_answer: str
    ideal_answer: str
    snippets: List[Dict]
    additional_data: dict = field(default_factory=dict)

    def __init__(
        self,
        body: str,
        exact_answer: str,
        ideal_answer: str,
        snippets: List[Dict],
        **kwargs
    ):
        self.body = body
        self.exact_answer = exact_answer
        self.ideal_answer = ideal_answer
        self.snippets = snippets
        self.additional_data = kwargs

    def get_id(self):
        return self.additional_data.get("id", None)

    def get_question(self):
        return self.body

    def get_exact_answer(self):
        return self.exact_answer

    def get_ideal_answer(self):
        return self.ideal_answer

    def get_snippets(self, top_k: int = 5):
        return "\n".join([d["text"] for d in self.snippets[:top_k]])


@dataclass
class RestDocument:
    """Class to represent a document"""

    body: str
    ideal_answer: str
    snippets: List[Dict]
    additional_data: dict = field(default_factory=dict)

    def __init__(
        self,
        body: str,
        ideal_answer: str,
        snippets: List[Dict],
        **kwargs,
    ):
        self.body = body
        self.ideal_answer = ideal_answer
        self.snippets = snippets
        self.additional_data = kwargs

    def get_id(self):
        return self.additional_data.get("id", None)

    def get_question(self):
        return self.body

    def get_exact_answer(self):
        return self.additional_data.get("exact_answer", None)

    def get_ideal_answer(self):
        return self.ideal_answer

    def get_snippets(self, top_k: int = 5):
        return "\n".join([d["text"] for d in self.snippets[:top_k]])


@dataclass
class AdversarialDocument:
    """Class to represent an adversarial document"""

    additional_data: dict = field(default_factory=dict)

    def __init__(self, **kwargs):
        self.additional_data = kwargs

    def get_id(self):
        return self.additional_data.get("id", None)

    def get_question(self):
        return self.additional_data.get("question", None)

    def get_true_answer(self):
        return self.additional_data.get("true_answer", None)

    def get_vanilla_answer(self):
        return self.additional_data.get("vanilla_answer", None)

    def get_predicted_answer(self):
        return self.additional_data.get("predicted_answer", None)

    def get_adversarial_answer(self):
        return self.additional_data.get("adversarial_answer", None)

    def get_adversarial_context(self):
        return self.additional_data.get("adversarial_context", None)


def json_to_dataframe(input_file):
    data = []
    with open(input_file, "r") as infile:
        for line in infile:
            data.append(json.loads(line.strip()))

    df = pd.DataFrame(data)
    return df


def row_to_dict(dataframe, row_id):
    row = dataframe.loc[dataframe["id"] == row_id]
    if not row.empty:
        return row.iloc[0].to_dict()
    else:
        raise ValueError(f"Row with id {row_id} not found in dataframe")
