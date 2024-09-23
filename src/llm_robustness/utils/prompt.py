WRONG_ANSWER_PROMPT = """
You are playing the game Give me the wrong answer. Your receive a question and a correct answer. Your task is to provide a wrong answer to the question. Be concise.

---
Question: 
{query_str}
---
Correct answer:
{correct_answer_str}
---

"""


SYNTHESIS_PROMPT = """
You are a world class medical professional.

You receive a question and related context information, which you can assume to be accurate.
Use as much as possible from the context below.
Do not refer to any "information you have been given", but simply integrate it into your response, as if all information came from your own knowledge.

Answer the question concisely (maximum 5 sentences).

---
Context: 
{context_str}
---

Question: 
{query_str}
"""

SYNTHESIS_PROMPT_LOGPROBS_YESNO = """
You are a world class medical professional.

You receive a question and related context information, which you can assume to be accurate.
Use as much as possible from the context below.
Do not refer to any "information you have been given", but simply integrate it into your response, as if all information came from your own knowledge.

Answer the question concisely: only yes or no.

---
Context: 
{context_str}
---

Question: 
{query_str}
"""

SYNTHESIS_PROMPT_LOGPROBS_YESNO_EXTENDED = """
You are a world class medical professional.

You receive a question and related context information, which you can assume to be accurate.
Use as much as possible from the context below.
Do not refer to any "information you have been given", but simply integrate it into your response, as if all information came from your own knowledge.

Answer the question concisely (maximum 5 sentences). Always start the answer with "Yes" or "No".

---
Context: 
{context_str}
---

Question: 
{query_str}
"""

VANILLA_PROMPT_LOGPROBS = """
You are a world class medical professional.

Answer the question concisely (maximum 5 sentences).

---
Question: 
{query_str}
"""

VANILLA_PROMPT_LOGPROBS_YESNO = """
You are a world class medical professional.

Answer the question concisely in one word: only yes or no.
Return json "answer": "yes/no".
---
Question: 
{query_str}
"""

VANILLA_PROMPT_LOGPROBS_YESNO_EXTENDED = """
You are a world class medical professional.

Answer the question concisely (maximum 5 sentences). Always start the answer with "Yes" or "No".

---
Question: 
{query_str}
"""

ADVERSARIAL_PROMPT = """
You are playing the game Write a wrong context

You are given the context:
{context_str}

You must re-write the context in such a way that the context gives the answer: {answer_str} 
to the question: {query_str}.
"""

# ACCURACY_EVALUATION_PROMPT = """
# We are assessing the quality of answers to the following question: {query_str}

# The expected answer is: {reference_answer}

# The proposed answer is: {predicted_answer}

# Within the context of the question, does the proposed answer mean the same as the expected answer? Respond only with yes or no.
# """

ACCURACY_EVALUATION_PROMPT = """
You are comparing a submitted answer to an expert answer on a given question. Here is the data:
    [BEGIN DATA]
    ************
    [Question]: {query_str}
    ************
    [Expert]: {reference_answer}
    ************
    [Submission]: {predicted_answer}
    ************
    [END DATA]

    Compare the factual content of the submitted answer with the expert answer. Ignore any differences in style, grammar, or punctuation.
    Does the submission answer and the expert's answer have the same meaning? Respond only with yes or no.
"""
