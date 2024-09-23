import instructor
import numpy as np
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from groq import Groq
from typing import Literal

from .prompt import *

load_dotenv()

class Generator:
    def __init__(self, client_name: str):
        self.client_name = client_name
        self.client, self.model_name = self._get_client(client_name)

    def _get_client(self, client_name: str):
        if client_name == "mixtral":
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            client = instructor.from_groq(client, mode=instructor.Mode.JSON)
            return client, "mixtral-8x7b-32768"
        elif client_name == "llama-3.1-8b":
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            client = instructor.from_groq(client, mode=instructor.Mode.JSON)
            return client, "llama-3.1-8b-instant"
        elif client_name == "llama-3.1-70b":
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            client = instructor.from_groq(client, mode=instructor.Mode.JSON)
            return client, "llama-3.1-70b-versatile"
        elif client_name == "llama-3.1-405b":
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            client = instructor.from_groq(client, mode=instructor.Mode.JSON)
            return client, "llama-3.1-405b-reasoning"
        elif client_name == "gemma2-9b":
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            client = instructor.from_groq(client, mode=instructor.Mode.JSON)
            return client, "gemma2-9b-it"
        elif client_name == "gemma-7b":
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            client = instructor.from_groq(client, mode=instructor.Mode.JSON)
            return client, "gemma-7b-it"
        elif client_name == "openai":
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # wouldn't work to return logprobs, since response_model argument is required
            # but providing it means using instructor function calling, which doesn't support logprobs
            # client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)
            # will use the older patch method for now
            client = instructor.patch(client, mode=instructor.Mode.TOOLS)
            return client, "gpt-4o-mini"
        else:
            raise ValueError(f"Client {client_name} not supported")

    def generate_answer(self, context, question):
        response = self.client.chat.completions.create(
            model=self.model_name,
            response_model=YesNoAnswer,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a world-class medical professional",
                },
                {
                    "role": "user",
                    "content": SYNTHESIS_PROMPT.format(
                        context_str=context, query_str=question
                    ),
                },
            ],
        )
        return response.answer

    def generate_answer_rest(self, context, question):
        response = self.client.chat.completions.create(
            model=self.model_name,
            response_model=RestAnswer,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a world-class medical professional",
                },
                {
                    "role": "user",
                    "content": SYNTHESIS_PROMPT.format(
                        context_str=context, query_str=question
                    ),
                },
            ],
        )
        return response.answer

    def generate_answer_yesno(self, context, question):
        response = self.client.chat.completions.create(
            model=self.model_name,
            response_model=YesNoAnswer,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a world-class medical professional. You must answer the question only 'yes' or 'no'",
                },
                {
                    "role": "user",
                    "content": SYNTHESIS_PROMPT_LOGPROBS_YESNO.format(
                        context_str=context, query_str=question
                    ),
                },
            ],
        )
        return response.answer

    def generate_answer_with_logprobs(self, context, question):
        if self.client_name != "openai":
            raise ValueError("Logprobs only supported for OpenAI")
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a world-class medical professional",
                    },
                    {
                        "role": "user",
                        "content": SYNTHESIS_PROMPT_LOGPROBS_YESNO.format(
                            context_str=context, query_str=question
                        ),
                    },
                ],
                logprobs=True,
                top_logprobs=3,
                max_tokens=1,
                temperature=0,
            )
        logprobs = response.choices[0].logprobs.content
        answer = logprobs[0].token
        logprob = logprobs[0].logprob
        probability = np.round(np.exp(logprob) * 100, 2)
        top_logprobs = logprobs[0].top_logprobs

        return answer, logprob, probability, top_logprobs

    def generate_answer_with_logprobs_extended(self, context, question):
        if self.client_name != "openai":
            raise ValueError("Logprobs only supported for OpenAI")
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a world-class medical professional",
                    },
                    {
                        "role": "user",
                        "content": SYNTHESIS_PROMPT_LOGPROBS_YESNO_EXTENDED.format(
                            context_str=context, query_str=question
                        ),
                    },
                ],
                logprobs=True,
                top_logprobs=3,
                max_tokens=75,
                temperature=0,
            )
        logprobs = [token.logprob for token in response.choices[0].logprobs.content]
        answer = response.choices[0].message.content
        probabilities = [np.round(np.exp(logprob) * 100, 2) for logprob in logprobs]
        top_logprobs = [c.top_logprobs for c in response.choices[0].logprobs.content]

        return answer, logprobs, probabilities, top_logprobs

    def generate_answer_with_logprobs_rest(self, context, question):
        if self.client_name != "openai":
            raise ValueError("Logprobs only supported for OpenAI")
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a world-class medical professional",
                    },
                    {
                        "role": "user",
                        "content": SYNTHESIS_PROMPT.format(
                            context_str=context, query_str=question
                        ),
                    },
                ],
                logprobs=True,
                top_logprobs=3,
                max_tokens=75,
                temperature=0,
            )
        logprobs = [token.logprob for token in response.choices[0].logprobs.content]
        answer = response.choices[0].message.content
        probabilities = [np.round(np.exp(logprob) * 100, 2) for logprob in logprobs]
        top_logprobs = [c.top_logprobs for c in response.choices[0].logprobs.content]

        return answer, logprobs, probabilities, top_logprobs

    def generate_vanilla_answer_with_logprobs(self, question):
        if self.client_name != "openai":
            raise ValueError("Logprobs only supported for OpenAI")
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a world-class medical professional",
                    },
                    {
                        "role": "user",
                        "content": VANILLA_PROMPT_LOGPROBS_YESNO.format(
                            query_str=question
                        ),
                    },
                ],
                logprobs=True,
                top_logprobs=3,
                max_tokens=1,
                temperature=0,
            )
        logprobs = response.choices[0].logprobs.content
        answer = logprobs[0].token
        logprob = logprobs[0].logprob
        probability = np.round(np.exp(logprob) * 100, 2)
        top_logprobs = logprobs[0].top_logprobs

        return answer, logprob, probability, top_logprobs

    def generate_vanilla_answer_with_logprobs_extended(self, question):
        if self.client_name != "openai":
            raise ValueError("Logprobs only supported for OpenAI")
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a world-class medical professional",
                    },
                    {
                        "role": "user",
                        "content": VANILLA_PROMPT_LOGPROBS_YESNO_EXTENDED.format(
                            query_str=question
                        ),
                    },
                ],
                logprobs=True,
                top_logprobs=3,
                max_tokens=75,
                temperature=0,
            )
        logprobs = [token.logprob for token in response.choices[0].logprobs.content]
        answer = response.choices[0].message.content
        probabilities = [np.round(np.exp(logprob) * 100, 2) for logprob in logprobs]
        top_logprobs = [c.top_logprobs for c in response.choices[0].logprobs.content]

        return answer, logprobs, probabilities, top_logprobs

    def generate_vanilla_answer_with_logprobs_rest(self, question):
        if self.client_name != "openai":
            raise ValueError("Logprobs only supported for OpenAI")
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a world-class medical professional",
                    },
                    {
                        "role": "user",
                        "content": VANILLA_PROMPT_LOGPROBS.format(query_str=question),
                    },
                ],
                logprobs=True,
                top_logprobs=3,
                max_tokens=75,
                temperature=0,
            )
        logprobs = [token.logprob for token in response.choices[0].logprobs.content]
        answer = response.choices[0].message.content
        probabilities = [np.round(np.exp(logprob) * 100, 2) for logprob in logprobs]
        top_logprobs = [c.top_logprobs for c in response.choices[0].logprobs.content]

        return answer, logprobs, probabilities, top_logprobs

    def generate_vanilla_answer(self, question):
        response = self.client.chat.completions.create(
            model=self.model_name,
            response_model=YesNoAnswer,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a world-class medical professional. Answer the question.",
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
        )
        return response.answer

    def generate_vanilla_answer_rest(self, question):
        response = self.client.chat.completions.create(
            model=self.model_name,
            response_model=RestAnswer,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a world-class medical professional. Answer the question.",
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
        )
        return response.answer

    def generate_vanilla_answer_yesno(self, question):
        response = self.client.chat.completions.create(
            model=self.model_name,
            response_model=YesNoAnswer,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a world-class medical professional. You must answer the question only 'yes' or 'no'.",
                },
                {
                    "role": "user",
                    "content": VANILLA_PROMPT_LOGPROBS_YESNO.format(query_str=question),
                },
            ],
        )
        return response.answer

    def generate_vanilla_answer_with_completion(self, question):
        response, completion = self.client.chat.completions.create_with_completion(
            model=self.model_name,
            response_model=YesNoAnswer,
            temperature=0,
            logprobs=True,
            top_logprobs=3,
            messages=[
                {
                    "role": "system",
                    "content": "You are a world-class medical professional. Answer the question.",
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
        )
        return response.answer, completion

    def generate_adversarial_context(self, context, question, answer):
        answer_map = {"yes": "no", "no": "yes"}
        answer_str = answer_map.get(answer)
        if not answer_str:
            raise ValueError(f"Invalid answer: {answer} | must be 'yes' or 'no'")
        response = self.client.chat.completions.create(
            model=self.model_name,
            response_model=AdversarialContext,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a world-class player in the game Write a wrong context",
                },
                {
                    "role": "user",
                    "content": ADVERSARIAL_PROMPT.format(
                        context_str=context, query_str=question, answer_str=answer_str
                    ),
                },
            ],
        )
        return response.context

    def generate_adversarial_context_rest(self, context, question, answer):
        response = self.client.chat.completions.create(
            model=self.model_name,
            response_model=AdversarialContext,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a world-class player in the game Write a wrong context",
                },
                {
                    "role": "user",
                    "content": ADVERSARIAL_PROMPT.format(
                        context_str=context, query_str=question, answer_str=answer
                    ),
                },
            ],
        )
        return response.context

    def generate_wrong_answer(self, correct_answer, question):
        response = self.client.chat.completions.create(
            model=self.model_name,
            response_model=WrongAnswer,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are playing the game Give me the wrong answer.",
                },
                {
                    "role": "user",
                    "content": WRONG_ANSWER_PROMPT.format(
                        correct_answer_str=correct_answer, query_str=question
                    ),
                },
            ],
        )
        return response.wrong_answer

    def evaluate_answer_accuracy(self, reference_answer, predicted_answer, question):
        response = self.client.chat.completions.create(
            model=self.model_name,
            response_model=YesNoAnswer,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a world-class answer accuracy assessor.",
                },
                {
                    "role": "user",
                    "content": ACCURACY_EVALUATION_PROMPT.format(
                        reference_answer=reference_answer,
                        predicted_answer=predicted_answer,
                        query_str=question,
                    ),
                },
            ],
        )
        return response.answer


class WrongAnswer(BaseModel):
    wrong_answer: str = Field(..., description="The wrong answer to the question.")


class RestAnswer(BaseModel):
    answer: str = Field(..., description="Answer to the question.")


class YesNoAnswer(BaseModel):
    answer: Literal["yes", "no"]


class AdversarialContext(BaseModel):
    context: str = Field(
        ..., description="The context that supports the answer to the question."
    )


class YesNoAnswerFlex(BaseModel):
    answer: str = Field(..., description="The answer to the question.")
