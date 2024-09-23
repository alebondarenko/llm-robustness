import json

from utils.data import Document, AdversarialDocument, row_to_dict


def to_list_dict(data):
    res = []
    for d in data:
        res.append({"token": d.token, "bytes": d.bytes, "logprob": d.logprob})
    return res


def generate_results_with_logprobs(data, df_with_snippets, generator, outfile):
    doc = AdversarialDocument(**data)
    doc_id = doc.get_id()
    question = doc.get_question()
    true_answer = doc.get_true_answer()
    (
        vanilla_answer,
        vanilla_logprob,
        vanilla_probability,
        vanilla_top_logprobs,
    ) = generator.generate_vanilla_answer_with_logprobs(question=question)
    doc_snippet = Document(**row_to_dict(df_with_snippets, doc_id))
    (
        predicted_answer,
        predicted_logprob,
        predicted_probability,
        predicted_top_logprobs,
    ) = generator.generate_answer_with_logprobs(
        context=doc_snippet.get_snippets(), question=question
    )
    adversarial_context = doc.get_adversarial_context()
    (
        adversarial_answer,
        adversarial_logprob,
        adversarial_probability,
        adversarial_top_logprobs,
    ) = generator.generate_answer_with_logprobs(
        context=adversarial_context, question=question
    )
    outfile.write(
        json.dumps(
            {
                "id": doc_id,
                "question": question,
                "true_answer": true_answer,
                "vanilla_answer": vanilla_answer,
                "vanilla_probability": vanilla_probability,
                "predicted_answer": predicted_answer,
                "predicted_probability": predicted_probability,
                "adversarial_answer": adversarial_answer,
                "adversarial_probability": adversarial_probability,
                "adversarial_context": adversarial_context,
                "vanilla_logprob": vanilla_logprob,
                "predicted_logprob": predicted_logprob,
                "adversarial_logprob": adversarial_logprob,
                "vanilla_top_logprobs": to_list_dict(vanilla_top_logprobs),
                "predicted_top_logprobs": to_list_dict(predicted_top_logprobs),
                "adversarial_top_logprobs": to_list_dict(adversarial_top_logprobs),
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    return None


def generate_results_with_logprobs_extended(data, df_with_snippets, generator, outfile):
    doc = AdversarialDocument(**data)
    doc_id = doc.get_id()
    question = doc.get_question()
    true_answer = doc.get_true_answer()
    (
        vanilla_answer,
        vanilla_logprob,
        vanilla_probability,
        vanilla_top_logprobs,
    ) = generator.generate_vanilla_answer_with_logprobs_extended(question=question)
    doc_snippet = Document(**row_to_dict(df_with_snippets, doc_id))
    true_answer_extended = doc_snippet.get_ideal_answer()
    (
        predicted_answer,
        predicted_logprob,
        predicted_probability,
        predicted_top_logprobs,
    ) = generator.generate_answer_with_logprobs_extended(
        context=doc_snippet.get_snippets(), question=question
    )
    adversarial_context = doc.get_adversarial_context()
    (
        adversarial_answer,
        adversarial_logprob,
        adversarial_probability,
        adversarial_top_logprobs,
    ) = generator.generate_answer_with_logprobs_extended(
        context=adversarial_context, question=question
    )
    outfile.write(
        json.dumps(
            {
                "id": doc_id,
                "question": question,
                "true_answer": true_answer,
                "true_answer_extended": true_answer_extended,
                "vanilla_answer": vanilla_answer,
                "vanilla_probability": vanilla_probability,
                "predicted_answer": predicted_answer,
                "predicted_probability": predicted_probability,
                "adversarial_answer": adversarial_answer,
                "adversarial_probability": adversarial_probability,
                "adversarial_context": adversarial_context,
                "vanilla_logprob": vanilla_logprob,
                "predicted_logprob": predicted_logprob,
                "adversarial_logprob": adversarial_logprob,
                "vanilla_top_logprobs": [to_list_dict(c) for c in vanilla_top_logprobs],
                "predicted_top_logprobs": [
                    to_list_dict(c) for c in predicted_top_logprobs
                ],
                "adversarial_top_logprobs": [
                    to_list_dict(c) for c in adversarial_top_logprobs
                ],
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    return None
