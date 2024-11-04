import json
import time

from loguru import logger
from pathlib import Path
from tqdm.auto import tqdm

from utils.data import AdversarialDocument, json_to_dataframe
from utils.generate import Generator
from utils.helper import to_list_dict


IN_FILE = Path(<path + rest_questions_adversarial_gemma9b_gemma9b.json>)
OUT_FILE = Path(<path + rest_questions_adversarial_gemma9b_gpt4o.json>)

NUM_ENTRIES = None
SLEEP = 0

generator = Generator("openai")

logger.info(f"Processing {IN_FILE.stem}...")
if OUT_FILE.exists():
    logger.info(f"File {OUT_FILE.stem} already exists. Will extend it...")
    idx = json_to_dataframe(OUT_FILE).id.tolist()
    c = 0
    with open(IN_FILE, "r") as infile, open(OUT_FILE, "a") as outfile:
        for line in tqdm(infile):
            data = json.loads(line.strip())
            if data["id"] not in idx:
                try:
                    doc = AdversarialDocument(**data)
                    doc_id = doc.get_id()
                    question = doc.get_question()
                    true_answer = doc.get_true_answer()
                    adversarial_context = doc.get_adversarial_context()
                    (
                        adversarial_answer,
                        adversarial_logprob,
                        adversarial_probability,
                        adversarial_top_logprobs,
                    ) = generator.generate_answer_with_logprobs_rest(
                        context=adversarial_context, question=question
                    )
                    outfile.write(
                        json.dumps(
                            {
                                "id": doc_id,
                                "question": question,
                                "true_answer": true_answer,
                                "adversarial_answer": adversarial_answer,
                                "adversarial_probability": adversarial_probability,
                                "adversarial_logprob": adversarial_logprob,
                                "adversarial_top_logprobs": [
                                    to_list_dict(c) for c in adversarial_top_logprobs
                                ],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    c += 1
                except Exception as e:
                    logger.error(f"Error processing document {doc_id}: {e}")
                time.sleep(SLEEP)
                if NUM_ENTRIES and c == NUM_ENTRIES:
                    break
else:
    c = 0
    with open(IN_FILE, "r") as infile, open(OUT_FILE, "w") as outfile:
        for line in tqdm(infile):
            data = json.loads(line.strip())
            try:
                doc = AdversarialDocument(**data)
                doc_id = doc.get_id()
                question = doc.get_question()
                true_answer = doc.get_true_answer()
                adversarial_context = doc.get_adversarial_context()
                (
                    adversarial_answer,
                    adversarial_logprob,
                    adversarial_probability,
                    adversarial_top_logprobs,
                ) = generator.generate_answer_with_logprobs_rest(
                    context=adversarial_context, question=question
                )
                outfile.write(
                    json.dumps(
                        {
                            "id": doc_id,
                            "question": question,
                            "true_answer": true_answer,
                            "adversarial_answer": adversarial_answer,
                            "adversarial_probability": adversarial_probability,
                            "adversarial_logprob": adversarial_logprob,
                            "adversarial_top_logprobs": [
                                to_list_dict(c) for c in adversarial_top_logprobs
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                c += 1
            except Exception as e:
                logger.error(f"Error processing document {doc_id}: {e}")
            time.sleep(SLEEP)
            if NUM_ENTRIES and c == NUM_ENTRIES:
                break
logger.info(f"Saving results to {OUT_FILE}...")
logger.info("Done.")
# import pdb; pdb.set_trace()
