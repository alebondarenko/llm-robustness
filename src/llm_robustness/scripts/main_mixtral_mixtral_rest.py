import json
import time

from loguru import logger
from pathlib import Path
from tqdm.auto import tqdm

from utils.data import RestDocument, json_to_dataframe
from utils.generate import Generator


IN_FILE = Path("../data/bioasq-data/rest_questions.json")
OUT_FILE = Path("temp/rest_questions_adversarial_mixtral_mixtral.json")

NUM_ENTRIES = None
SLEEP = 0

generator = Generator("mixtral")

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
                    doc = RestDocument(**data)
                    doc_id = doc.get_id()
                    question = doc.get_question()
                    true_answer_ideal = doc.get_ideal_answer()
                    wrong_answer_ideal = generator.generate_wrong_answer(
                        true_answer_ideal, question
                    )
                    adversarial_context = generator.generate_adversarial_context_rest(
                        doc.get_snippets(), question, wrong_answer_ideal
                    )
                    vanilla_answer = generator.generate_vanilla_answer_rest(question=question)
                    predicted_answer = generator.generate_answer_rest(
                        context=doc.get_snippets(), question=question
                    )
                    adversarial_answer = generator.generate_answer_rest(
                        context=adversarial_context, question=question
                    )
                    outfile.write(
                        json.dumps(
                            {
                                "id": doc_id,
                                "question": question,
                                "true_answer": true_answer_ideal,
                                "vanilla_answer": vanilla_answer,
                                "predicted_answer": predicted_answer,
                                "adversarial_answer": adversarial_answer,
                                "adversarial_context": adversarial_context,
                                "wrong_answer_ideal": wrong_answer_ideal,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    c += 1
                except Exception as e:
                    logger.error(f"Error processing document {doc_id}: {e}")
                    time.sleep(1)
                time.sleep(SLEEP)
                if NUM_ENTRIES and c == NUM_ENTRIES:
                    break

else:
    c = 0
    with open(IN_FILE, "r") as infile, open(OUT_FILE, "w") as outfile:
        for line in tqdm(infile):
            data = json.loads(line.strip())
            try:
                doc = RestDocument(**data)
                doc_id = doc.get_id()
                question = doc.get_question()
                true_answer_ideal = doc.get_ideal_answer()
                wrong_answer_ideal = generator.generate_wrong_answer(
                    true_answer_ideal, question
                )
                adversarial_context = generator.generate_adversarial_context_rest(
                    doc.get_snippets(), question, wrong_answer_ideal
                )
                vanilla_answer = generator.generate_vanilla_answer_rest(question=question)
                predicted_answer = generator.generate_answer_rest(
                    context=doc.get_snippets(), question=question
                )
                adversarial_answer = generator.generate_answer_rest(
                    context=adversarial_context, question=question
                )
                outfile.write(
                    json.dumps(
                        {
                            "id": doc_id,
                            "question": question,
                            "true_answer": true_answer_ideal,
                            "vanilla_answer": vanilla_answer,
                            "predicted_answer": predicted_answer,
                            "adversarial_answer": adversarial_answer,
                            "adversarial_context": adversarial_context,
                            "wrong_answer_ideal": wrong_answer_ideal,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                c += 1
            except Exception as e:
                logger.error(f"Error processing document {doc_id}: {e}")
                time.sleep(1)
            time.sleep(SLEEP)
            if NUM_ENTRIES and c == NUM_ENTRIES:
                break

logger.info(f"Saving results to {OUT_FILE}...")
logger.info("Done.")
# import pdb; pdb.set_trace()
