import argparse
import json
import time

from loguru import logger
from pathlib import Path
from tqdm.auto import tqdm

from llm_robustness.utils.data import json_to_dataframe
from llm_robustness.utils.generate import Generator

parser = argparse.ArgumentParser()
parser.add_argument("--in_file", type=Path, required=True, help="Path to input file")
parser.add_argument("--out_file", type=Path, required=True, help="Path to output file")
parser.add_argument(
    "--generator",
    type=str,
    required=True,
    help="Name of the generator model from: openai, mixtral, llama-3.1-8b, llama-3.1-70b, gemma2-7b, gemma2-9b",
)
parser.add_argument(
    "--num_entries",
    type=int,
    default=None,
    help="Number of entries in the in_file to process; default (None) is all entries",
)
parser.add_argument(
    "--sleep",
    type=float,
    default=0,
    help="Sleep time between requests to the generator model's API in seconds; default is 0",
)
args = parser.parse_args()

IN_FILE = Path(args.in_file)
OUT_FILE = Path(args.out_file)

NUM_ENTRIES = args.num_entries
SLEEP = args.sleep

generator = Generator(args.generator)

if OUT_FILE.exists():
    idx = json_to_dataframe(OUT_FILE).id.tolist()
    c = 0
    with open(IN_FILE, "r") as infile, open(OUT_FILE, "a") as outfile:
        for line in tqdm(infile):
            data = json.loads(line.strip())
            if data["id"] not in idx:
                try:
                    reference_answer = data["true_answer"]
                    question = data["question"]
                    data["adversarial_correct"] = generator.evaluate_answer_accuracy(
                        reference_answer, data["adversarial_answer"], question
                    )
                    outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                    c += 1
                except Exception as e:
                    logger.error(f"Error processing document {data['id']}: {e}")
                time.sleep(SLEEP)
                if NUM_ENTRIES and c == NUM_ENTRIES:
                    break
else:
    c = 0
    with open(IN_FILE, "r") as infile, open(OUT_FILE, "w") as outfile:
        for line in tqdm(infile):
            data = json.loads(line.strip())
            try:
                reference_answer = data["true_answer"]
                question = data["question"]
                data["adversarial_correct"] = generator.evaluate_answer_accuracy(
                    reference_answer, data["adversarial_answer"], question
                )
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                c += 1
            except Exception as e:
                logger.error(f"Error processing document {data['id']}: {e}")
            time.sleep(SLEEP)
            if NUM_ENTRIES and c == NUM_ENTRIES:
                break
