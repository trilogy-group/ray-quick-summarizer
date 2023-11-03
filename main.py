import re
from time import perf_counter
from typing import List

from ray import serve
from starlette.requests import Request
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "pszemraj/led-large-book-summary"


def count_tokens(text: str, tokenizer) -> int:
    tokens = tokenizer(text, truncation=False)["input_ids"]
    return len(tokens)


def split_with_separator(text: str, separator: str) -> List[str]:
    chunks = re.split(separator, text)
    for i in range(len(chunks) - 1):
        chunks[i] += separator
    return chunks


def merge_splits(
    splits: List[str], chunk_size: int, overlap: int, tokenizer
) -> List[str]:
    # We now want to combine these smaller pieces into medium size
    # chunks to send to the LLM.

    docs = []
    current_doc: List[str] = []
    total = 0
    for d in splits:
        length = count_tokens(d, tokenizer)
        if total + length > chunk_size:
            if total > chunk_size:
                print(
                    f"Created a chunk of size {total}, "
                    f"which is longer than the specified {chunk_size}"
                )
            if len(current_doc) > 0:
                doc = "".join(current_doc)
                if doc:
                    docs.append(doc)
                # Keep on popping if:
                # - we have a larger chunk than in the chunk overlap
                # - or if we still have any chunks and the length is long
                while total > overlap or (total + length > chunk_size and total > 0):
                    total -= count_tokens(current_doc[0], tokenizer)
                    current_doc = current_doc[1:]
        current_doc.append(d)
        total += length
    doc = "".join(current_doc)
    if doc:
        docs.append(doc)
    return docs


def split_text(
    text: str,
    tokenizer,
    separators: List[str] = ["\n\n", "\n", " ", ""],
    chunk_size: int = 512,
    overlap: int = 2,
):
    final_chunks = []
    # Get appropriate separator to use
    separator = separators[-1]
    new_separators = []
    for i, sep in enumerate(separators):
        if sep == "":
            separator = sep
            break
        if sep in text:
            separator = sep
            new_separators = separators[i + 1 :]
            break
    splits = split_with_separator(text, separator)

    # Now go merging things, recursively splitting longer texts.
    good_splits = []
    for s in splits:
        if count_tokens(s, tokenizer) < chunk_size:
            good_splits.append(s)
        else:
            if good_splits:
                merged_text = merge_splits(good_splits, chunk_size, overlap, tokenizer)
                final_chunks.extend(merged_text)
                good_splits = []
            if not new_separators:
                final_chunks.append(s)
            else:
                other_info = split_text(s, new_separators)
                final_chunks.extend(other_info)
    if good_splits:
        merged_text = merge_splits(good_splits, chunk_size, overlap, tokenizer)
        final_chunks.extend(merged_text)
    return final_chunks


# TODO: Find ideal cpu and gpu config
@serve.deployment(
    ray_actor_options={"num_cpus": 1, "num_gpus": 1},
    autoscaling_config={
        "target_num_ongoing_requests_per_replica": 5,
        "min_replicas": 0,
        "initial_replicas": 0,
        "max_replicas": 200,
    },
)
class Summarizer:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("Loading model")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(self.device)

    def summarize(self, text: str) -> str:
        print("Starting summary")
        with torch.inference_mode():
            start = perf_counter()
            chunks = split_text(text, self.tokenizer)
            input_ids = self.tokenizer(
                chunks, padding=True, truncation=True, return_tensors="pt"
            ).input_ids
            outputs = self.model.generate(
                input_ids.to(self.device),
                min_length=0,
                max_new_tokens=sum(map(len, input_ids)),
            )
            summary = "".join(
                self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            )
            end = perf_counter()
            print(f"Summarized {len(chunks)} chunks in {end-start:0.2f}s")
            return summary

    async def __call__(self, http_request: Request) -> str:
        text: str = (await http_request.json())["text"]
        return self.summarize(text)


summarizer_app = Summarizer.bind()
