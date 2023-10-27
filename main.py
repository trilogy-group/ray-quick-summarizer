from time import perf_counter
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from ray import serve
from starlette.requests import Request
import tiktoken
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "pszemraj/led-large-book-summary"


def count_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = len(encoding.encode(text))
    return tokens


def split_into_chunks(
    text: str, chunk_size: int = 512, chunk_overlap: int = 2
) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        length_function=count_tokens,
        separators=["\n\n", "\n", " ", ""],
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(text)
    return chunks


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
            chunks = split_into_chunks(text)
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
