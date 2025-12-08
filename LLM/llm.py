from langchain_community.llms import LlamaCpp
from pathlib import Path

def import_llm() :
    here = Path(__file__).resolve().parent 
    model_path = here / "Qwen3-1.7B-Q8_0.gguf"
    llm = LlamaCpp(
        model_path=str(model_path),
        n_threads=4,
        temperature=0.5,
        top_k=20,
        top_p=0.8,
        n_ctx = 8040,
        max_tokens=2048,
        verbose=True,
        repeat_penalty = 1.5
    )
    return llm
