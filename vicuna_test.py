from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex, \
    GPTSimpleVectorIndex, PromptHelper, LLMPredictor, Document, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import torch
from langchain.llms.base import LLM
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

#!export
#PYTORCH_CUDA_ALLOC_CONF = max_split_size_mb:512


class CustomLLM(LLM):
    model_name = "eachadea/vicuna-13b-1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0,
                        model_kwargs={"torch_dtype": torch.bfloat16})

    def _call(self, prompt, stop=None):
        return self.pipeline(prompt, max_length=2000)[0]["generated_text"]

    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    def _llm_type(self):
        return "custom"


llm_predictor = LLMPredictor(llm=CustomLLM())