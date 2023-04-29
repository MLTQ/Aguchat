from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex, \
    GPTSimpleVectorIndex, PromptHelper, LLMPredictor, Document, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import torch
from langchain.llms.base import LLM
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from fastchat.conversation import conv_templates, SeparatorStyle
#!export
#PYTORCH_CUDA_ALLOC_CONF = max_split_size_mb:512
from llama_index import download_loader
from pathlib import Path
from typing import Optional, List, Mapping, Any
from langchain.llms.base import LLM
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex, PromptHelper, LLMPredictor, ServiceContext, GPTSimpleVectorIndex
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext
from configparser import ConfigParser
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
def load_vicuna():
    model_name = r"anon8231489123/vicuna-13b-GPTQ-4bit-128g"
    device = "cuda"
    num_gpus = "auto"
    wbits = 4

    # Model
    if device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
        if num_gpus != 1:
            kwargs.update({
                "device_map": "auto",
                "max_memory": {1: "13GiB" }#for i in range(num_gpus)},
            })
    elif device == "cpu":
        kwargs = {}
    else:
        raise ValueError(f"Invalid device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if wbits > 0:
        from fastchat.serve.load_gptq_model import load_quantized

        print("Loading GPTQ quantized model...")
        model = load_quantized(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     low_cpu_mem_usage=True, **kwargs)

    if device == "cuda" and num_gpus == 1:
        model.cuda()
        model.to(device)
    return model, tokenizer


data_config = ConfigParser()
data_config.read('config.ini')
max_input_size = 2048
num_output = 300
max_chunk_overlap = 102
chunk_size_limit = 600
prompt_helper = PromptHelper(max_input_size, num_output,max_chunk_overlap,chunk_size_limit=chunk_size_limit)
ObsidianReader = download_loader('ObsidianReader')
documents = ObsidianReader(data_config['data']['obsidian']).load_data()
model, tokenizer  = load_vicuna()
class CustomLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", )
        input_ids = inputs["input_ids"].cuda()
        model.cuda()
        generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.2,
        )

        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            output_scores=True,
            max_new_tokens=num_output,
        )
        response = tokenizer.decode(generation_output[0],skip_special_tokens=True).strip()
        print('Split prompt length: ')
        print(len(prompt.split(' ')))
        print('Response full: ')
        print(response)
        return response[len(prompt):]

    #def _call(self, prompt, stop=None):
    #    return self.model(prompt, max_length=2000)[0]["generated_text"]

    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    def _llm_type(self):
        return "custom"


llm_predictor = LLMPredictor(llm=CustomLLM())

embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model,
                                               prompt_helper=prompt_helper, chunk_size_limit=500)


#documents = SimpleDirectoryReader('./data').load_data()
index = GPTSimpleVectorIndex.from_documents(documents,service_context=service_context)
index.save_to_disk('index.json')


#llm = GPT4ALL_LLM()
query_text = "What is an Idol afraid of?"
response = index.query(query_text,response_mode="compact",service_context=service_context, similarity_top_k=1)
print(f'Response: {response}')
x=1