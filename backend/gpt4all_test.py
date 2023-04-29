
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import textwrap
peft_model_id = "nomic-ai/gpt4all-lora"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
gpt4all_model = PeftModel.from_pretrained(model, peft_model_id)


from llama_index import download_loader
from pathlib import Path
from typing import Optional, List, Mapping, Any
from langchain.llms.base import LLM
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex, PromptHelper, LLMPredictor, ServiceContext, GPTSimpleVectorIndex
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext
from configparser import ConfigParser
data_config = ConfigParser()
data_config.read('config.ini')
max_input_size = 2048
num_output = 300
max_chunk_overlap = 102
chunk_size_limit = 600
prompt_helper = PromptHelper(max_input_size, num_output,max_chunk_overlap,chunk_size_limit=chunk_size_limit)
ObsidianReader = download_loader('ObsidianReader')
documents = ObsidianReader(data_config['data']['obsidian']).load_data()

class GPT4ALL_LLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", )
        input_ids = inputs["input_ids"].cuda()
        generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.2,
        )

        generation_output = gpt4all_model.generate(
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
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": "GPT4ALL"}
    @property
    def _llm_type(self) -> str:
        return "custom"


llm_predictor = LLMPredictor(llm=GPT4ALL_LLM())
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model,
                                               prompt_helper=prompt_helper, chunk_size_limit=500)


#documents = SimpleDirectoryReader('./data').load_data()
index = GPTSimpleVectorIndex.from_documents(documents,)#service_context=service_context)
index.save_to_disk('index.json')


#llm = GPT4ALL_LLM()
query_text = "What is an Idol afraid of?"
response = index.query(query_text,response_mode="compact",service_context=service_context, similarity_top_k=1)
print(f'Response: {response}')
x=1