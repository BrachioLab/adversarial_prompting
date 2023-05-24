
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, pipeline
import abc
from typing import List
import torch

#TODO: MODIFY RETURN VALUES
 
from utils.constants import PATH_TO_LLAMA_WEIGHTS, PATH_TO_LLAMA_TOKENIZER, PATH_TO_VICUNA

class LanguageModel():
    def __init__(self, max_gen_length, num_gen_seq):
        self.max_gen_length = max_gen_length
        self.num_gen_seq = num_gen_seq
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device_num = 0 if torch.cuda.is_available() else -1

    @abc.abstractmethod
    def load_model(self, model_name: str, **kwargs):
        """Load the language model."""
        pass

    def generate_text(self, prompts: List[str], seed_text = None) -> List[str]:
        """Generate text given a prompts."""
        if seed_text is not None:
            prompts = [cur_prompt + " " + seed_text for cur_prompt in prompts]
        return self._generate(prompts)

    @abc.abstractmethod
    def _generate(self, prompts:  List[str]) -> List[str]:
        pass

    @abc.abstractmethod
    def get_vocab_tokens(self) -> List[str]:
        '''
        Returns the vocabulary of the language model
        Necessary so that the tokenizer only searches over valid tokens
        '''
        pass

class HuggingFaceLanguageModel(LanguageModel):
    def load_model(self, model_name: str):
        valid_model_names = ["facebook/opt-125m","facebook/opt-350m","facebook/opt-1.3b","facebook/opt-2.7b", "facebook/opt-6.7b", "facebook/opt-13b", "facebook/opt-30b","gpt2",]
        assert model_name in valid_model_names, f"model name was {model_name} but must be one of {valid_model_names}"
        self.model_name = model_name
        self.generator = pipeline("text-generation", model=model_name, device=self.device_num)

    def _generate(self, prompts: List[str]) -> List[str]:
        gen_texts = self.generator(prompts, max_length=self.max_gen_length, num_return_sequences=self.num_gen_seq, top_k=50, do_sample = True)
        gen_texts = [(prompts[i],[cur_dict['generated_text'][len(prompts[i]):] for cur_dict in gen]) for i,gen in enumerate(gen_texts)]
        return gen_texts

    def get_vocab_tokens(self):

        lm_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
        return list(lm_tokenizer.get_vocab().keys())

class LlamaHuggingFace(LanguageModel):
    def load_model(self, model_name: str):
        self.model_name = model_name
        from transformers import LlamaForCausalLM, LlamaTokenizer
       
        valid_model_names = ["llama-7b","llama-13b","llama-30b"]
        assert model_name in valid_model_names, f"model name was {model_name} but must be one of {valid_model_names}"
        self.model = LlamaForCausalLM.from_pretrained(PATH_TO_LLAMA_WEIGHTS + model_name).to(self.torch_device)
        self.tokenizer = LlamaTokenizer.from_pretrained(PATH_TO_LLAMA_TOKENIZER, padding_side='left')
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))

    def _generate(self, prompts: List[str]) -> List[str]:
        input_ids = self.tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to(self.torch_device)
        gen_ids = self.model.generate(input_ids, max_length=self.max_gen_length, num_return_sequences=self.num_gen_seq, top_k=50, do_sample = True, pad_token_id=self.tokenizer.pad_token_id)
        gen_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        
        gen_texts = [(prompts[i],[gen_texts[i * self.num_gen_seq + j][len(prompts[i]):] for j in range(self.num_gen_seq)]) for i in range(len(prompts))]
        return gen_texts

    def get_vocab_tokens(self):
        return list(self.tokenizer.get_vocab().keys())

class StableHuggingFace(LanguageModel):
    def load_model(self, model_name: str):
        self.model_name = model_name
        
        valid_model_names = ["stablelm"]
        assert model_name in valid_model_names, f"model name was {model_name} but must be one of {valid_model_names}"
        model_path = "StabilityAI/stablelm-tuned-alpha-7b"
        model_path = "eachadea/vicuna-13b-1.1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.torch_device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.half().cuda()
        self.system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
        - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
        - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
        - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
        - StableLM will refuse to participate in anything that could harm a human.
        """

        
    def generate_text(self, prompts: List[str], seed_text = None) -> List[str]:
        """Generate text given a prompts."""
        prepended_prompts = prompts
        if seed_text is not None:
            prepended_prompts = [cur_prompt + " " + seed_text for cur_prompt in prepended_prompts]
        processed_prompts = [f"{self.system_prompt}<|USER|>{cur_prompt}<|ASSISTANT|>" for cur_prompt in prepended_prompts]

        inputs = self.tokenizer(processed_prompts, return_tensors="pt",padding=True).to(self.torch_device)
        tokens = self.model.generate(
            **inputs,
            max_new_tokens=self.max_gen_length,
            temperature=0.7,
            do_sample=True,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
            num_return_sequences=self.num_gen_seq,pad_token_id=self.tokenizer.pad_token_id
            )
        
        completion_tokens = tokens[:,inputs['input_ids'].size(1):]
        gen_texts = self.tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)

        processed_gen_texts = []
        for i in range(len(prompts)):
            prompt_gen_texts = [gen_texts[i * self.num_gen_seq + j] for j in range(self.num_gen_seq)]
            processed_gen_texts.append((prepended_prompts[i],prompt_gen_texts))
        return processed_gen_texts

    def get_vocab_tokens(self):
        return list(self.tokenizer.get_vocab().keys())

class VicunaHuggingFace(LanguageModel):
    def load_model(self, model_name: str):
        self.model_name = model_name
        
        valid_model_names = ["vicuna1.1"]
        assert model_name in valid_model_names, f"model name was {model_name} but must be one of {valid_model_names}"
        self.model = AutoModelForCausalLM.from_pretrained(PATH_TO_VICUNA).to(self.torch_device)
        self.tokenizer = AutoTokenizer.from_pretrained(PATH_TO_VICUNA, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.half().cuda()
    def generate_text(self, prompts: List[str], seed_text = None) -> List[str]:
        prepended_prompts = prompts
        if seed_text is not None:
            prepended_prompts = [cur_prompt + " " + seed_text for cur_prompt in prepended_prompts]
        processed_prompts = [f"Human: {cur_prompt} ASSISTANT:" for cur_prompt in prepended_prompts]
        input_ids = self.tokenizer(processed_prompts, return_tensors="pt", padding=True).input_ids.to(self.torch_device)
        gen_ids = self.model.generate(input_ids, max_new_tokens=self.max_gen_length, num_return_sequences=self.num_gen_seq, top_k=50, do_sample = True, pad_token_id=self.tokenizer.pad_token_id, temperature=0.7)
        gen_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        
        gen_texts = [(prompts[i],[gen_texts[i * self.num_gen_seq + j][len(processed_prompts[i]):] for j in range(self.num_gen_seq)]) for i in range(len(prompts))]
        return gen_texts

    def get_vocab_tokens(self):
        return list(self.tokenizer.get_vocab().keys())
    

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False