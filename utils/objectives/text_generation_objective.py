from transformers import GPT2Tokenizer, OPTModel, pipeline
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import sys 
sys.path.append("../")
from utils.objective import Objective 


class TextGenerationObjective(Objective):
    def __init__(
        self,
        num_calls=0,
        n_tokens=1,
        minimize=False,
        batch_size=10,
        prepend_to_text="",
        num_gen_seq=5,
        max_gen_length=10,
        dist_metric="sq_euclidean",
        lb=None,
        ub=None,
        text_gen_model="opt",
        loss_type="log_prob_neg", # log_prob_neg, log_prob_pos
        target_string="t",
        **kwargs,
    ):
        super().__init__(
            num_calls=num_calls,
            task_id='adversarial4',
            dim=n_tokens*768,
            lb=lb,
            ub=ub,
            **kwargs,
        ) 
        assert dist_metric in ['cosine_sim', "sq_euclidean"]
        # find models here: https://huggingface.co/models?sort=downloads&search=facebook%2Fopt
        if text_gen_model == "opt":
            model_string = "facebook/opt-125m"
        elif text_gen_model == "opt13b":
            model_string = "facebook/opt-13b"
        elif text_gen_model == "opt66b":
            model_string = "facebook/opt-66b"
        elif text_gen_model == "opt350": 
            model_string = "facebook/opt-350m" 
        elif text_gen_model == "gpt2":
            model_string = "gpt2"
        else:
            assert 0 

        self.target_string = target_string 
        self.loss_type = loss_type 
        self.prepend_to_text = prepend_to_text
        self.N_extra_prepend_tokens = len(self.prepend_to_text.split() )
        self.dist_metric = dist_metric 
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_string)
        self.distilBert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.distilBert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.generator = pipeline("text-generation", model=model_string)
        self.model = OPTModel.from_pretrained(model_string) 
        self.model = self.model.to(self.torch_device)
        self.word_embedder = self.model.get_input_embeddings()
        self.vocab = self.tokenizer.get_vocab()
        self.num_gen_seq = num_gen_seq
        self.max_gen_length = max_gen_length + n_tokens + self.N_extra_prepend_tokens 
       
        if self.loss_type not in ['log_prob_pos', 'log_prob_neg']: 
            self.related_vocab = [self.target_string]
            self.all_token_idxs = self.get_non_related_values() 
        else:
            self.all_token_idxs = list(self.vocab.values())
        self.all_token_embeddings = self.word_embedder(torch.tensor(self.all_token_idxs).to(self.torch_device)) 
        self.all_token_embeddings_norm = self.all_token_embeddings / self.all_token_embeddings.norm(dim=-1, keepdim=True)
        self.n_tokens = n_tokens
        self.minmize = minimize 
        self.batch_size = batch_size
        assert not minimize 
        self.search_space_dim = 768 
        self.dim = self.n_tokens*self.search_space_dim

    def get_non_related_values(self):
        tmp = [] 
        for word in self.related_vocab:
            tmp.append(word)
            tmp.append(word+'</w>') 
        self.related_vocab = tmp
        non_related_values = [] 
        for key in self.vocab.keys():
            if not key in self.related_vocab:
                non_related_values.append(self.vocab[key])
        return non_related_values

    def proj_word_embedding(self, word_embedding):
        '''
            Given a word embedding, project it to the closest word embedding of actual tokens using cosine similarity
            Iterates through each token dim and projects it to the closest token
            args:
                word_embedding: (batch_size, max_num_tokens, 768) word embedding
            returns:
                proj_tokens: (batch_size, max_num_tokens) projected tokens
        '''
        assert self.dist_metric == "sq_euclidean"
        # Get word embeddings of all possible tokens as torch tensor
        proj_tokens = []
        # Iterate through batch_size
        for i in range(word_embedding.shape[0]):
            # Euclidean Norm
            dists =  torch.norm(self.all_token_embeddings.unsqueeze(1) - word_embedding[i,:,:], dim = 2)
            closest_tokens = torch.argmin(dists, axis = 0)
            closest_tokens = torch.tensor([self.all_token_idxs[token] for token in closest_tokens]).to(self.torch_device)
            closest_vocab = self.tokenizer.decode(closest_tokens)
            if self.prepend_to_text: 
                closest_vocab = closest_vocab + " " + self.prepend_to_text
            # cur_proj_tokens = [closest_vocab]
            proj_tokens.append(closest_vocab)  # cur_proj_tokens) 
        return proj_tokens

    def prompt_to_text(self, prompts):
        gen_texts = self.generator( prompts, max_length=self.max_gen_length, num_return_sequences=self.num_gen_seq, num_beams=self.num_gen_seq)
        gen_texts = [[cur_dict['generated_text'] for cur_dict in cur_gen] for cur_gen in gen_texts]
        return gen_texts
        
    def text_to_loss(self, text): # , loss_type='log_prob_pos') 
        if self.loss_type in ['log_prob_pos', 'log_prob_neg']: 
            num_prompts = len(text) 
            flattened_text = [item for sublist in text for item in sublist]
            inputs = self.distilBert_tokenizer(flattened_text, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = self.distilBert_model(**inputs).logits
            probs = torch.softmax(logits, dim = 1)
            if self.loss_type == 'log_prob_pos':
                loss = torch.log(probs[:,1])
            elif self.loss_type == 'log_prob_neg':
                loss = torch.log(probs[:,0])
            else:
                raise ValueError(f"loss_type must be one of ['log_prob_pos', 'log_prob_neg'] but was {self.loss_type}")
            loss = loss.reshape(num_prompts, -1) 
        elif self.loss_type in ["perc_target", "num_target", "target_occurrences"]: # else: #s if self.loss_type == 'perc_ts':
            n_input = self.n_tokens + self.N_extra_prepend_tokens  
            losses = []
            for outputs in text:
                scores_for_prompt = []
                for output in outputs:
                    words_with_target = 0.0 
                    total_words = 0.0 
                    occurrences = 0.0 
                    for word in output.split()[n_input:]:
                        if self.target_string in word:
                            words_with_target += 1.0 
                            for char in word:
                                if char == self.target_string:
                                    occurrences += 1.0 
                        total_words += 1.0 
                    if self.loss_type == "perc_target":
                        if total_words > 0:
                            score = words_with_target/total_words
                        else:
                            score = 0.0 
                    elif self.loss_type == "num_target": # num words_with_target
                        score = words_with_target
                    elif self.loss_type == "target_occurrences": # total number of chars
                        score = occurrences 
                    scores_for_prompt.append(score) 
                scores_for_prompt = torch.tensor(scores_for_prompt).float() 
                losses.append(scores_for_prompt.unsqueeze(0))
            loss = torch.cat(losses) 
        else:
            assert 0 

        return loss  # torch.Size([2, 5]) = torch.Size([bsz, N_avg_over])
        
    def pipe(self, input_type, input_value, output_types):
        valid_input_types = ['raw_word_embedding' ,'prompt']
        valid_output_types = ['prompt', 'generated_text', 'loss']
        # Check that types are valid 
        if input_type not in valid_input_types:
            raise ValueError(f"input_type must be one of {valid_input_types} but was {input_type}")
        for cur_output_type in output_types:
            if cur_output_type not in valid_output_types:
                raise ValueError(f"output_type must be one of {valid_output_types} but was {cur_output_type}")
        # Check that output is downstream
        pipeline_order = ["raw_word_embedding", "prompt", "generated_text", "loss"]
        pipeline_maps = {"raw_word_embedding": self.proj_word_embedding,
                        "prompt": self.prompt_to_text, # prompt to generated text 
                        "generated_text": self.text_to_loss, # text to generated loss 
                        }
        start_index = pipeline_order.index(input_type)
        max_end_index = start_index
        for cur_output_type in output_types:
            cur_end_index = pipeline_order.index(cur_output_type)
            if start_index >= cur_end_index:
                raise ValueError(f"{cur_output_type} is not downstream of {input_type}.")
            else:
                max_end_index = max(max_end_index,cur_end_index)

        cur_pipe_val = input_value
        output_dict = {}
        for i in range(start_index, max_end_index):
            cur_type = pipeline_order[i]
            mapping = pipeline_maps[cur_type]
            cur_pipe_val = mapping(cur_pipe_val)
            next_type = pipeline_order[i+1]
            if next_type in output_types:
                output_dict[next_type] =  cur_pipe_val
        return output_dict

    def query_oracle(self, x ):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float16)
        x = x.cuda() 
        x = x.reshape(-1, self.n_tokens, self.search_space_dim) 
        out_dict = self.pipe(
            input_type="raw_word_embedding", 
            input_value=x, 
            output_types=['prompt','generated_text','loss'] 
        ) 
        y = out_dict['loss'].mean(-1 ) 
        if self.minmize: 
            y = y*-1 
        return out_dict['prompt'], y, out_dict["generated_text"]

