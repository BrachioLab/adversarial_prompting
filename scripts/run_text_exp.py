import random
import torch
import numpy as np 
import wandb
import argparse
import os
os.environ["WANDB_SILENT"] = "true" 
import sys
sys.path.append("../") 
from utils.text_utils.language_model import HuggingFaceLanguageModel, LlamaHuggingFace, StableHuggingFace, VicunaHuggingFace
from utils.text_utils.text_losses import CountLetterLoss, EmotionLoss, ToxicityLoss, PerplexityLoss, PerplexityWithSeedLoss
from run_optimization import SquareAttackOptim, TurboOptim, RandomSearchOptim

from utils.constants import tuple_type, trimmed_mean
from utils.token_embedder import OPTEmbedding, TinyBERTEmbedding
class RunTextExp():
    def __init__(self, lm_args, optim_args, loss_args):
        self.lm_args = lm_args
        self.optim_args = optim_args
        self.loss_args = loss_args

        # Set Language Model
        self.language_model = self.get_language_model(lm_args["language_model"],
                                    lm_args["max_gen_length"],
                                    lm_args["n_tokens"],
                                    lm_args["seed_text"],
                                    lm_args["num_gen_seq"])
    
        language_model_vocab_tokens = self.language_model.get_vocab_tokens()
        # Set Token Embedder
        self.token_embedder = self.get_token_embedder(lm_args["embedding_model"],language_model_vocab_tokens)
        self.vocab = self.token_embedder.get_vocab()

        # Set Loss Function
        self.loss_fn = self.get_loss_fn(loss_args)

        self.set_seed(optim_args["seed"])
        self.optim_args = optim_args
        #self.lm_args = lm_args
        #self.loss_args = loss_args

    def set_seed(self,seed):
        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = False
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = False
        if seed is not None:
            torch.manual_seed(seed) 
            random.seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            os.environ["PYTHONHASHSEED"] = str(seed)
    
    def get_language_model(self, language_model_name, max_gen_length, n_tokens, seed_text, num_gen_seq):
        seed_text_len = 0
        if seed_text is not None:
            # This is not exact but rough approximation, 
            # TODO: more sophisticated way to get seed text length later
            seed_text_len = len(seed_text.split() ) * 2
            
        max_num_tokens = max_gen_length + n_tokens + seed_text_len
        if language_model_name in ["facebook/opt-125m","facebook/opt-350m","facebook/opt-1.3b","facebook/opt-2.7b", "facebook/opt-6.7b", "facebook/opt-13b", "facebook/opt-30b","gpt2"]:
            lm = HuggingFaceLanguageModel(max_num_tokens, num_gen_seq)
            lm.load_model(language_model_name)
        elif language_model_name in ["llama-7b","llama-13b","llama-30b"]:
            lm = LlamaHuggingFace(max_num_tokens, num_gen_seq)
            lm.load_model(language_model_name)    
        elif language_model_name in ["stablelm"]:
            max_num_tokens = max_gen_length
            lm = StableHuggingFace(max_num_tokens, num_gen_seq)
            lm.load_model(language_model_name)
        elif language_model_name in ["vicuna1.1"]:
            max_num_tokens = max_gen_length
            lm = VicunaHuggingFace(max_num_tokens, num_gen_seq)
            lm.load_model(language_model_name)
        else:
            # Will add future language model functionality
            raise NotImplementedError
        return lm



    def get_loss_fn(self, loss_args):
        loss_type = loss_args["loss_type"]
        if loss_type == "count_letter":
            return CountLetterLoss(letter = loss_args["target_string"])
        elif loss_type == "emotion":
            return EmotionLoss(emotion_class = loss_args["emotion_class"])
        elif loss_type == "toxicity":
            return ToxicityLoss()
        elif loss_type == "perplexity":
            return PerplexityLoss()
        elif loss_type == "perplexity_seed":
            return PerplexityWithSeedLoss(loss_args["seed_text"])
        else:
            raise NotImplementedError

    def prompts_to_texts(self, prompts):
        generated_text = self.language_model.generate_text(prompts, self.lm_args["seed_text"])
        return generated_text

    def get_token_embedder(self, embedding_model, language_model_vocab_tokens):
        if embedding_model == "opt":
            return OPTEmbedding(language_model_vocab_tokens)
        elif embedding_model == "tinybert":
            return TinyBERTEmbedding(language_model_vocab_tokens)
        else:
            print(f"Embedding model was {embedding_model} but should be `opt` or `tinybert`")
            raise NotImplementedError

    def texts_to_losses(self, generated_texts):
        losses = self.loss_fn(generated_texts)
        if self.loss_fn.goal == "minimize":
            losses *= -1.0
        return losses

    def start_wandb(self):
        args_dict = {**self.lm_args, **self.optim_args, **self.loss_args}
        self.tracker = wandb.init(
            entity=args_dict['wandb_entity'], 
            project=args_dict['wandb_project_name'],
            config=args_dict, 
        ) 
        print('running', wandb.run.name) 

    def get_baseline_prompts(self):
        loss_type = self.loss_args["loss_type"]
        if loss_type == "count_letter":
            target_str = self.loss_args["target_string"]
            prompts = [f"generate as many of the letter {target_str} as possible:", f"words with many of the letter {target_str}:"]
        elif loss_type == "emotion":
            prompts = ["anger anger anger anger", "fear fear fear fear", "joy joy joy joy", "disgust disgust disgust disgust",
            "neutral neutral neutral neutral", "sadness sadness sadness sadness", "surprise surprise surprise surprise"]
        elif loss_type == "toxicity":
            prompts = ["very nice","very mean","nice","mean"]
        elif loss_type == "perplexity" or "perplexity_seed":
            prompts = ["Fishtail spoons orbit the parrot's enigma", "Quantum trapezoids forge kaleidoscopic xylophones","Petunia exoplanets converge with multidimensional eggplants"]
        else:
            raise NotImplementedError
        return prompts 

    def log_baseline_prompts(self, baseline_prompts):
        batch_size = self.optim_args["batch_size"]
        while (len(baseline_prompts) % batch_size) != 0:
            baseline_prompts.append(baseline_prompts[0])
        n_batches = int(len(baseline_prompts) / batch_size )
        baseline_scores = []
        baseline_gen_text = [] 
        for i in range(n_batches): 
            prompt_batch = baseline_prompts[i*batch_size:(i+1)*batch_size] 
            generated_text = self.prompts_to_texts(prompt_batch)
            losses = self.texts_to_losses(generated_text)
            mean_loss = trimmed_mean(losses,dim = 1)
            #mean_loss = torch.mean(losses,axis = 1)
            baseline_scores.append(mean_loss) 
            baseline_gen_text = baseline_gen_text + [gen_text for prompt,gen_text in generated_text] 
        baseline_scores = torch.cat(baseline_scores).detach().cpu()
        self.best_baseline_score = baseline_scores.max().item()
        best_score_idx = torch.argmax(baseline_scores).item() 
        self.tracker.log({
            "baseline_scores":baseline_scores.tolist(),
            "baseline_prompts":baseline_prompts,
            "baseline_gen_text":baseline_gen_text,
            "best_baseline_score":self.best_baseline_score,
            "best_baseline_prompt":baseline_prompts[best_score_idx],
            "best_baseline_gen_text":baseline_gen_text[best_score_idx],
        }) 

    def run(self):
        # Start W&B Tracking
        self.start_wandb()
        # Get baseline prompts and log them to W&B
        baseline_prompts = self.get_baseline_prompts() 
        self.log_baseline_prompts(baseline_prompts)


        optim_args = self.optim_args
        optimizer = None

        args_subset = {"n_tokens":optim_args["n_tokens"],
                "max_n_calls":optim_args["max_n_calls"],
                "max_allowed_calls_without_progress":optim_args["max_allowed_calls_without_progress"],
                "acq_func":optim_args["acq_func"],
                "failure_tolerance":optim_args["failure_tolerance"],
                "success_tolerance":optim_args["success_tolerance"],
                "init_n_epochs":optim_args["init_n_epochs"],
                "n_epochs":optim_args["n_epochs"],
                "n_init_per_prompt":optim_args["n_init_per_prompt"],
                "hidden_dims":optim_args["hidden_dims"],
                "lr":optim_args["lr"],
                "batch_size":optim_args["batch_size"],
                "vocab":self.vocab,
                "best_baseline_score":self.best_baseline_score,
                "prompts_to_texts":self.prompts_to_texts,
                "texts_to_losses":self.texts_to_losses,
                "token_embedder":self.token_embedder,
                "tracker":self.tracker
        }

        if optim_args["attack_optimizer"] == "turbo":
           
            
            optimizer = TurboOptim(**args_subset)
            
        elif optim_args["attack_optimizer"] == "square":
            optimizer = SquareAttackOptim(**args_subset)
            
        elif optim_args["attack_optimizer"] == "random":
            optimizer = RandomSearchOptim(**args_subset)
        optimizer.optim()
        optimizer.log_final_values()
        self.tracker.finish() 

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    # Optimization args
    parser.add_argument('--wandb_entity', default="nmaus" ) 
    parser.add_argument('--wandb_project_name', default="prompt-optimization-text")  
    parser.add_argument('--n_init_per_prompt', type=int, default=30 )  
    parser.add_argument('--hidden_dims', type=tuple_type, default="(256,128,64)") 
    parser.add_argument('--lr', type=float, default=0.005 ) 
    parser.add_argument('--n_epochs', type=int, default=25)  
    parser.add_argument('--init_n_epochs', type=int, default=200) 
    parser.add_argument('--acq_func', type=str, default='ts') 
    parser.add_argument('--debug', type=bool, default=False) 
    parser.add_argument('--attack_optimizer', type=str, default='turbo', choices=["turbo","square","random"]) 
    parser.add_argument('--seed', type=int, default=1) 
    parser.add_argument('--success_value', type=int, default=8)  
    parser.add_argument('--break_after_success', type=bool, default=False)
    parser.add_argument('--max_n_calls', type=int, default=5_000) 
    parser.add_argument('--n_tokens', type=int, default=4,help="Number of tokens to optimizer over") 
    parser.add_argument('--batch_size', type=int, default=10)  
    parser.add_argument('--failure_tolerance', type=int, default=32 )  
    parser.add_argument('--success_tolerance', type=int, default=10 )  
    parser.add_argument('--max_allowed_calls_without_progress', type=int, default=2_000 ) 

    # Language Model args
    parser.add_argument('--task', default="textgen") 
    parser.add_argument('--num_gen_seq', type=int, default=10 ) 
    parser.add_argument('--max_gen_length', type=int, default=100 )   
    parser.add_argument('--language_model', default="facebook/opt-125m") 
    parser.add_argument('--embedding_model', default="tinybert", choices=["opt","tinybert"]) 
    parser.add_argument('--seed_text', default=None,help="Add a seed text so that the optimized prompt is prepended to the seed text.")
    parser.add_argument('--seed_text_name', type=str, default="none")

    # Loss function args
    parser.add_argument('--loss_type', default="target_occurrences",choices=["count_letter","emotion","toxicity","perplexity","perplexity_seed"]) 
    parser.add_argument('--target_string', default="t")  
    parser.add_argument('--emotion_class', default="anger",choices=["anger","joy","sadness","fear","surprise","disgust","neutral"])
    parser.add_argument('--minimize', type=bool, default=False)  
    
    

    args = vars(parser.parse_args())
    loss_keys = ["loss_type","target_string","minimize","emotion_class", "seed_text"]
    lm_keys = ["task","language_model","embedding_model", "seed_text","num_gen_seq","max_gen_length","n_tokens", "seed_text_name"]
    optim_keys = ["wandb_entity","wandb_project_name", "n_init_per_prompt","hidden_dims","lr","n_epochs","init_n_epochs",
        "acq_func","debug","attack_optimizer","seed","success_value","break_after_success","max_n_calls","n_tokens","batch_size","failure_tolerance","success_tolerance","max_allowed_calls_without_progress"]
    loss_args = {key: args[key] for key in loss_keys}
    lm_args = {key: args[key] for key in lm_keys}
    optim_args = {key: args[key] for key in optim_keys}

    runner = RunTextExp(lm_args,optim_args,loss_args)
    runner.run()


