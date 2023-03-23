
import torch
import numpy as np 
import argparse 
import wandb 
import math 
import os 
import pandas as pd 
import copy 
import sys 
sys.path.append("../")
from utils.objectives.text_generation_objective import TextGenerationObjective
os.environ["WANDB_SILENT"] = "true" 
from scripts.image_optimization import (
    RunTurbo,
    tuple_type,
)

class OptimizeText(RunTurbo):
    def __init__(self, args):
        self.args = args 


    def get_baseline_prompts(self):
        ## ** , here we maximize this "loss" ! 
        if self.args.loss_type not in ["log_prob_neg", "log_prob_pos"]:
            target_str = self.args.target_string # perc_target, num_target, "target_occurrences"
            prompts = [f"generate lots of {target_str}s", f"words with many {target_str}s"]
        else:
            prompts = [] # 5 example baseline prompts 
            if self.args.loss_type == "log_prob_pos":
                target_str = "happy"
            elif self.args.loss_type == "log_prob_neg":
                target_str = "sad"
            else:
                assert 0 

            # "happy happy happy happy" 
            prompt1 = target_str 
            for i in range(self.args.n_tokens - 1):
                prompt1 +=  f" {target_str}" 
            prompts.append(prompt1) 

            # "very very very happy"
            prompt2 = "very"
            for _ in range(self.args.n_tokens - 2):
                prompt2 += f" very" 
            prompt2 +=  f" {target_str}" 
            prompts.append(prompt2)

        # If prepend task: 
        if self.args.prepend_task:
            temp = []
            for prompt in prompts:
                temp.append(prompt + f" {self.args.prepend_to_text}")
            prompts = temp 
    
        return prompts 

    def log_baseline_prompts(self):
        baseline_prompts = self.get_baseline_prompts() 
        while (len(baseline_prompts) % self.args.bsz) != 0:
            baseline_prompts.append(baseline_prompts[0])
        n_batches = int(len(baseline_prompts) / self.args.bsz )
        baseline_scores = []
        baseline_gen_text = [] 
        for i in range(n_batches): 
            prompt_batch = baseline_prompts[i*self.args.bsz:(i+1)*self.args.bsz] 
            out_dict = self.args.objective.pipe(
                input_type="prompt", 
                input_value=prompt_batch, 
                output_types=['generated_text','loss'] 
            ) 
            ys = out_dict['loss'].mean(-1 ) 
            gen_text = out_dict["generated_text"]
            baseline_scores.append(ys) 
            baseline_gen_text = baseline_gen_text + gen_text  

        baseline_scores = torch.cat(baseline_scores).detach().cpu() # self.best_baseline_score
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
        self.prcnt_latents_correct_class_most_probable = 1.0 # for compatibility with image gen task

    def get_init_data(self,):
        # get scores for baseline_prompts 
        self.log_baseline_prompts() 
        # initialize random starting data 
        YS = [] 
        XS = [] 
        PS = []
        GS = []
        # if do batches of more than 10, get OOM 
        n_batches = math.ceil(self.args.n_init_pts / self.args.bsz) 
        for ix in range(n_batches): 
            X = torch.randn(self.args.bsz, self.args.objective.dim )*0.01
            XS.append(X)   
            prompts, ys, gen_text = self.args.objective(X.to(torch.float16))
            YS.append(ys) 
            PS = PS + prompts
            GS = GS + gen_text 
        Y = torch.cat(YS).detach().cpu() 
        self.args.X = torch.cat(XS).float().detach().cpu() 
        self.args.Y = Y.unsqueeze(-1)  
        self.args.P = PS
        self.args.G = GS 
        self.best_baseline_score = -1 # filler b/c used my image opt 

    def save_stuff(self):
        # X = self.args.X
        Y = self.args.Y
        P = self.args.P 
        G = self.args.G 
        # best_x = X[Y.argmax(), :].squeeze().to(torch.float16)
        # torch.save(best_x, f"../best_xs/{wandb.run.name}-best-x.pt") 
        best_prompt = P[Y.argmax()]
        self.tracker.log({"best_prompt":best_prompt}) 
        best_gen_text = G[Y.argmax()]
        self.tracker.log({"best_gen_text":best_gen_text}) 
        # save_path = f"../best_xs/{wandb.run.name}-all-data.csv"
        # prompts_arr = np.array(P)
        # loss_arr = Y.squeeze().detach().cpu().numpy() 
        # gen_text_arr = np.array(G)  # (10, 5)  = N, n_gen_text 
        # df = pd.DataFrame() 
        # df['prompt'] = prompts_arr
        # df["loss"] = loss_arr 
        # for i in range(gen_text_arr.shape[-1]): 
        #     df[f"gen_text{i+1}"] = gen_text_arr[:,i] 
        # df.to_csv(save_path, index=None)

    def call_oracle_and_update_next(self, x_next):
        p_next, y_next, g_next = self.args.objective(x_next.to(torch.float16))
        self.args.P = self.args.P + p_next # prompts 
        self.args.G = self.args.G + g_next # generated text 
        return y_next

    def init_objective(self,):
        self.args.objective = TextGenerationObjective(
            num_gen_seq=self.args.num_gen_seq,
            max_gen_length=self.args.max_gen_length,
            dist_metric=self.args.dist_metric, # "sq_euclidean",
            n_tokens=self.args.n_tokens,
            minimize=self.args.minimize, 
            batch_size=self.args.bsz,
            prepend_to_text=self.args.prepend_to_text,
            lb = self.args.lb,
            ub = self.args.ub,
            text_gen_model=self.args.text_gen_model,
            loss_type=self.args.loss_type,
            target_string=self.args.target_string,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--n_init_per_prompt', type=int, default=None ) 
    parser.add_argument('--n_init_pts', type=int, default=None) 
    parser.add_argument('--lr', type=float, default=0.01 ) 
    parser.add_argument('--n_epochs', type=int, default=2)  
    parser.add_argument('--init_n_epochs', type=int, default=80) 
    parser.add_argument('--acq_func', type=str, default='ts' ) 
    parser.add_argument('--debug', type=bool, default=False) 
    parser.add_argument('--minimize', type=bool, default=False)  
    parser.add_argument('--task', default="textgen") 
    parser.add_argument('--hidden_dims', type=tuple_type, default="(256,128,64)") 
    parser.add_argument('--more_hdims', type=bool, default=True) # for >8 tokens only 
    parser.add_argument('--seed', type=int, default=1 ) 
    parser.add_argument('--success_value', type=int, default=8)  
    parser.add_argument('--break_after_success', type=bool, default=False)
    parser.add_argument('--max_n_calls', type=int, default=3_000) 
    parser.add_argument('--num_gen_seq', type=int, default=5 ) 
    parser.add_argument('--max_gen_length', type=int, default=20 ) 
    parser.add_argument('--dist_metric', default="sq_euclidean" )  
    parser.add_argument('--n_tokens', type=int, default=4 ) 
    parser.add_argument('--failure_tolerance', type=int, default=32 )  
    parser.add_argument('--success_tolerance', type=int, default=10 )  
    parser.add_argument('--max_allowed_calls_without_progress', type=int, default=1_000 ) # for square baseline! 
    parser.add_argument('--text_gen_model', default="opt" ) 
    parser.add_argument('--square_attack', type=bool, default=False) 
    parser.add_argument('--bsz', type=int, default=10)  
    parser.add_argument('--prepend_task', type=bool, default=False)  
    parser.add_argument('--prepend_to_text', default="I am happy")
    parser.add_argument('--loss_type', default="target_occurrences" ) 
    parser.add_argument('--target_string', default="t" )  
    parser.add_argument('--wandb_entity', default="nmaus" ) 
    parser.add_argument('--wandb_project_name', default="prompt-optimization-text" )  
    args = parser.parse_args() 
    if args.loss_type == "log_prob_neg":
        args.prepend_to_text = "I am happy"
    elif args.loss_type == "log_prob_pos":
        args.prepend_to_text = "I am sad"
    assert args.text_gen_model in ["gpt2", "opt", "opt350", "opt13b", "opt66b"] 

    runner = OptimizeText(args)
    runner.run()

