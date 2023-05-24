import torch
import gpytorch
import numpy as np 
import sys 
import copy 
sys.path.append("../") 
from gpytorch.mlls import PredictiveLogLikelihood 
from utils.bo_utils.trust_region import (
    TrustRegionState, 
    generate_batch, 
    update_state
)
from utils.bo_utils.ppgpr import (
    GPModelDKL,
)
from torch.utils.data import (
    TensorDataset, 
    DataLoader
) 

from utils.imagenet_classes import get_imagenet_sub_classes
from utils.objectives.image_generation_objective import ImageGenerationObjective  
from utils.constants import trimmed_mean
 
import math 
import os 

import random 
from utils.constants import PREPEND_TASK_VERSIONS
from tqdm import tqdm
import abc
from utils.objectives.text_generation_objective import TextGenerationObjective

class RunOptim():
    def __init__(self, 
        n_tokens,
        max_n_calls,
        max_allowed_calls_without_progress,
        acq_func,
        failure_tolerance,
        success_tolerance,
        init_n_epochs,
        n_epochs,
        n_init_per_prompt,
        hidden_dims,
        lr,
        batch_size,
        vocab,
        best_baseline_score,
        prompts_to_texts,
        texts_to_losses,
        token_embedder,  
        tracker      
        ):

        self.n_tokens = n_tokens
        self.max_n_calls = max_n_calls
        self.max_allowed_calls_without_progress = max_allowed_calls_without_progress
        self.acq_func = acq_func

        self.failure_tolerance = failure_tolerance
        self.success_tolerance = success_tolerance
        self.init_n_epochs = init_n_epochs 
        self.n_epochs = n_epochs 
        self.n_init_per_prompt = n_init_per_prompt
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.vocab = vocab
        self.best_baseline_score = best_baseline_score
        self.beat_best_baseline = False # Has not beat baseline yet
        self.prompts_to_texts = prompts_to_texts
        self.texts_to_losses = texts_to_losses
        self.token_embedder = token_embedder
        
        self.tracker = tracker
        
        self.num_optim_updates = 0

        # Optimization has losses, prompts, and generated text
        self.losses = None
        self.prompts = None 
        self.generated_text = None
        self.embeddings = None 

        self.num_calls = 0

        
        self.seed_text = None
        
        # flags for wandb recording 
        self.update_state_fix = True 
        self.update_state_fix2 = True 
        self.update_state_fix3 = True 
        self.record_most_probable_fix2 = True 
        self.flag_set_seed = True
        self.flag_fix_max_n_tokens = True
        self.flag_fix_args_reset = True  
        self.flag_reset_gp_new_data = True # reset gp every 10 iters up to 1024 data points 
        self.n_init_pts = self.batch_size * self.n_init_per_prompt
        assert self.n_init_pts % self.batch_size == 0

        self.objective = None
    def log_values(self):
        losses = self.losses
        prompts = self.prompts
        generated_text = self.generated_text
       
        # Log best values
        best_index = losses.argmax()
        best_prompt = prompts[best_index]
        self.tracker.log({"best_prompt":best_prompt}) 
        best_gen_text = generated_text[best_index]
        self.tracker.log({"best_gen_text":best_gen_text}) 
        print(f"BEST PROMPT: {best_prompt}")
        print(f"BEST LOSS: {round(losses[best_index].item(),3)}")
        print(f"BEST GEN TEXT:")
        for i,text in enumerate(best_gen_text):
            print(f"{i+1}/{len(best_gen_text)}: {text}")
        
        print("\n\n")


    def get_init_prompts(self):
        # TODO: Fix this to be just concatenating tokens
        starter_vocab = self.vocab
        prompts = [] 
        iters = math.ceil(self.n_init_pts / self.n_init_per_prompt) 
        for i in range(iters):
            prompt = ""
            for j in range(self.n_tokens): # N
                prompt += random.choice(starter_vocab)
            prompts.append(prompt)

        return prompts
    
    def get_objective(self):
        return TextGenerationObjective(
            n_tokens=self.n_tokens,
            lb = None,
            ub = None,
            prompts_to_texts = self.prompts_to_texts,
            texts_to_losses = self.texts_to_losses,
            token_embedder = self.token_embedder,
        )

    def get_init_data(self,):
        assert self.objective is not None
        print("Computing Scores for Initialization Data")
        # initialize random starting data 
        self.losses, self.embeddings, self.prompts, self.generated_text = [], [], [], []

        # if do batches of more than 10, get OOM 
        n_batches = math.ceil(self.n_init_pts / self.batch_size) 

        for ix in range(n_batches): 
            cur_embedding = torch.normal(mean = self.weights_mean, std = self.weights_std, size=(self.batch_size, self.objective.dim))
            
            self.embeddings.append(cur_embedding)   
            prompts, losses, generated_text = self.objective(cur_embedding.to(torch.float16))
            self.losses.append(losses) 
            self.prompts = self.prompts + prompts
            self.generated_text = self.generated_text + [gen_text for prompt, gen_text in generated_text]
        self.losses = torch.cat(self.losses).detach().cpu().unsqueeze(-1)
        self.embeddings = torch.cat(self.embeddings).float().detach().cpu()

    def log_final_values(self):
        pass
        #self.tracker.log({"all_losses":self.losses,
        #                  "all_prompts":self.prompts,
                          #"all_generated_text":self.generated_text, 
        #                  })

    @abc.abstractmethod
    def optim(self):
        pass

    def call_oracle_and_update_next(self, embeddings_next):
        prompts, mean_loss, generated_text = self.objective(embeddings_next.to(torch.float16))
        self.prompts = self.prompts + prompts # prompts 
        self.generated_text = self.generated_text + [gen_text for prompt,gen_text in generated_text] # generated text 
        return mean_loss
        
class SquareAttackOptim(RunOptim):
    
    def optim(self):
        self.objective = self.get_objective()
        self.weights_mean, self.weights_std = self.token_embedder.get_embeddings_mean_and_std()
        self.get_init_data()
        
        print("Starting Square Attack")
        AVG_DIST_BETWEEN_VECTORS = self.token_embedder.get_embeddings_avg_dist()
        prev_best = -torch.inf 
        n_iters = 0
        n_calls_without_progress = 0
        prev_loss_batch_std = self.losses.std().item() 
        print("Starting Main Optimization Loop")
        pbar = tqdm(total = self.max_n_calls)
        while self.objective.num_calls < self.max_n_calls:
            self.tracker.log({
                'num_calls':self.objective.num_calls,
                'best_loss':self.losses.max(),
                "beat_best_baseline":self.beat_best_baseline,
            } ) 
            if self.losses.max().item() > prev_best: 
                n_calls_without_progress = 0
                prev_best = self.losses.max().item() 
                self.log_values() # Log values due to improvement
                if prev_best > self.best_baseline_score: 
                    self.beat_best_baseline = True 
            else:
                n_calls_without_progress += self.batch_size 
            
            if n_calls_without_progress > self.max_allowed_calls_without_progress:
                break
            prev_loss_batch_std = max(prev_loss_batch_std, 1e-4)
            noise_level = AVG_DIST_BETWEEN_VECTORS / (10*prev_loss_batch_std) # One tenth of avg dist between vectors
            embedding_center = self.embeddings[self.losses.argmax(), :].squeeze() 
            embedding_next = [] 
            for _ in range(self.batch_size):
                embedding_n = copy.deepcopy(embedding_center)
                # select random 10% of dims to modify  
                dims_to_modify = random.sample(range(self.embeddings.shape[-1]), int(self.embeddings.shape[-1]*0.1))
                rand_noise =  torch.normal(mean=torch.zeros(len(dims_to_modify),), std=torch.ones(len(dims_to_modify),)*noise_level) 
                embedding_n[dims_to_modify] = embedding_n[dims_to_modify] + rand_noise 
                embedding_next.append(embedding_n.unsqueeze(0)) 
            embedding_next = torch.cat(embedding_next) 
            self.embeddings = torch.cat((self.embeddings, embedding_next.detach().cpu()), dim=-2) 
            losses_next = self.call_oracle_and_update_next(embedding_next)
            prev_loss_batch_std = losses_next.std().item() 
            losses_next = losses_next.unsqueeze(-1)
            self.losses = torch.cat((self.losses, losses_next.detach().cpu()), dim=-2) 
            n_iters += 1 
            pbar.update(self.objective.num_calls - pbar.n)
        pbar.close()

    

class TurboOptim(RunOptim):
    

    def get_objective(self):
        return TextGenerationObjective(
            n_tokens=self.n_tokens,
            lb = None, # TODO: Do we need this?
            ub = None,
            prompts_to_texts = self.prompts_to_texts,
            texts_to_losses = self.texts_to_losses,
            token_embedder = self.token_embedder,
        )
    
    def init_global_surrogate_model(self, init_points, hidden_dims):
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda() 
        model = GPModelDKL(
            init_points.cuda(), 
            likelihood=likelihood,
            hidden_dims=hidden_dims,
        ).cuda()
        model = model.eval() 
        model = model.cuda()
        self.optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': self.lr} ], lr=self.lr)
        return model  

    def update_surr_model(
        self,
        model,
        train_z,
        train_y,
        n_epochs,
    ):
        model = model.train() 
        mll = PredictiveLogLikelihood(model.likelihood, model, num_data=train_z.shape[0] )
        #optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': self.lr} ], lr=self.lr)
        train_batch_size = min(len(train_y),128)
        train_dataset = TensorDataset(train_z.cuda(), train_y.cuda())
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        for epoch in range(n_epochs):
            for (inputs, scores) in train_loader:
                #optimizer.zero_grad()
                self.optimizer.zero_grad()
                output = model(inputs.cuda())
                loss = -mll(output, scores.cuda()).mean() 
                self.tracker.log({"num_optim_updates": self.num_optim_updates, "GP Loss":loss.item()})
                loss.backward()
                self.num_optim_updates += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                #optimizer.step()
                self.optimizer.step()
        model = model.eval()
        
        return model

    def optim(self):
        self.objective = self.get_objective()
        self.weights_mean, self.weights_std = self.token_embedder.get_embeddings_mean_and_std()
        self.get_init_data()
        
        print("Initializing Surrogate Model")
        model = self.init_global_surrogate_model(
            self.embeddings, 
            hidden_dims=self.hidden_dims
        ) 
        print("Pretraining Surrogate Model on Initial Data")
        model = self.update_surr_model(
            model=model,
            train_z=self.embeddings,
            train_y=self.losses,
            n_epochs=self.n_epochs,
        )

        tr = TrustRegionState(
            dim=self.objective.dim,
            failure_tolerance=self.failure_tolerance,
            success_tolerance=self.success_tolerance,
        )

        prev_best = -torch.inf 
        num_tr_restarts = 0  
        n_iters = 0
        n_calls_without_progress = 0
        
        print("Starting Main Optimization Loop")
        pbar = tqdm(total = self.max_n_calls)
        while self.objective.num_calls < self.max_n_calls:
            self.tracker.log({
                'num_calls':self.objective.num_calls,
                'best_loss':self.losses.max(),
                #'best_x':self.embeddings[self.losses.argmax(), :].squeeze().tolist(), 
                'tr_length':tr.length,
                'tr_success_counter':tr.success_counter,
                'tr_failure_counter':tr.failure_counter,
                'num_tr_restarts':num_tr_restarts,
                "beat_best_baseline":self.beat_best_baseline,
            } ) 

            if self.losses.max().item() > prev_best:
                n_calls_without_progress = 0
                prev_best = self.losses.max().item() 
                self.log_values()
                self.tracker.log({"best_n_calls":self.objective.num_calls})
                if prev_best > self.best_baseline_score: 
                    self.beat_best_baseline = True
                
            else:
                n_calls_without_progress += self.batch_size 
            
            if n_calls_without_progress > self.max_allowed_calls_without_progress:
                break

            embeddings_next = generate_batch( 
                state = tr,
                model = model,
                X = self.embeddings,
                Y = self.losses,
                batch_size = self.batch_size, 
                acqf = self.acq_func,
                absolute_bounds = (self.objective.lb, self.objective.ub)
            ) 
            self.embeddings = torch.cat((self.embeddings, embeddings_next.detach().cpu()), dim=-2) 
            losses_next = self.call_oracle_and_update_next(embeddings_next).unsqueeze(-1)
            self.losses = torch.cat((self.losses, losses_next.detach().cpu()), dim=-2) 
            tr = update_state(tr, losses_next) 
            if tr.restart_triggered:
                num_tr_restarts += 1
                tr = TrustRegionState(
                    dim=self.objective.dim,
                    failure_tolerance=self.failure_tolerance,
                    success_tolerance=self.success_tolerance,
                )
                model = self.init_global_surrogate_model(self.embeddings, hidden_dims=self.hidden_dims) 
                model = self.update_surr_model(
                    model = model,
                    train_z = self.embeddings,
                    train_y = self.losses,
                    n_epochs = self.init_n_epochs
                )
            # flag_reset_gp_new_data 
            elif (self.embeddings.shape[0] < 2048) and (n_iters % 10 == 0): # restart gp and update on all data 
                model = self.init_global_surrogate_model(self.embeddings, hidden_dims=self.hidden_dims) 
                model = self.update_surr_model(
                    model = model,
                    train_z = self.embeddings,
                    train_y = self.losses,
                    n_epochs = self.init_n_epochs
                )
            else:
                model = self.update_surr_model(
                    model = model,
                    train_z = embeddings_next,
                    train_y = losses_next, 
                    n_epochs = self.n_epochs
                )
            n_iters += 1
            pbar.update(self.objective.num_calls - pbar.n)
        pbar.close()

    

   

    
class RandomSearchOptim(RunOptim):
    
    def optim(self):
        
        self.losses, self.prompts, self.generated_text =torch.empty(0,1), [], []
        print("Starting Random Search Attack")
        
        
        prev_best = -torch.inf 
        n_iters = 0
        n_calls_without_progress = 0
        
        print("Starting Main Optimization Loop")
        pbar = tqdm(total = self.max_n_calls)
        self.num_calls = 0
        while self.num_calls < self.max_n_calls:
            
            self.update_optim()
            self.tracker.log({
                'num_calls':self.num_calls,
                'best_loss':self.losses.max(),
                "beat_best_baseline":self.beat_best_baseline,
            } ) 
            if self.losses.max().item() > prev_best: 
                n_calls_without_progress = 0
                prev_best = self.losses.max().item() 
                self.log_values() # Log values due to improvement
                if prev_best > self.best_baseline_score: 
                    self.beat_best_baseline = True 
            else:
                n_calls_without_progress += self.batch_size 
            
            if n_calls_without_progress > self.max_allowed_calls_without_progress:
                break
            
            n_iters += 1 
            pbar.update(self.num_calls - pbar.n)
        pbar.close()

    def get_random_prompts(self,batch_size):
        prompts = []
        for i in range(batch_size):
            
            prompt = " ".join(random.sample(self.token_embedder.sorted_vocab_keys,self.n_tokens))
            
            prompts.append(prompt)
        return prompts
            
                
    def update_optim(self):
        prompts = self.get_random_prompts(self.batch_size)
        generated_text = self.prompts_to_texts(prompts)
        losses = self.texts_to_losses(generated_text)
        mean_loss = trimmed_mean(losses, dim = 1).unsqueeze(-1)
        self.losses = torch.cat((self.losses, mean_loss.detach().cpu()), dim=-2) 
        self.prompts = self.prompts + prompts # prompts 
        self.generated_text = self.generated_text + [gen_text for prompt,gen_text in generated_text] # generated text 
        self.num_calls += len(generated_text)