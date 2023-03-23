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
import argparse 
import wandb 
import math 
import os 
os.environ["WANDB_SILENT"] = "true" 
import random 
from utils.constants import PREPEND_TASK_VERSIONS

class RunTurbo():
    def __init__(self, args):
        self.args = args 

    def initialize_global_surrogate_model(self, init_points, hidden_dims):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda() 
        model = GPModelDKL(
            init_points.cuda(), 
            likelihood=likelihood,
            hidden_dims=hidden_dims,
        ).cuda()
        model = model.eval() 
        model = model.cuda()
        return model  

    def start_wandb(self):
        args_dict = vars(self.args) 
        self.tracker = wandb.init(
            entity=args_dict['wandb_entity'], 
            project=args_dict['wandb_project_name'],
            config=args_dict, 
        ) 
        print('running', wandb.run.name) 

    def set_seed(self):
        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = False
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = False
        seed = self.args.seed  
        if seed is not None:
            torch.manual_seed(seed) 
            random.seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            os.environ["PYTHONHASHSEED"] = str(seed)

    def update_surr_model(
        self,
        model,
        learning_rte,
        train_z,
        train_y,
        n_epochs
    ):
        model = model.train() 
        mll = PredictiveLogLikelihood(model.likelihood, model, num_data=train_z.shape[0] )
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rte} ], lr=learning_rte)
        train_bsz = min(len(train_y),128)
        train_dataset = TensorDataset(train_z.cuda(), train_y.cuda())
        train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
        for _ in range(n_epochs):
            for (inputs, scores) in train_loader:
                optimizer.zero_grad()
                output = model(inputs.cuda())
                loss = -mll(output, scores.cuda()).sum() 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        model = model.eval()
        return model


    def get_init_prompts(self):
        related_vocab = self.args.objective.related_vocab
        all_vocab = list(self.args.objective.vocab.keys()) 
        random.shuffle(all_vocab)
        starter_vocab = all_vocab[0:100]
        tmp = [] 
        for vocab_word in starter_vocab:
            if not vocab_word in related_vocab:
                tmp.append(vocab_word)
        starter_vocab = tmp 
        prompts = [] 
        iters = math.ceil(self.args.n_init_pts/self.args.n_init_per_prompt) 
        for i in range(iters):
            prompt = ""
            for j in range(self.args.n_tokens): # N
                if j > 0: 
                    prompt += " "
                # if i == 0:
                #     prompt += self.args.objective.optimal_class 
                # else:
                prompt += random.choice(starter_vocab)
            prompts.append(prompt)

        return prompts 
    

    def get_baseline_prompts(self):
        prompts = [] # 5 example baseline prompts 
        obj_cls = self.args.objective.optimal_class 
        # "CLS CLS CLS CLS" 
        prompt1 = obj_cls 
        for i in range(self.args.n_tokens - 1):
            prompt1 +=  f" {obj_cls }" 
        prompts.append(prompt1) 

        # "CLS end end end"
        prompt2 = obj_cls 
        for _ in range(self.args.n_tokens - 1):
            prompt2 += " <|endoftext|>"
        prompts.append(prompt2)

        # # "a picture of a CLS" 
        if self.args.n_tokens == 2:
            prompts.append(f"a {obj_cls}")
        elif self.args.n_tokens == 3:
            prompts.append(f"picture of {obj_cls}")
        elif self.args.n_tokens == 4:
            prompts.append(f"picture of a {obj_cls}")
        elif self.args.n_tokens == 5:
            prompts.append(f"a picture of a {obj_cls}")
        elif self.args.n_tokens > 5:
            prompt3 = f"a picture of a {obj_cls}"
            for _ in range(self.args.n_tokens - 5):
                prompt3 += " <|endoftext|>"
            prompts.append(prompt3)
    
        return prompts 

    def get_init_data(self ):
        # get scores for baseline_prompts 
        self.log_baseline_prompts() 
        # then get initialization prompts + scores ... 
        prompts = self.get_init_prompts()
        YS = [] 
        XS = [] 
        PS = []
        most_probable_clss = []
        prcnt_correct_clss = []
        # if do batches of more than 10, get OOM 
        n_batches = math.ceil(self.args.n_init_pts / (self.args.bsz*self.args.n_init_per_prompt)) 
        for i in range(n_batches): 
            prompt_batch = prompts[i*self.args.bsz:(i+1)*self.args.bsz] 
            X = self.args.objective.get_init_word_embeddings(prompt_batch) 
            X = X.detach().cpu() 
            X = X.reshape(self.args.bsz, self.args.objective.dim ) 
            for j in range(self.args.n_init_per_prompt): # 10 randoms near each prompt ! 
                if j > 0:
                    X = X + torch.randn(self.args.bsz, self.args.objective.dim)*0.01
                XS.append(X)   
                xs, ys = self.args.objective(X.to(torch.float16))
                YS.append(ys) 
                PS = PS + xs 
                most_probable_clss = most_probable_clss + self.args.objective.most_probable_classes
                prcnt_correct_clss = prcnt_correct_clss + self.args.objective.prcnts_correct_class
        Y = torch.cat(YS).detach().cpu() 
        Y = Y.unsqueeze(-1)  
        XS = torch.cat(XS).float().detach().cpu() 
        self.args.X = XS
        self.args.Y = Y 
        self.args.P = PS 
        self.args.most_probable_clss = most_probable_clss 
        self.args.prcnt_correct_clss = prcnt_correct_clss 

    def log_baseline_prompts(self):
        baseline_prompts = self.get_baseline_prompts() 
        while (len(baseline_prompts) % self.args.bsz) != 0:
            baseline_prompts.append(baseline_prompts[0])
        n_batches = int(len(baseline_prompts) / self.args.bsz )
        baseline_scores = []
        out_baseline_prompts = [] 
        baseline_most_probable_clss = []
        baseline_prcnt_correct_clss = []
        for i in range(n_batches): 
            prompt_batch = baseline_prompts[i*self.args.bsz:(i+1)*self.args.bsz] 
            X = self.args.objective.get_init_word_embeddings(prompt_batch) 
            X = X.detach().cpu() 
            X = X.reshape(self.args.bsz, self.args.objective.dim )
            ## XXX  
            xs, ys = self.args.objective(X.to(torch.float16))
            baseline_scores.append(ys) 
            out_baseline_prompts = out_baseline_prompts + xs
            baseline_most_probable_clss = baseline_most_probable_clss + self.args.objective.most_probable_classes
            baseline_prcnt_correct_clss = baseline_prcnt_correct_clss + self.args.objective.prcnts_correct_class 
        baseline_scores = torch.cat(baseline_scores).detach().cpu() # self.best_baseline_score
        self.best_baseline_score = baseline_scores.max().item()
        best_score_idx = torch.argmax(baseline_scores).item() 
        self.tracker.log({
            "baseline_scores":baseline_scores.tolist(),
            "baseline_prompts":out_baseline_prompts,
            "baseline_most_probable_classes":baseline_most_probable_clss,
            "baseline_prcnt_latents_correct_class_most_probables":baseline_prcnt_correct_clss,
            "best_baseline_score":self.best_baseline_score,
            "best_baseline_prompt":out_baseline_prompts[best_score_idx],
            "best_baseline_most_probable_class":baseline_most_probable_clss[best_score_idx],
            "best_baseline_prcnt_latents_correct_class_most_probable":baseline_prcnt_correct_clss[best_score_idx],
        }) 

    def save_stuff(self ):
        # X = self.args.X
        Y = self.args.Y
        P = self.args.P 
        C = self.args.most_probable_clss
        PRC = self.args.prcnt_correct_clss 
        best_prompt = P[Y.argmax()] 
        self.tracker.log({"best_prompt":best_prompt}) 
        print(f"Best score found: {self.args.Y.max().item()}")
        print(f"Best prompt found: {best_prompt} \n")
        print("")
        # most probable class (mode over latents)
        most_probable_class = C[Y.argmax()] 
        self.tracker.log({"most_probable_class":most_probable_class}) 
        # prcnt of latents where most probable class is correct (ie 3/5)
        self.prcnt_latents_correct_class_most_probable = PRC[Y.argmax()] 
        self.tracker.log({"prcnt_latents_correct_class_most_probable":self.prcnt_latents_correct_class_most_probable}) 
        # save_path = f"../best_xs/{wandb.run.name}-all-data.csv"
        # df = pd.DataFrame() 
        # df['prompt'] = np.array(P)
        # df['most_probable_class'] = np.array(C)
        # df['prcnt_latents_correct_class_most_probable'] = np.array(PRC) 
        # df["loss"] = Y.squeeze().detach().cpu().numpy() 
        # df.to_csv(save_path, index=None)

    def init_args(self):
        self.args.n_init_per_prompt = 10 
        if not self.args.prepend_task: 
            self.args.prepend_to_text = ""
        self.args.lb = None
        self.args.ub = None
        # flags for wandb recording 
        self.args.update_state_fix = True 
        self.args.update_state_fix2 = True 
        self.args.update_state_fix3 = True 
        self.args.record_most_probable_fix2 = True 
        self.args.flag_set_seed = True
        self.args.flag_fix_max_n_tokens = True
        self.args.flag_fix_args_reset = True  
        self.args.flag_reset_gp_new_data = True # reset gp every 10 iters up to 1024 data points 
        self.args.n_init_pts = self.args.bsz * self.args.n_init_per_prompt
        assert self.args.n_init_pts % self.args.bsz == 0

    def call_oracle_and_update_next(self, x_next):
        prompts_next, y_next = self.args.objective(x_next.to(torch.float16))
        self.args.P = self.args.P + prompts_next
        self.args.most_probable_clss = self.args.most_probable_clss + self.args.objective.most_probable_classes
        self.args.prcnt_correct_clss = self.args.prcnt_correct_clss + self.args.objective.prcnts_correct_class
        return y_next

    def init_objective(self):
        self.args.objective = ImageGenerationObjective(
            n_tokens=self.args.n_tokens,
            minimize=self.args.minimize, 
            batch_size=self.args.bsz,
            use_fixed_latents=False,
            project_back=True,
            avg_over_N_latents=self.args.avg_over_N_latents,
            exclude_high_similarity_tokens=self.args.exclude_high_similarity_tokens,
            seed=self.args.seed,
            prepend_to_text=self.args.prepend_to_text,
            optimal_class=self.args.optimal_class,
            optimal_class_level=self.args.optimal_class_level,# 1,
            optimal_sub_classes=self.args.optimal_sub_classes, # [],
            lb = self.args.lb,
            ub = self.args.ub,
            similar_token_threshold=self.args.similar_token_threshold, 
        )

    def get_avg_dist_between_vectors(self):
        # Only needed to compute once. 
        embeddings = self.args.objective.all_token_embeddings.cpu().to(torch.float32)
        dists = torch.cdist(embeddings.cuda(), embeddings.cuda(), p=2.0)
        dists2 = dists.flatten()
        dists2 = dists2.detach().cpu()
        dists3 = dists2[dists2 > 0.0]
        avg_dist = dists3.mean() 
        print(avg_dist) # tensor(0.5440)
        # saved in AVG_DIST_BETWEEN_VECTORS
        return avg_dist 

    def square_attack(self):
        print("setting seed")
        self.set_seed()
        self.init_args()  
        self.start_wandb() # initialized self.tracker 
        print("initializing objective")
        self.init_objective() 
        print("computing scores for initialization data")
        self.get_init_data() 
        AVG_DIST_BETWEEN_VECTORS = 0.5440
        prev_best = -torch.inf 
        n_iters = 0
        self.beat_best_baseline = False 
        self.prcnt_latents_correct_class_most_probable = 0.0
        n_calls_without_progress = 0
        prev_loss_batch_std = self.args.Y.std().item() 
        print("Starting main optimizatiton loop")
        while self.args.objective.num_calls < self.args.max_n_calls:
            self.tracker.log({
                'num_calls':self.args.objective.num_calls,
                'best_y':self.args.Y.max(),
                'best_x':self.args.X[self.args.Y.argmax(), :].squeeze().tolist(), 
                "beat_best_baseline":self.beat_best_baseline,
            } ) 
            if self.args.Y.max().item() > prev_best: 
                n_calls_without_progress = 0
                prev_best = self.args.Y.max().item() 
                self.save_stuff()
                if prev_best > self.best_baseline_score: 
                    self.beat_best_baseline = True 
            else:
                n_calls_without_progress += self.args.bsz 
            
            if n_calls_without_progress > self.args.max_allowed_calls_without_progress:
                break
            if prev_loss_batch_std == 0: # catch 0 std case! 
                prev_loss_batch_std = 1e-4
            noise_level = AVG_DIST_BETWEEN_VECTORS / (10*prev_loss_batch_std) # One 10th of avg dist between vectors
            x_center = self.args.X[self.args.Y.argmax(), :].squeeze() 
            x_next = [] 
            for _ in range(self.args.bsz):
                x_n = copy.deepcopy(x_center)
                # select random 10% of dims to modify  
                dims_to_modify = random.sample(range(self.args.X.shape[-1]), int(self.args.X.shape[-1]*0.1))
                rand_noise =  torch.normal(mean=torch.zeros(len(dims_to_modify),), std=torch.ones(len(dims_to_modify),)*noise_level) 
                x_n[dims_to_modify] = x_n[dims_to_modify] + rand_noise 
                x_next.append(x_n.unsqueeze(0)) 
            x_next = torch.cat(x_next) 
            self.args.X = torch.cat((self.args.X, x_next.detach().cpu()), dim=-2) 
            y_next = self.call_oracle_and_update_next(x_next)
            prev_loss_batch_std = y_next.std().item() 
            y_next = y_next.unsqueeze(-1)
            self.args.Y = torch.cat((self.args.Y, y_next.detach().cpu()), dim=-2) 
            n_iters += 1 
        self.tracker.finish() 
        return self 

    def run(self):
        if self.args.square_attack:
            self.square_attack() 
        else:
            self.optimize()

    def optimize(self):
        print("setting seed")
        self.set_seed()
        self.init_args()  
        print("starting wandb tracker")
        self.start_wandb() # initialized self.tracker
        print("initializing objective")
        self.init_objective() 
        print("computing scores for initialization data")
        self.get_init_data() 
        print("initializing surrogate model")
        model = self.initialize_global_surrogate_model(
            self.args.X, 
            hidden_dims=self.args.hidden_dims
        ) 
        print("pretraining surrogate model on initial data")
        model = self.update_surr_model(
            model=model,
            learning_rte=self.args.lr,
            train_z=self.args.X,
            train_y=self.args.Y,
            n_epochs=self.args.init_n_epochs
        )
        prev_best = -torch.inf 
        num_tr_restarts = 0  
        tr = TrustRegionState(
            dim=self.args.objective.dim,
            failure_tolerance=self.args.failure_tolerance,
            success_tolerance=self.args.success_tolerance,
        )
        n_iters = 0
        self.beat_best_baseline = False 
        self.prcnt_latents_correct_class_most_probable = 0.0
        n_calls_without_progress = 0
        print("Starting main optimization loop")
        while self.args.objective.num_calls < self.args.max_n_calls:
            self.tracker.log({
                'num_calls':self.args.objective.num_calls,
                'best_y':self.args.Y.max(),
                'best_x':self.args.X[self.args.Y.argmax(), :].squeeze().tolist(), 
                'tr_length':tr.length,
                'tr_success_counter':tr.success_counter,
                'tr_failure_counter':tr.failure_counter,
                'num_tr_restarts':num_tr_restarts,
                "beat_best_baseline":self.beat_best_baseline,
            } ) 
            if self.args.Y.max().item() > prev_best:
                n_calls_without_progress = 0
                prev_best = self.args.Y.max().item() 
                self.save_stuff()
                if prev_best > self.best_baseline_score: 
                    self.beat_best_baseline = True
            else:
                n_calls_without_progress += self.args.bsz 
            
            if n_calls_without_progress > self.args.max_allowed_calls_without_progress:
                break
            x_next = generate_batch( 
                state=tr,
                model=model,
                X=self.args.X,
                Y=self.args.Y,
                batch_size=self.args.bsz, 
                acqf=self.args.acq_func,
                absolute_bounds=(self.args.objective.lb, self.args.objective.ub)
            ) 
            self.args.X = torch.cat((self.args.X, x_next.detach().cpu()), dim=-2) 
            y_next = self.call_oracle_and_update_next(x_next)
            y_next = y_next.unsqueeze(-1)
            self.args.Y = torch.cat((self.args.Y, y_next.detach().cpu()), dim=-2) 
            tr = update_state(tr, y_next) 
            if tr.restart_triggered:
                num_tr_restarts += 1
                tr = TrustRegionState(
                    dim=self.args.objective.dim,
                    failure_tolerance=self.args.failure_tolerance,
                    success_tolerance=self.args.success_tolerance,
                )
                model = self.initialize_global_surrogate_model(self.args.X, hidden_dims=self.args.hidden_dims) 
                model = self.update_surr_model(
                    model=model,
                    learning_rte=self.args.lr,
                    train_z=self.args.X,
                    train_y=self.args.Y,
                    n_epochs=self.args.init_n_epochs
                )
            # flag_reset_gp_new_data 
            elif (self.args.X.shape[0] < 1024) and (n_iters % 10 == 0): # restart gp and update on all data 
                model = self.initialize_global_surrogate_model(self.args.X, hidden_dims=self.args.hidden_dims) 
                model = self.update_surr_model(
                    model=model,
                    learning_rte=self.args.lr,
                    train_z=self.args.X,
                    train_y=self.args.Y,
                    n_epochs=self.args.init_n_epochs
                )
            else:
                model = self.update_surr_model(
                    model=model,
                    learning_rte=self.args.lr,
                    train_z=x_next,
                    train_y=y_next, 
                    n_epochs=self.args.n_epochs
                )
            n_iters += 1
        self.tracker.finish() 

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--wandb_entity', default="nmaus" ) 
    parser.add_argument('--wandb_project_name', default="prompt-optimization-images" )  
    parser.add_argument('--lr', type=float, default=0.01 ) 
    parser.add_argument('--n_epochs', type=int, default=2)  
    parser.add_argument('--init_n_epochs', type=int, default=80) 
    parser.add_argument('--acq_func', type=str, default='ts' ) 
    parser.add_argument('--minimize', type=bool, default=True)    
    parser.add_argument('--hidden_dims', type=tuple_type, default="(256,128,64)") 
    parser.add_argument('--avg_over_N_latents', type=int, default=5) 
    parser.add_argument('--seed', type=int, default=1 ) 
    parser.add_argument('--max_n_calls', type=int, default=5_000) 
    parser.add_argument('--bsz', type=int, default=10)  
    parser.add_argument('--exclude_high_similarity_tokens', type=bool, default=False)  
    parser.add_argument('--similar_token_threshold', type=float, default=-3.0 ) 
    parser.add_argument('--n_tokens', type=int, default=4 )  
    parser.add_argument('--optimal_class', default="dog" )  
    parser.add_argument('--prepend_task', type=bool, default=False)
    parser.add_argument('--prepend_task_version', type=int, default=1) # 1, 2, 3 (dog, mountain, ocean)
    parser.add_argument('--failure_tolerance', type=int, default=32 ) # for TuRBO 
    parser.add_argument('--success_tolerance', type=int, default=10 ) # for TuRBO 
    parser.add_argument('--optimal_sub_classes', type=list, default=[])  
    parser.add_argument('--square_attack', type=bool, default=False) 
    parser.add_argument('--max_allowed_calls_without_progress', type=int, default=1_000 )
    args = parser.parse_args() 
    args.prepend_to_text = PREPEND_TASK_VERSIONS[args.prepend_task_version]
    args.optimal_class_level, args.optimal_sub_classes = get_imagenet_sub_classes(args.optimal_class)

    runner = RunTurbo(args) 
    runner.run()
 
