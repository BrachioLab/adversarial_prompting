# Adversarial Prompting for Black Box Foundation Models
Official implementation for Adversarial Prompting for Black Box Foundation Models
https://arxiv.org/abs/2302.04237

See directions below to run the code and reproduce all results provided in the paper. 

## Getting Started

### Cloning the Repo (Git Lfs)
This repository uses git lfs to store larger data files and model checkpoints. Git lfs must be installed before cloning the repository. 

```Bash
conda install -c conda-forge git-lfs
```

### Huggingface Token
To use the Huggingface models, update `HUGGING_FACE_TOKEN` in `utils/constants.py`.

### Docker
We provide a Dockerfile in `docker/Dockerfile` that can be used to easily set up the environment needed to run all code in this repository.

### Weights and Biases (wandb) Tracking
This repo is set up to automatically track optimization progress using the Weights and Biases (wandb) API. Wandb stores and updates data during optimization and automatically generates live plots of progress. If you are unfamiliar with wandb, we recommend creating a free account here:
https://wandb.ai/site
Once you have created your account, be sure to pass in your unique wandb identifier/entity when running the code using the following command:
```
--wandb_entity $YOUR_WANDB_ENTITY 
```
Additionally, when you create an account you will receive a unique wandb API key. We recommend adding this key to the provided docker file (see comments in `docker/Dockerfile`), this will allow you to automatically log in to your wandb account when you start a new run. 


## Running the code 

In the paper, we provide prompt optimization results for both image and text generation tasks. The commands below can be used to reproduce all results in the paper. 
By default all optimization tasks will be run using TuRBO since we found this to be the more successful optimization method. To instead run any task with the Square Attack optimization method, pass in the argument `--square_attack True`. 

### Image generation 
Here we provide commands to replicate the image generation results for each of the three threat models discussed in Section 3.2. 

#### Threat Model 1: Unrestricted Prompts
For this task, run the following command once with each optimal/target imagenet class you'd like to use. Change the target class for each run using the `--optimal_class` argument. In the example below we use the target class "bus". 

```Bash
python3 image_optimization.py --optimal_class bus --max_allowed_calls_without_progress 1000 --max_n_calls 5000 --seed 0
```

#### Threat Model 2: Restricted Prompts
For this task, run the following command once with each desired optimal/target imagenet class. To exactly reproduce the paper results, use the target classes listed in Tables 2 and 3. Change the target class for each run using the `--optimal_class argument`. In the example below we use the target class bus. 

```Bash
python3 image_optimization.py --optimal_class bus --max_allowed_calls_without_progress 1000 --max_n_calls 5000 --seed 0 --exclude_high_similarity_tokens True
```

#### Threat Model 3: Restricted Prepending Prompts
For this task, run the following command once with each desired optimal/target Imagenet class. To exactly reproduce the paper results, use the target classes listed in Tables 2 and 3. Change the target class for each run using the `--optimal_class argument`. In the example below we use the target class "bus". 
Additionally, for each target class, run once with each of the three versions so of the prepending task. With `--prepend_task_version 1` we optimize prompts prepended to "a picture of a dog", with `--prepend_task_version 2` we optimize prompts prepended to "a picture of a mountain", and with `--prepend_task_version 3` we optimize prompts prepended to "a picture of the ocean". You can modify the dictionary in `utils/constants.py` to add additional prepending task options if desired. 

```Bash
python3 image_optimization.py --optimal_class bus --max_allowed_calls_without_progress 3000 --max_n_calls 10000 --seed 0 --exclude_high_similarity_tokens True --prepend_task True --prepend_task_version 1
```

### Text generation 
Here we provide commands to replicate the text generation results for each of the two threat models discussed in Section 4. 

#### Threat Model 1: Sentiment Prepending (4.2.1)
Run the following to optimize prompts that, when prepended to "I am sad", still cause the OPT model to generate text with positive sentiment according to the sentiment classifier. 
```Bash
python3 text_optimization.py --loss_type log_prob_pos --seed 0
```


Run the following to optimize prompts that, when prepended to "I am happy", still cause the OPT model to generate text with negative sentiment according to the sentiment classifier. 
```Bash
python3 text_optimization.py --loss_type log_prob_neg --seed 0
```

#### Threat Model 2: Target Characters (4.2.2)
Run the following to optimize prompts that cause the OPT model to generate text with a large number of occurrences of some target character. The target character can be specified with the `--target_string argument`. In the example command below the target character used is "t". In the paper we provide results for the following target characters: a, e, i, f, t, y, z, q. 

```Bash
python3 text_optimization.py --loss_type target_occurrences --target_string t --seed 0
```
