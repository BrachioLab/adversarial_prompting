# Adversarial Prompting for Black Box Foundation Models
Official implementation for Adversarial Prompting for Black Box Foundation Models
https://arxiv.org/abs/2302.04237

See directions below to run the code and reproduce all results provided in the paper. 

## Getting Started

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

The script to run image optimization is in `scripts/image_optimization.py` and the script to run text optimization is in `run_text_exp.py`.

## Image generation 
Here we provide commands to replicate the image generation results for each of the three threat models discussed in Section 3.2. By default all optimization tasks will be run using TuRBO since we found this to be the more successful optimization method. 

To instead run any task with the Square Attack optimization method, pass in the argument `--square_attack True`. 

#### Threat Model 1: Unrestricted Prompts
For this task, run the following command once with each optimal/target imagenet class you'd like to use. Change the target class for each run using the `--optimal_class` argument. In the example below we use the target class "bus". 

```Bash
python3 image_optimization.py --optimal_class bus --max_allowed_calls_without_progress 1000 --max_n_calls 5000 --seed 0
```

#### Threat Model 2: Restricted Prompts
For this task, run the following command once with each desired optimal/target imagenet class. To exactly reproduce the paper results, use the target classes listed in Tables 2 and 3. Change the target class for each run using the `--optimal_class` argument. In the example below we use the target class bus. 

```Bash
python3 image_optimization.py --optimal_class bus --max_allowed_calls_without_progress 1000 --max_n_calls 5000 --seed 0 --exclude_high_similarity_tokens True
```

#### Threat Model 3: Restricted Prepending Prompts
For this task, run the following command once with each desired optimal/target Imagenet class. To exactly reproduce the paper results, use the target classes listed in Tables 2 and 3. Change the target class for each run using the `--optimal_class` argument. In the example below we use the target class "bus". 
Additionally, for each target class, run once with each of the three versions so of the prepending task. With `--prepend_task_version 1` we optimize prompts prepended to "a picture of a dog", with `--prepend_task_version 2` we optimize prompts prepended to "a picture of a mountain", and with `--prepend_task_version 3` we optimize prompts prepended to "a picture of the ocean". You can modify the dictionary in `utils/constants.py` to add additional prepending task options if desired. 

```Bash
python3 image_optimization.py --optimal_class bus --max_allowed_calls_without_progress 3000 --max_n_calls 10000 --seed 0 --exclude_high_similarity_tokens True --prepend_task True --prepend_task_version 1
```

## Text generation 
Here we provide commands to replicate the text generation results. The text generation pipeline is more flexible to allow for custom loss functions and language models.


Run the following to optimize prompts that, when prepended to `Explain list comprehension in Python.`, cause Vicuna 13B-v1.1 to generate text maximizing the log perplexity, aiming to generate nonsensical text.

```Bash
python3 run_text_exp.py --loss_type perplexity --seed 0 --language_model vicuna1.1 --embedding_model tinybert --seed_text "Explain list comprehension in Python."
```
### Functionality for Various Language Models
Here we choose `Vicuna 13B-v1.1`, we provide functionality for `OPT-125M/350M,1.3B,2.7B,6.7B,13B,30B, GPT2, LLaMA-7B/13B/30B, StableLM, Vicuna 13B-v1.1`. It is easy to add your own custom language model to this pipeline. To do so, you can modify `get_language_model` in `scripts/run_text_exp.py` and add your custom language model to `utils/text_utils/language_model.py`.

### Functionality for Various Loss Functions
Here we choose the perplexity loss, but it is also easy to add your own loss function. We currently implement loss functions `count_letter`, `emotion`, `toxicity`, and `perplexity`. To implement your own loss function, you can modify `get_loss_fn` in `scripts/run_text_exp.py` and add your own loss to `utils/text_utils/text_losses.py`.

### Questions
For any questions, please contact us at `pchao@wharton.upenn.edu`.