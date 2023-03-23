FROM continuumio/miniconda3
COPY adv_env.txt adv_env.txt
RUN conda create --name adv_env --file adv_env.txt
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install accelerate
RUN pip install transformers 
RUN pip install diffusers
RUN pip install nltk
RUN pip install gpytorch 
RUN pip install botorch
RUN pip install wandb
RUN pip install fire 
RUN pip install matplotlib
RUN pip install pandas
# ENV WANDB_API_KEY=$YOUR_WANDB_API_KEY
""" If you are using weights and biases (wandb) to track
    results, before building the container, 
    uncomment the line above add YOUR_WANDB_API_KEY 
    to allow for easy wandb sign in wihtin your containder 
"""

"""
Example to build container and start interactive terminal:
$ docker build -t $YOUR_DOCKER_USERNAME/advenv .
$ docker run -v $PATH/adversarial_img_bo/:/workspace/adversarial_bo/ -it $YOUR_DOCKER_USERNAME/advenv
"""
