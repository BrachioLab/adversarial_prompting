import sys 
sys.path.append("../")
from utils.objective import Objective 
from utils.text_utils.text_losses import *
from utils.token_embedder import OPTEmbedding
from utils.text_utils.language_model import *

class TextGenerationObjective(Objective):
    def __init__(
        self,
        n_tokens,
        prompts_to_texts,
        texts_to_losses,
        token_embedder,
        lb = None,
        ub = None,
        **kwargs,
    ):

        self.prompts_to_texts = prompts_to_texts
        self.texts_to_losses = texts_to_losses
        self.token_embedder = token_embedder

        self.n_tokens = n_tokens
        super().__init__(
            num_calls = 0,
            task_id = 'adversarial4',
            dim = n_tokens * self.token_embedder.embed_dim,
            lb = lb,
            ub = ub,
            **kwargs,
        ) 

    def query_oracle(self, embeddings):
        if not torch.is_tensor(embeddings):
            embeddings = torch.tensor(embeddings, dtype=torch.float16)
        embeddings = embeddings.cuda() 
        embeddings = embeddings.reshape(-1, self.n_tokens, self.token_embedder.embed_dim) 
        prompts = self.token_embedder.batched_embeddings_to_tokens(embeddings)
        generated_text = self.prompts_to_texts(prompts)
        losses = self.texts_to_losses(generated_text)
        mean_loss = torch.mean(losses, axis = 1)
        return prompts, mean_loss, generated_text

