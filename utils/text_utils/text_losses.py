from typing import List
import torch
import abc
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline, pipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class LossFunction():
    def __init__(self):
        self.goal = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device_num = 0 if torch.cuda.is_available() else -1
    @abc.abstractmethod
    def __call__(self, generated_texts: List[tuple[str,List[str]]], **kwargs) -> torch.Tensor:
        '''
        Compute the loss on the generated text and prompts
    
        '''
        pass

class CountLetterLoss(LossFunction):
    def __init__(self, letter = 't'):
        self.goal = 'maximize'
        self.letter = letter
        
    def __call__(self, generated_texts: List[tuple[str, List[str]]]) -> torch.Tensor:
        n_prompts = len(generated_texts)
        n_gen_texts = len(generated_texts[0][1])
        losses = torch.zeros((n_prompts,n_gen_texts))
        for i in range(n_prompts):
            prompt, cur_generated_texts = generated_texts[i]
            losses[i,:] = torch.tensor([text.lower().count(self.letter) for text in cur_generated_texts])
        return losses
        
class ToxicityLoss(LossFunction):
    def __init__(self):
        super().__init__()
        self.goal = 'maximize'
        
        # https://huggingface.co/martin-ha/toxic-comment-model
        model_path = "martin-ha/toxic-comment-model"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        self.pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer,device=self.device_num)

    def __call__(self,  generated_texts: List[tuple[str, List[str]]]) -> torch.Tensor:
        n_prompts = len(generated_texts)
        n_gen_texts = len(generated_texts[0][1])
        preprocessed_text = []
        for i in range(n_prompts):
            prompt, cur_generated_texts = generated_texts[i]            
            preprocessed_text.extend([text for text in cur_generated_texts])
        toxicity_preds = self.pipeline(preprocessed_text)
        losses = [pred['score'] if pred['label'] == 'toxic'  else 1 - pred['score'] for pred in toxicity_preds]
        losses = torch.tensor(losses).reshape(n_prompts, n_gen_texts)
        return losses

class EmotionLoss(LossFunction):
    def __init__(self, emotion_class):
        super().__init__()
        self.goal = 'maximize'
        self.emotion_class = emotion_class
        model_path = "j-hartmann/emotion-english-distilroberta-base"
        self.classifier = pipeline("text-classification", model=model_path, top_k=None, device=self.device_num)

    def __call__(self, generated_texts: List[tuple[str, List[str]]]) -> torch.Tensor:
        n_prompts = len(generated_texts)
        n_gen_texts = len(generated_texts[0][1])
        preprocessed_text = []
        for i in range(n_prompts):
            prompt, cur_generated_texts = generated_texts[i]
            preprocessed_text.extend([text for text in cur_generated_texts])
        emotion_preds = self.classifier(preprocessed_text)
        
        # The first entry is the 'anger' emotion class we care about, the order is:
        # anger, disgust, fear, joy, neutral, sadness, surprise
        losses = [pred[i]['score']  for pred in emotion_preds for i in range(len(pred)) if pred[i]['label'] == self.emotion_class]
        losses = torch.tensor(losses).reshape(n_prompts, n_gen_texts)
        return losses

class PerplexityLoss(LossFunction):
    def __init__(self):
        super().__init__()
        self.goal = 'maximize'
        model_name = "gpt2"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

    def __call__(self, generated_texts: List[tuple[str, List[str]]]):
        n_prompts = len(generated_texts)
        n_gen_texts = len(generated_texts[0][1])
        preprocessed_text = []
        for i in range(n_prompts):
            prompt, cur_generated_texts = generated_texts[i]
            preprocessed_text.extend([text for text in cur_generated_texts])
        with torch.no_grad():
            inputs = self.tokenizer(preprocessed_text, return_tensors="pt", padding=True, truncation=True)
            inputs.to(self.device)
            outputs = self.model(**inputs)
            log_probs = outputs.logits.log_softmax(dim=-1)
            token_log_probs = torch.gather(log_probs, -1, inputs.input_ids.unsqueeze(-1)).squeeze(-1)

            masked_indices = inputs.attention_mask.to(bool)
            token_log_probs_masked = token_log_probs[masked_indices]

            sequence_lengths = inputs.attention_mask.sum(dim=-1) 
            sentence_log_probs = token_log_probs_masked.split(sequence_lengths.tolist())
            batch_log_probs = [sentence_log_probs[i].sum().item() for i in range(len(sentence_log_probs))]
            batch_lengths = [len(sentence_log_probs[i]) for i in range(len(sentence_log_probs))]
            # Log perplexity
            batch_perplexities = [(-log_prob / length) if length > 0 else 0 for log_prob, length in zip(batch_log_probs, batch_lengths)]


        losses = torch.tensor(batch_perplexities).reshape(n_prompts, n_gen_texts)
        return losses
