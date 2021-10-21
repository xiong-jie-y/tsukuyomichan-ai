
from transformers import AutoTokenizer
import onnxruntime

import torch

from utils.file import get_model_file_from_gdrive

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class ONNXSentenceTransformer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

        providers = ['CPUExecutionProvider']

        self.session = onnxruntime.InferenceSession(
            get_model_file_from_gdrive("https://drive.google.com/uc?id=1I32hr_6VTPwgIRFuPJnrSSPsedkxvtoc"),
            providers=providers
        )

    def encode(self, text):
        sentences = [text]
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        model_output = self.session.run(
            [], 
            {"input_ids": encoded_input["input_ids"].tolist(), 
            "attention_mask": encoded_input["attention_mask"].tolist()})
        model_output[0] = torch.tensor(model_output[0])
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        return sentence_embeddings[0]