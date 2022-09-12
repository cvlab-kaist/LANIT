import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

class PromptLearner(nn.Module):
    def __init__(self, device, K, classes, clip_model, init_prompt='a photo of the {}.', rand_token_len=4):
        super().__init__()
        #self.args = args

        prefix, suffix = [x.strip() for x in init_prompt.split('{}')]
        
        self.K = K
        self.rand_token_len = rand_token_len
     
        prompt_prefix = clip.tokenize(prefix).to(device)
        prompt_suffix = clip.tokenize(suffix).to(device)
        class_tokens = clip.tokenize(classes).to(device)

        self.n_prompt_prefix = self.count_token(prompt_prefix).item()
        self.n_prompt_suffix = self.count_token(prompt_suffix).item()
        self.len_classes = self.count_token(class_tokens)
        
        self.max_len = prompt_prefix.shape[-1]

        with torch.no_grad():
            prefix_embedding = clip_model.token_embedding(prompt_prefix).squeeze(0)
            suffix_embedding = clip_model.token_embedding(prompt_suffix).squeeze(0)
            class_embedding = clip_model.token_embedding(class_tokens)

            sos_token = prefix_embedding[0] # 항상 일정하니까 하나로 둠.
            eos_token = prefix_embedding[self.n_prompt_prefix + 1]
            padding = prefix_embedding[-1] # 항상 일정하니까 하나로 둠.

        class_embeddings = []
        for i, l in enumerate(self.len_classes):
            class_embeddings.append(nn.Parameter(
                class_embedding[i, 1:l+1] # 클래스도 딱 ctx부분만. 
            ))

        rand_tokens = torch.zeros(K - len(classes), rand_token_len, class_embedding.size(-1)).to(device)
        nn.init.normal_(rand_tokens, std=0.02)

        self.rand_tokens = nn.Parameter(rand_tokens) # K - len(classes), 1, 512

        # prefix도 사이만.
        self.prefix_tokens = nn.Parameter(prefix_embedding[1:1 + self.n_prompt_prefix]) # (length_prefix, 512)

        with torch.no_grad():
            self.class_tokens = nn.ParameterList(class_embeddings) # List of l, 512
            # suffix도 사이만.
            self.suffix_tokens = nn.Parameter(suffix_embedding[1:1 + self.n_prompt_suffix]) # n_prompt, 512
            """ class token 과 suffix_token은 update 안함 -> prefix만 update 진행. """
            self.class_tokens.requires_grad = False
            self.suffix_tokens.requires_grad = False

        self.register_buffer('sos_token', sos_token)
        self.register_buffer('eos_token', eos_token)
        self.register_buffer('padding', padding)

    def count_token(self, x):
        return (x != 0).sum(1) - 2
    
    def get_embedding(self):
        embeddings = []
        for i, cls in enumerate(self.class_tokens):
            embed = torch.cat((
                self.sos_token[None],
                self.prefix_tokens,
                cls,
                self.suffix_tokens,
                self.eos_token[None]
            ))
            padding = self.padding[None].repeat(self.max_len - embed.size(0), 1)
            embeddings.append(torch.cat((embed, padding), 0))
        embeddings = torch.stack(embeddings)
        
        rand_len = self.rand_tokens.size(0)

        

        rand_embeddings = torch.cat((
            self.sos_token[None, None].repeat(rand_len, 1, 1),
            self.prefix_tokens[None].repeat(rand_len, 1, 1),
            self.rand_tokens,
            self.suffix_tokens[None].repeat(rand_len, 1, 1),
            self.eos_token[None, None].repeat(rand_len, 1, 1),
        ), dim=1)
        rand_embeddings = torch.cat((
            rand_embeddings,
            self.padding[None, None].repeat(rand_len, self.max_len - rand_embeddings.size(1), 1)
        ), dim=1)
        
        return torch.cat((embeddings, rand_embeddings), 0)
    
    def forward(self, clip_model):
        x = self.get_embedding()

        x = x + clip_model.positional_embedding.to(clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.ln_final(x).type(clip_model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x_cls = x[range(len(self.len_classes)), self.len_classes + self.n_prompt_prefix + self.n_prompt_suffix + 1] @ clip_model.text_projection
        x_rand = x[len(self.len_classes):, self.rand_token_len + self.n_prompt_prefix + self.n_prompt_suffix + 1] @ clip_model.text_projection

        return torch.cat((x_cls, x_rand), 0)