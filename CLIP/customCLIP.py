from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torch.utils.checkpoint import checkpoint

from CLIP.simple_tokenizer import SimpleTokenizer as _Tokenizer


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        # The EOT token has the largest token id in each CLIP sequence.
        eot_indices = tokenized_prompts.argmax(dim=-1)
        x = x[
            torch.arange(x.shape[0], device=x.device),
            eot_indices,
        ] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        self.args = args
        n_cls = len(classnames)
        n_ctx = args.N_CTX
        ctx_init = args.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init:
            ctx_init = ctx_init.replace('_', ' ')
            n_ctx = len(ctx_init.split(' '))
            prompt = tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if args.CSC:
                print('Initializing class-specific contexts')
                ctx_vectors = torch.empty(
                    n_cls,
                    n_ctx,
                    ctx_dim,
                    dtype=dtype,
                )
            else:
                print('Initializing a generic context')
                ctx_vectors = torch.empty(
                    n_ctx,
                    ctx_dim,
                    dtype=dtype,
                )
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = ' '.join(['X'] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f'Number of context words (tokens): {n_ctx}')

        self.ctx = nn.Parameter(ctx_vectors)
        self.prompt_prefix = prompt_prefix
        self.dtype = dtype

        classnames = [name.replace('_', ' ') for name in classnames]
        name_lens = [
            len(_tokenizer.encode(name))
            for name in classnames
        ]
        prompts = [
            prompt_prefix + ' ' + name + '.'
            for name in classnames
        ]

        tokenized_prompts = torch.cat([
            tokenize(prompt)
            for prompt in prompts
        ])
        with torch.no_grad():
            embedding = clip_model.token_embedding(
                tokenized_prompts
            ).type(dtype)

        self.register_buffer(
            'token_prefix',
            embedding[:, :1, :],
        )
        self.register_buffer(
            'token_suffix',
            embedding[:, 1 + n_ctx:, :],
        )

        # [SGG dynamic prompt]
        # Dynamic S-P-O prompts need token embeddings at runtime. Keep the
        # frozen weight on the correct device without duplicating it in
        # checkpoints or exposing it to the optimizer.
        self.register_buffer(
            'token_embedding_weight',
            clip_model.token_embedding.weight.detach(),
            persistent=False,
        )

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = args.CLASS_TOKEN_POSITION

    def _expand_context(self, class_indices):
        # [SGG dynamic prompt]
        # Reuse each predicate's learned context for every S-O pair.
        ctx = self.ctx
        if ctx.dim() == 2:
            return ctx.unsqueeze(0).expand(
                len(class_indices),
                -1,
                -1,
            )

        class_indices = class_indices.to(
            device=ctx.device,
            dtype=torch.long,
        )

        if len(class_indices) > 0:
            if class_indices.min().item() < 0:
                raise ValueError('Class indices must be non-negative')
            if class_indices.max().item() >= self.n_cls:
                raise ValueError(
                    'Class index exceeds the learned context count: '
                    f'max={class_indices.max().item()}, '
                    f'n_cls={self.n_cls}'
                )

        return ctx.index_select(0, class_indices)

    def _assemble_prompts(
        self,
        prefix,
        context,
        suffix,
        name_lens,
    ):
        if self.class_token_position == 'end':
            return torch.cat(
                [prefix, context, suffix],
                dim=1,
            )

        if self.class_token_position == 'middle':
            half_n_ctx = self.n_ctx // 2
            prompts = []

            for index, name_len in enumerate(name_lens):
                prompt = torch.cat(
                    [
                        prefix[index:index + 1],
                        context[index:index + 1, :half_n_ctx],
                        suffix[index:index + 1, :name_len],
                        context[index:index + 1, half_n_ctx:],
                        suffix[index:index + 1, name_len:],
                    ],
                    dim=1,
                )
                prompts.append(prompt)

            return torch.cat(prompts, dim=0)

        if self.class_token_position == 'front':
            prompts = []

            for index, name_len in enumerate(name_lens):
                prompt = torch.cat(
                    [
                        prefix[index:index + 1],
                        suffix[index:index + 1, :name_len],
                        context[index:index + 1],
                        suffix[index:index + 1, name_len:],
                    ],
                    dim=1,
                )
                prompts.append(prompt)

            return torch.cat(prompts, dim=0)

        raise ValueError(
            'Unsupported class token position: '
            f'{self.class_token_position}'
        )

    def forward(self):
        if self.ctx.dim() == 2:
            class_indices = torch.zeros(
                self.n_cls,
                dtype=torch.long,
                device=self.ctx.device,
            )
        else:
            class_indices = torch.arange(
                self.n_cls,
                device=self.ctx.device,
            )

        context = self._expand_context(class_indices)

        return self._assemble_prompts(
            self.token_prefix,
            context,
            self.token_suffix,
            self.name_lens,
        )

    def forward_dynamic(
        self,
        classnames,
        class_indices,
    ):
        """Build prompts whose text changes while contexts remain shared."""
        # [SGG dynamic prompt]
        # Unlike forward(), the suffix text is rebuilt for every S-P-O
        # candidate while the original predicate contexts are preserved.
        if len(classnames) != len(class_indices):
            raise ValueError(
                'Dynamic prompt count mismatch: '
                f'classnames={len(classnames)}, '
                f'class_indices={len(class_indices)}'
            )

        if len(classnames) == 0:
            raise ValueError('Dynamic prompt input must not be empty')

        classnames = [
            name.replace('_', ' ')
            for name in classnames
        ]
        name_lens = [
            len(_tokenizer.encode(name))
            for name in classnames
        ]
        prompt_strings = [
            self.prompt_prefix + ' ' + name + '.'
            for name in classnames
        ]

        tokenized_prompts = torch.cat([
            tokenize(prompt)
            for prompt in prompt_strings
        ]).to(self.ctx.device)

        embedding = F.embedding(
            tokenized_prompts.long(),
            self.token_embedding_weight,
        ).type(self.dtype)

        prefix = embedding[:, :1, :]
        suffix = embedding[:, 1 + self.n_ctx:, :]
        context = self._expand_context(class_indices)

        prompts = self._assemble_prompts(
            prefix,
            context,
            suffix,
            name_lens,
        )

        return prompts, tokenized_prompts


class CustomCLIP(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(
            args,
            classnames,
            clip_model,
        )
        self.tokenized_prompts = (
            self.prompt_learner.tokenized_prompts
        )
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def encode_dynamic_text(
        self,
        classnames,
        class_indices,
        chunk_size=256,
        use_checkpoint=False,
    ):
        """Encode dynamic prompt text in bounded-size chunks."""
        # [SGG dynamic prompt]
        # Chunking bounds text-transformer memory without changing logits.
        if chunk_size <= 0:
            raise ValueError(
                f'chunk_size must be positive, got {chunk_size}'
            )

        if len(classnames) != len(class_indices):
            raise ValueError(
                'Dynamic text input count mismatch: '
                f'classnames={len(classnames)}, '
                f'class_indices={len(class_indices)}'
            )

        text_features = []

        for start in range(0, len(classnames), chunk_size):
            stop = min(
                start + chunk_size,
                len(classnames),
            )

            prompts, tokenized_prompts = (
                self.prompt_learner.forward_dynamic(
                    classnames[start:stop],
                    class_indices[start:stop],
                )
            )

            if self.training and use_checkpoint:
                features = checkpoint(
                    self.text_encoder,
                    prompts,
                    tokenized_prompts,
                    use_reentrant=False,
                )
            else:
                features = self.text_encoder(
                    prompts,
                    tokenized_prompts,
                )

            text_features.append(features)

        return torch.cat(text_features, dim=0)

    def forward(self, image):
        image_features = self.image_encoder(
            image.type(self.dtype)
        )

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(
            prompts,
            tokenized_prompts,
        )

        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True,
        )
        text_features = text_features / text_features.norm(
            dim=-1,
            keepdim=True,
        )

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        raise ValueError
        return logits


_tokenizer = _Tokenizer()


def tokenize(
    texts: Union[str, List[str]],
    context_length: int = 77,
    truncate: bool = False,
    return_sot=True,
) -> Union[torch.IntTensor, torch.LongTensor]:
    """Tokenize text using the original CLIP vocabulary."""
    del return_sot

    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder['<|startoftext|>']
    eot_token = _tokenizer.encoder['<|endoftext|>']
    all_tokens = [
        [sot_token] + _tokenizer.encode(text) + [eot_token]
        for text in texts
    ]

    # [Compatibility]
    # Import packaging.version explicitly because some packaging releases
    # do not expose the version module from the top-level package.
    if version.parse(
        torch.__version__
    ) < version.parse('1.8.0'):
        result = torch.zeros(
            len(all_tokens),
            context_length,
            dtype=torch.long,
        )
    else:
        result = torch.zeros(
            len(all_tokens),
            context_length,
            dtype=torch.int,
        )

    for index, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f'Input {texts[index]} is too long '
                    f'for context length {context_length}'
                )

        result[index, :len(tokens)] = torch.tensor(tokens)

    return result
