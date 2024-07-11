def apply_attention_mask(x, tokenizer):
    return x.ne(tokenizer.pad_token_id).to(int)
