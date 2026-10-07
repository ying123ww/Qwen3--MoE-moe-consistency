"""Freeze unique short DAPO prompts using the real HF chat template; CPU only."""
import hashlib
import json
from pathlib import Path
import struct
from transformers import AutoTokenizer

model = Path('/storage/data/yingxy2024/Models/Qwen3-30B-A3B')
dataset = Path('/storage/data/yingxy2024/data/dapo-math-17k/dapo-math-17k.jsonl')
output = Path(__file__).parent / 'prompts.json'
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
samples, seen, excluded = [], set(), []
for row_number, raw in enumerate(dataset.read_bytes().splitlines(keepends=True), 1):
    row = json.loads(raw)
    messages = row['prompt']
    canonical = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    prompt_hash = hashlib.sha256(canonical.encode()).hexdigest()
    if prompt_hash in seen:
        excluded.append({'row': row_number, 'reason': 'duplicate_prompt'})
        continue
    seen.add(prompt_hash)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer(rendered, add_special_tokens=False)['input_ids']
    direct = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(direct, 'keys'):
        direct = direct['input_ids']
    assert tokens == direct
    if not 0 < len(tokens) <= 384:
        excluded.append({'row': row_number, 'reason': 'prompt_length_outside_1_384', 'tokens': len(tokens)})
        continue
    samples.append({'index': len(samples), 'source_row_1based': row_number,
                    'source_sha256': hashlib.sha256(raw).hexdigest(),
                    'prompt_sha256': prompt_hash, 'prompt': messages, 'label': row.get('label'),
                    'prompt_tokens': tokens, 'prompt_length': len(tokens),
                    'prompt_token_sha256': hashlib.sha256(struct.pack('<' + 'q' * len(tokens), *tokens)).hexdigest()})
    if len(samples) == 32:
        break
assert len(samples) == 32 and len({s['prompt_token_sha256'] for s in samples}) == 32
manifest = {'purpose': '32 distinct real DAPO prompts for multi-slot route capture; not representative full-dataset performance evidence',
            'dataset': str(dataset), 'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
            'model': str(model), 'selection': 'first 32 unique prompts with 1..384 tokens in dataset file order',
            'rows_scanned': row_number, 'excluded': excluded,
            'chat_template': {'add_generation_prompt': True, 'other_kwargs': 'tokenizer defaults (no thinking override)',
                              'sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()},
            'tokenizer_files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in model.glob('*token*') if p.is_file()},
            'response_tokens_planned': 128, 'max_total_tokens_planned': 512,
            'no_generated_response_routes_or_logprobs': True, 'samples': samples}
output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'output': str(output), 'samples': len(samples), 'rows_scanned': row_number,
                  'prompt_lengths': [s['prompt_length'] for s in samples], 'excluded': excluded}))
