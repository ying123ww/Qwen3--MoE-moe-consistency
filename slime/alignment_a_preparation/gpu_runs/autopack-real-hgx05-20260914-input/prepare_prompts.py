"""CPU-only official-script prompt fixture, epoch 0, no length filtering."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import random
from transformers import AutoTokenizer

root = Path('/public/home/yingxy2024/moe-consistency/slime')
sys.path.insert(0, str(root))
from slime.utils.data import Dataset
model = Path('/storage/data/yingxy2024/Models/Qwen3-30B-A3B')
dataset = Path('/storage/data/yingxy2024/data/dapo-math-17k/dapo-math-17k.jsonl')
script = root / 'scripts/run-qwen3-30B-A3B.sh'
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
rows = []
for n, raw in enumerate(dataset.read_bytes().splitlines(keepends=True), 1):
    if raw.strip():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        rows.append((n, raw, row))
# Invoke the repository shuffle directly with source indices as stand-in samples.
# No length filter is configured, so this is the same full dataset index space.
d = Dataset.__new__(Dataset)
d.seed, d.epoch_id = 42, -1
d.origin_samples = list(range(len(rows)))
d.samples = d.origin_samples
d.shuffle(0)
expected = list(range(len(rows)))
random.Random(42).shuffle(expected)
assert d.samples == expected
samples = []
for group, source_idx in enumerate(d.samples[:32]):
    line, raw, row = rows[source_idx]
    messages = row['prompt']
    rendered = tokenizer.apply_chat_template(messages, tools=None, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.encode(rendered, add_special_tokens=False)
    direct = tokenizer.apply_chat_template(messages, tools=None, tokenize=True, add_generation_prompt=True)
    if hasattr(direct, 'keys'):
        direct = direct['input_ids']
    assert tokens == direct and tokens
    for repeat in range(8):
        samples.append({'index': len(samples), 'group_index': group, 'repeat_index': repeat,
                        'source_row_1based': line, 'source_dataset_index': source_idx,
                        'source_sha256': hashlib.sha256(raw).hexdigest(), 'prompt': messages,
                        'label': row.get('label'), 'prompt_tokens': tokens, 'prompt_length': len(tokens),
                        'prompt_token_sha256': hashlib.sha256(struct.pack('<'+'q'*len(tokens), *tokens)).hexdigest()})
assert len(samples) == 256
assert len({s['source_row_1based'] for s in samples}) == 32
lengths = [s['prompt_length'] for s in samples[::8]]
sourcefiles = [script, root/'slime/utils/data.py', root/'slime/rollout/data_source.py', root/'slime/utils/arguments.py', root/'slime/rollout/sglang_rollout.py']
manifest = {'purpose': 'Official checked-in script configuration prompt fixture, not proof of a prior production run or full-distribution benchmark',
            'dataset': str(dataset), 'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
            'model': str(model), 'script_original_dataset_path': '/root/dapo-math-17k/dapo-math-17k.jsonl',
            'script_original_model_path': '/root/Qwen3-30B-A3B',
            'selection': 'Dataset.shuffle(epoch=0), default rollout_seed=42, first32 prompt groups,8 repeats each; no prompt length filtering',
            'dataset_valid_rows': len(rows), 'rollout_seed': 42, 'epoch_id': 0,
            'configuration_status': 'Reconstructed from current repository script/defaults; not verified production invocation',
            'source_files_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sourcefiles},
            'chat_template': {'add_generation_prompt': True, 'tools': None, 'other_kwargs': 'defaults; no thinking override',
                              'sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()},
            'tokenizer_files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in model.glob('*token*') if p.is_file()},
            'planned_config': {'rollout_batch_size': 32, 'n_samples_per_prompt': 8, 'global_batch_size': 256,
                              'max_new_tokens': 8192, 'temperature': 1.0, 'top_p': 1.0, 'top_k': -1,
                              'max_tokens_per_gpu': 20480, 'rollout_max_prompt_len': None,
                              'tensor_model_parallel_size': 4, 'expert_model_parallel_size': 8},
            'prompt_length_stats': {'min': min(lengths), 'max': max(lengths), 'mean': sum(lengths)/len(lengths), 'lengths': lengths},
            'no_generated_response_routes_or_logprobs': True, 'samples': samples}
out = Path(__file__).parent/'prompts.json'
out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
print(json.dumps({'output': str(out), 'samples': len(samples), 'prompt_groups':32,
                  'source_rows': [s['source_row_1based'] for s in samples[::8]], 'length_stats': manifest['prompt_length_stats']}))
