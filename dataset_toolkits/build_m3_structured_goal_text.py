"""Append label-derived structured builder goals to an existing M3 text cache."""
import argparse
import hashlib
import json
from pathlib import Path

from miworld.data.inserted_policy_dataset import structured_goal_text


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, source_hashes = [], {}
    for path in args.index:
        path = Path(path)
        source_hashes[str(path.resolve())] = sha(path)
        rows.extend(json.loads(line) for line in path.open() if line.strip())
    builders = [row for row in rows if row['family'] == 'language_builder']
    goals = sorted({structured_goal_text(row) for row in builders})
    if not goals:
        raise ValueError('no language_builder structured goals found')

    catalog = json.loads(args.base_catalog.read_text())
    old = sorted(catalog['texts'], key=lambda row: row['text_id'])
    if [row['text_id'] for row in old] != list(range(len(old))):
        raise ValueError('base catalog text ids must be contiguous')
    existing = {row['text'] for row in old}
    missing = [goal for goal in goals if goal not in existing]
    catalog['texts'] = old + [
        {'text_id': len(old) + index, 'text': goal,
         'sha256': hashlib.sha256(goal.encode()).hexdigest()}
        for index, goal in enumerate(missing)
    ]
    catalog['encoder'] = {**catalog['encoder'], 'status': 'append-pending',
                          'cache_file': 'text_cache.safetensors'}
    catalog['structured_goal_extension'] = {
        'schema': 'simple_task_goal_v1',
        'format': 'task=<family> material=<material> size=<x>x<y>x<z>',
        'base_catalog': str(args.base_catalog.resolve()),
        'base_catalog_sha256': sha(args.base_catalog),
        'base_cache': str(args.base_cache.resolve()),
        'base_count': len(old),
        'goal_count': len(goals),
        'added_count': len(missing),
        'source_indices': source_hashes,
        'builder_rows': len(builders),
        'all_fields_validated': True,
    }
    args.output.mkdir(parents=True)
    (args.output / 'text_catalog.json').write_text(json.dumps(catalog, indent=2) + '\n')
    (args.output / 'structured_goal_manifest.json').write_text(json.dumps(
        catalog['structured_goal_extension'], indent=2) + '\n')
    print(json.dumps(catalog['structured_goal_extension']), flush=True)


def encode(args):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer, T5EncoderModel

    catalog_path = args.output / 'text_catalog.json'
    catalog = json.loads(catalog_path.read_text())
    extension = catalog['structured_goal_extension']
    target = args.output / 'text_cache.safetensors'
    if target.exists():
        raise FileExistsError(target)
    old = load_file(str(args.base_cache), device='cpu')
    count = int(extension['base_count'])
    if old['encoder_hidden'].shape[0] != count:
        raise ValueError('base cache and catalog counts disagree')
    texts = [row['text'] for row in catalog['texts']]
    max_length = int(catalog['encoder']['max_length'])
    source = args.model or catalog['encoder']['name']
    tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=args.model_cache)
    model = T5EncoderModel.from_pretrained(source, cache_dir=args.model_cache).eval().to(args.device)

    def run(items):
        tokens = tokenizer(items, padding='max_length', truncation=True,
                           max_length=max_length, return_tensors='pt')
        mask = tokens['attention_mask'].byte()
        with torch.no_grad():
            hidden = model(**{key: value.to(args.device) for key, value in tokens.items()}
                           ).last_hidden_state.cpu()
        return hidden.to(old['encoder_hidden'].dtype), mask.to(old['attention_mask'].dtype)

    probes = [index for index in range(count) if texts[index]][:8]
    hidden, mask = run([texts[index] for index in probes])
    if not torch.equal(mask, old['attention_mask'][probes]):
        raise ValueError('T5 tokenizer does not reproduce base cache masks')
    valid = mask.bool()
    reference, actual = old['encoder_hidden'][probes][valid].float(), hidden[valid].float()
    cosine = float(F.cosine_similarity(actual, reference, dim=-1).mean())
    mae = float((actual - reference).abs().mean())
    if cosine < .999 or mae > .01:
        raise ValueError(f'T5 encoder/cache mismatch: cosine={cosine}, mae={mae}')

    hidden_parts, mask_parts = [old['encoder_hidden']], [old['attention_mask']]
    for start in range(count, len(texts), args.batch_size):
        encoded, encoded_mask = run(texts[start:start + args.batch_size])
        hidden_parts.append(encoded); mask_parts.append(encoded_mask)
        print(json.dumps({'encoded': min(start + args.batch_size, len(texts)) - count,
                          'added': len(texts) - count}), flush=True)
    save_file({'encoder_hidden': torch.cat(hidden_parts).contiguous(),
               'attention_mask': torch.cat(mask_parts).contiguous()}, str(target))
    check = load_file(str(target), device='cpu')
    for key in old:
        if not torch.equal(check[key][:count], old[key]):
            raise ValueError(f'base cache vectors changed: {key}')
    catalog['encoder']['status'] = 'complete'
    extension.update(base_cache=str(args.base_cache.resolve()),
                     base_cache_sha256=sha(args.base_cache), cache_sha256=sha(target),
                     encoder_source=source, probe_cosine=cosine, probe_mae=mae,
                     original_vectors_bitwise_preserved=True)
    catalog_path.write_text(json.dumps(catalog, indent=2) + '\n')
    (args.output / 'structured_goal_manifest.json').write_text(
        json.dumps(extension, indent=2) + '\n')
    print(json.dumps(extension), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('build', 'encode'))
    parser.add_argument('--index', type=Path, action='append', default=[])
    parser.add_argument('--base-catalog', type=Path)
    parser.add_argument('--base-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model')
    parser.add_argument('--model-cache')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.mode == 'build':
        if not args.index or args.base_catalog is None:
            parser.error('build requires --index and --base-catalog')
        build(args)
    else:
        encode(args)


if __name__ == '__main__':
    main()
