"""Append causally aligned component instructions without changing samples or loss."""
import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path


def canonical(value): return ' '.join(str(value or '').strip().split())


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event_timeline(events):
    result = {}
    for event in events:
        kind = event['event']
        if kind not in ('component_started', 'task_completed'): continue
        frame = int(event['observation_frame'])
        if 'transition_index' in event and frame != int(event['transition_index']) + 1:
            raise ValueError('component boundary must be the observation AFTER its transition')
        text = canonical(event.get('component_text')) if kind == 'component_started' else ''
        if kind == 'component_started' and not text: raise ValueError('empty component instruction')
        result[frame] = {'observation_frame': frame, 'text': text,
                         'component_id': event.get('component_id', '') if text else ''}
    if not any(e['text'] for e in result.values()): raise ValueError('no component instructions')
    return [result[i] for i in sorted(result)]


def text_at(timeline, frame):
    i = bisect_right([e['observation_frame'] for e in timeline], frame) - 1
    return timeline[i]['text'] if i >= 0 else ''


def build(args):
    if args.output.exists(): raise FileExistsError(args.output)
    timelines, evidence, texts = {}, [], set()
    indices = {}
    for name in ('train', 'train_replay'):
        rows = [json.loads(x) for x in (args.source/(name+'.jsonl')).open()]
        out = []
        for row in rows:
            timeline = []
            if row['family'] == 'language_builder':
                ep = args.dataset_root/row['episode_path']
                if row['episode_id'] not in timelines:
                    events = [json.loads(x) for x in (ep/'events.jsonl').open()]
                    timeline = event_timeline(events)
                    timelines[row['episode_id']] = timeline
                    texts.update(e['text'] for e in timeline)
                    evidence.append({'episode_id': row['episode_id'], 'timeline': timeline,
                                     'event_sha256': sha(ep/'events.jsonl')})
                timeline = timelines[row['episode_id']]
                if not text_at(timeline, row['anchor']):
                    raise ValueError('trainable first anchor has no active instruction: '+str(row))
            out.append({**row, 'current_text_schema': 'component_observation_v1',
                        'current_text_timeline': timeline})
        indices[name] = out
    catalog = json.loads(args.base_catalog.read_text())
    old = sorted(catalog['texts'], key=lambda r: r['text_id'])
    assert [r['text_id'] for r in old] == list(range(len(old)))
    missing = sorted(texts - {canonical(r['text']) for r in old})
    catalog['texts'] = old + [{'text_id':len(old)+i, 'text':t,
                               'sha256':hashlib.sha256(t.encode()).hexdigest()} for i,t in enumerate(missing)]
    catalog['encoder'] = {**catalog['encoder'], 'status':'append-pending',
                          'cache_file':'text_cache.safetensors'}
    catalog['component_extension'] = {'base_catalog':str(args.base_catalog.absolute()),
        'base_catalog_sha256':sha(args.base_catalog), 'base_cache':str(args.base_cache.absolute()),
        'base_count':len(old), 'added_count':len(missing), 'alignment':'observation_frame'}
    args.output.mkdir(parents=True)
    for name, rows in indices.items():
        with (args.output/(name+'.jsonl')).open('x') as f:
            for row in rows: f.write(json.dumps(row, sort_keys=True)+'\n')
    (args.output/'text_catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
    report = {'source_index_dir':str(args.source.resolve()),
        'source_hashes':{name:sha(args.source/(name+'.jsonl')) for name in indices},
        'clips':len(indices['train']), 'replay_clips':len(indices['train_replay']),
        'episodes':len({r['episode_id'] for r in indices['train']}),
        'building_episodes':len(timelines), 'component_texts':len(texts-{''}),
        'new_texts':len(missing), 'timelines':evidence,
        'sampling':'identical sample order and weights; no label changes',
        'shared_text':'unchanged episode task', 'current_text':'per-observation active component, empty after completion'}
    (args.output/'component_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'timelines'}),flush=True)


def encode(args):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer, T5EncoderModel
    catalog_path = args.output/'text_catalog.json'
    catalog = json.loads(catalog_path.read_text()); extension = catalog['component_extension']
    target = args.output/'text_cache.safetensors'
    if target.exists(): raise FileExistsError(target)
    # taiji and GPU nodes expose the shared filesystem through different mounts.
    base_cache = args.base_cache or Path(extension['base_cache'])
    old = load_file(str(base_cache), device='cpu')
    count = extension['base_count']
    assert old['encoder_hidden'].shape[0] == count
    texts = [r['text'] for r in catalog['texts']]
    max_length = int(catalog['encoder']['max_length'])
    assert old['encoder_hidden'].shape[1] == max_length
    source = args.model or catalog['encoder']['name']
    tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=args.model_cache)
    model = T5EncoderModel.from_pretrained(source, cache_dir=args.model_cache).eval().to(args.device)
    def run(items):
        tokens = tokenizer(items, padding='max_length', truncation=True,
                           max_length=max_length,return_tensors='pt')
        mask = tokens['attention_mask'].byte()
        with torch.no_grad():
            hidden = model(**{k:v.to(args.device) for k,v in tokens.items()}).last_hidden_state.cpu()
        return hidden.to(old['encoder_hidden'].dtype), mask.to(old['attention_mask'].dtype)
    # Confirm the new encoder matches the old cache before adding any vectors.
    probes = [i for i in range(count) if texts[i]][:8]
    hidden, mask = run([texts[i] for i in probes])
    assert torch.equal(mask,old['attention_mask'][probes])
    valid = mask.bool()
    reference = old['encoder_hidden'][probes][valid].float(); actual = hidden[valid].float()
    cosine = float(F.cosine_similarity(actual,reference,dim=-1).mean())
    mae = float((actual-reference).abs().mean())
    if cosine < .999 or mae > .01:
        raise ValueError(f'T5 encoder/cache mismatch: cosine={cosine}, mae={mae}')
    hs, ms = [old['encoder_hidden']], [old['attention_mask']]
    for start in range(count,len(texts),args.batch_size):
        h,m = run(texts[start:start+args.batch_size]); hs.append(h); ms.append(m)
        print(json.dumps({'encoded':min(start+args.batch_size,len(texts))-count,'added':len(texts)-count}),flush=True)
    save_file({'encoder_hidden':torch.cat(hs).contiguous(), 'attention_mask':torch.cat(ms).contiguous()},str(target))
    check = load_file(str(target),device='cpu')
    for key in old: assert torch.equal(check[key][:count],old[key]), key
    catalog['encoder']['status'] = 'complete'
    catalog['component_extension'].update(base_cache=str(base_cache.absolute()), base_cache_sha256=sha(base_cache),
        cache_sha256=sha(target), encoder_source=source, probe_cosine=cosine, probe_mae=mae,
        original_vectors_bitwise_preserved=True)
    catalog_path.write_text(json.dumps(catalog,indent=2)+'\n')
    print(json.dumps(catalog['component_extension']),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('build','encode'))
    p.add_argument('--source',type=Path); p.add_argument('--dataset-root',type=Path)
    p.add_argument('--base-catalog',type=Path); p.add_argument('--base-cache',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model'); p.add_argument('--model-cache'); p.add_argument('--device',default='cpu')
    p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args()
    if a.mode=='build': build(a)
    else: encode(a)


if __name__=='__main__': main()
