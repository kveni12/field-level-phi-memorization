#!/usr/bin/env python3
"""Score the paper's first-data-token J-Lens on matched training records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--jacobians', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--model-name', default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--canaries', type=Path, default=ROOT / 'data/training/llama31_8b_balanced_2k_phi_benign_v2_luna_9f/train_canaries.jsonl')
    parser.add_argument('--samples-per-label', type=int, default=800)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ssn-last-four', action='store_true', help='For SSN/account, average RR across all final-four-digit tokens; other fields retain first-token scoring')
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        parser.error('shard-index must be between zero and shard-count minus one')

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from memo_diff.analysis.cohort import _load_canaries
    from memo_diff.analysis.field_targets import first_token_targets, reciprocal_rank, ssn_last_four_token_indices
    from memo_diff.analysis.jacobian_lens import validate_lens_manifest
    from memo_diff.analysis.model_utils import _final_norm_and_head
    from memo_diff.analysis.readout import load_jacobians_pt, _lens_logits

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, attn_implementation='eager',
        torch_dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
    ).to(device).eval()
    norm, head = _final_norm_and_head(model)
    jacobians, target, manifest, _ = load_jacobians_pt(args.jacobians)
    validate_lens_manifest(manifest, model=model, tokenizer=tokenizer,
                          checkpoint_path=args.checkpoint, jacobians=jacobians, target_layer=target)
    jacobians = {layer: matrix.to(device) for layer, matrix in jacobians.items()}
    selected = _load_canaries(args.canaries, per_label=args.samples_per_label,
                             seed=args.seed, pair_twins=True, train_member=True)
    rows = [row for label in ('protected_association', 'benign_arbitrary_association')
            for row in selected[label]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f'scores_shard_{args.shard_index}.jsonl'
    if output.exists():
        raise FileExistsError(f'{output} already exists; select a fresh output directory')
    with output.open('w') as handle, torch.no_grad():
        for number, row in enumerate(rows[args.shard_index::args.shard_count]):
            ids, targets = first_token_targets(
                full_text=row['text'], dossier=row['dossier'], label=row['label'],
                tokenizer=tokenizer, add_special_tokens=True, strip_value_prefixes=True,
            )
            targets = {field: [index] for field, index in targets.items()}
            if args.ssn_last_four:
                targets['ssn'] = ssn_last_four_token_indices(full_text=row['text'],
                    value=row['dossier']['ssn'], tokenizer=tokenizer)
            positions = [index - 1 for indices in targets.values() for index in indices]
            out = model(input_ids=torch.tensor([ids], device=device),
                        output_hidden_states=True, use_cache=False)
            states = [hidden[0] for hidden in out.hidden_states]
            for layer, logits in _lens_logits(states, jacobians, target_layer=target,
                                              norm=norm, lm_head=head, positions=positions):
                for field, indices in targets.items():
                    token_scores = [reciprocal_rank(logits[index - 1], ids[index]) for index in indices]
                    rank = token_scores[0][0] if len(indices) == 1 else None
                    rr = sum(score for _, score in token_scores) / len(token_scores)
                    handle.write(json.dumps({
                        'analysis': 'preanswer_jlens', 'evaluation_split': 'train_member',
                        'example_id': row['id'], 'pair_id': row['pair_id'], 'label': row['label'],
                        'field': field, 'layer': layer, 'rank': rank, 'reciprocal_rank': rr,
                        'gold_token_ids': [ids[index] for index in indices], 'target_token_indices': indices,
                        'target_kind': 'ssn_last_four_token_mean' if args.ssn_last_four and field == 'ssn' else 'first_data_token',
                        'step': args.step,
                    }) + '\n')
            print(f'shard={args.shard_index} documents={number + 1}', flush=True)
    metadata = {
        'step': args.step, 'checkpoint': str(args.checkpoint),
        'jacobians': str(args.jacobians), 'samples_per_label': args.samples_per_label,
        'shard_index': args.shard_index, 'shard_count': args.shard_count,
        'data_sha256': hashlib.sha256(args.canaries.read_bytes()).hexdigest(),
        'score_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'target': ('SSN final-four token mean RR; other fields first data token.' if args.ssn_last_four
                   else 'First data token; fixed MRN-/HPB- literals skipped; SSN 000/999 prefix retained.'),
    }
    output.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
