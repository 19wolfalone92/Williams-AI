"""Actual optimizer/save/reload diagnostic, NOT Williams/Qwen knowledge training.

Tiny randomly initialized Qwen3, artificial token IDs, no books or human
approvals. Only a public metrics JSON is exported; toy weights are disposable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from peft import PeftModel
    from transformers import Qwen3Config, Qwen3ForCausalLM, set_seed
    from williams_ai.lora_training import LoraPlan, fit_lora
    torch.set_num_threads(2)
    set_seed(314)
    config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
        max_position_embeddings=128,bos_token_id=1,eos_token_id=2,pad_token_id=0)
    base=Qwen3ForCausalLM(config)
    frozen=[(p,p.detach().clone()) for p in base.parameters()]
    def row(ed):
        prefix=[1,20+ed,8,9]
        answer=[10,11,12,13]*4+[2]
        ids=prefix+answer
        return {'input_ids':ids,'attention_mask':[1]*len(ids),
                'labels':[-100]*len(prefix)+answer}
    train=[row(ed) for ed in range(4)]*2
    validation={ed:[row(i)] for i,ed in enumerate(('TC1','NTD','TC2','COURSE'))}
    with TemporaryDirectory(prefix='williams-synthetic-lora-') as directory:
        root=Path(directory)
        base.save_pretrained(root/'base',safe_serialization=True)
        model,report=fit_lora(base,train,validation,workdir=root/'fit',pad_token_id=0,
            max_steps=40,plan=LoraPlan(rank=8,learning_rate=5e-4,
                gradient_accumulation=1,target_profile='attention-mlp'),use_cpu=True)
        assert all(not p.requires_grad and torch.equal(p.detach(),before) for p,before in frozen)
        assert report['adapter_weights_changed'] and report['best_validation_checkpoint_reloaded']
        assert report['adapted_by_book']['ALL'] < report['baseline_by_book']['ALL']
        model.save_pretrained(root/'adapter',safe_serialization=True)
        restored=PeftModel.from_pretrained(Qwen3ForCausalLM.from_pretrained(root/'base',
            use_safetensors=True,local_files_only=True),root/'adapter',local_files_only=True)
        model.eval();restored.eval()
        ids=torch.tensor([row(0)['input_ids']])
        with torch.no_grad():
            difference=float((model(ids).logits-restored(ids).logits).abs().max())
        assert difference < 1e-6
        # Invalid baseline must stop before the optimizer can execute.
        from unittest.mock import patch
        broken=Qwen3ForCausalLM(config)
        with torch.no_grad():broken.model.embed_tokens.weight.fill_(float('nan'))
        rejected=False
        with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('BASELINE_MUST_STOP_BEFORE_STEP')):
            try:
                fit_lora(broken,train,validation,workdir=root/'bad-fit',pad_token_id=0,
                    max_steps=1,plan=LoraPlan(gradient_accumulation=1),use_cpu=True)
            except ValueError as exc:
                assert 'FINITE_FOUR_BOOK_VALIDATION' in str(exc)
                rejected=True
        assert rejected
        report.update(source_commit=os.environ.get('SOURCE_HEAD',os.environ.get('GITHUB_SHA')),
            mode='SYNTHETIC_TINY_QWEN3_LORA_SOFTWARE_DIAGNOSTIC',
            actual_optimizer_and_weight_updates=True,actual_safetensors_reload=True,
            maximum_reload_logit_difference=difference,downloaded_pretrained_weights=False,
            nonfinite_baseline_rejected_before_optimizer_step=rejected,
            private_books_used=False,independent_expert_supervision=False,
            independent_market_holdout=False,williams_knowledge_trained=False,
            gguf_adapter_application_tested=False,trading_quality_proven=False,
            execution_allowed=False,trade_action=None)
        Path('/tmp/williams_lora_cpu_diagnostic.json').write_text(
            json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
