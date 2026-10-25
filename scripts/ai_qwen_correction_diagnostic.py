"""Frozen public debugging comparison; no expert labels or trading acceptance."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

source=Path(os.environ.get('WILLIAMS_DIAGNOSTIC_SOURCE_ROOT',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(source))


def main():
    from importlib.metadata import version
    from llama_cpp import Llama
    from williams_ai.unified_ai import WilliamsAI
    from williams_ai.real_gguf_acceptance import actual_gguf_sha256
    from williams_ai.causal_paper_lab import frozen_markets
    model=Path(os.environ['GGUF_MODEL_PATH'])
    actual_gguf_sha256(model,'dbc27452971c440c0b94483fe6c101cf950ea50af6bcafe1282353c3886e6366')
    calls=[];loads=[]
    fields=('setup','side','entry_state','rule_ids','learning_need','learning_question','reason')
    class Tracer:
        def __init__(self,**kwargs):
            loads.append(1);self.native=Llama(**kwargs)
        def __getattr__(self,name):return getattr(self.native,name)
        def create_chat_completion(self,**kwargs):
            response=self.native.create_chat_completion(**kwargs)
            payload=json.loads(kwargs['messages'][1]['content'])
            choices=response.get('choices',[])
            event={'feedback_in_actual_prompt':bool(payload.get('previous_decision_review')),
                   'usage':response.get('usage',{}),'decision':None}
            if choices:
                event['finish_reason']=choices[0].get('finish_reason')
                try:
                    decision=json.loads(choices[0]['message']['content'])
                    event['decision']={k:decision[k] for k in fields if k in decision}
                except (TypeError,KeyError,ValueError):event['parse_error']=True
            calls.append(event)
            return response
    brain=WilliamsAI(local_model=model,memory=Path('/tmp/qwen_correction_memory.sqlite3'),
        public_rules_only=True,loader=Tracer)
    brain.memory.retain('A closed-candle observation is not an exchange fill. '
        'Фрактал и стоп требуют отдельных подтверждений.',source='public-technical-fixture',
        title='Technical memory fixture, not an expert trading label',edition='TC2',
        available_at='2026-01-01T00:00:00Z')
    market=frozen_markets()['ETHUSDT']
    results=[]
    for index in (299,300):
        start=len(calls)
        case_hash=hashlib.sha256(json.dumps(market['bars'][:index+1],sort_keys=True,
            separators=(',',':'),allow_nan=False).encode()).hexdigest()
        print('START_FROZEN_CASE',index,flush=True)
        try:
            result=brain.analyze('ETHUSDT',market['bars'],tick_size=market['tick_size'],
                bar_index=index,include_topdown=False)
            error=None
        except (ValueError,RuntimeError,TypeError,KeyError):
            result=None;error='ACTUAL_MODEL_OR_PIPELINE_FAILED'
        results.append({'index':index,'closed_history_sha256':case_hash,
            'calls':calls[start:],'result':result,'error':error})
        print('FINISH_FROZEN_CASE',index,flush=True)
    assert len(loads)==1
    successful=[x for x in results if x['result'] is not None
                and x['result']['research_reasoning_consistent']]
    report={'source_commit':os.environ['SOURCE_HEAD'],'arm':os.environ['DIAGNOSTIC_ARM'],
        'critic_sha256':hashlib.sha256((source/'williams_ai/algorithm_guard.py').read_bytes()).hexdigest(),
        'model_sha256':'dbc27452971c440c0b94483fe6c101cf950ea50af6bcafe1282353c3886e6366',
        'llama_cpp_python_version':version('llama-cpp-python'),
        'mode':'SEEN_TWO_CUTOFF_PUBLIC_QWEN_DEBUGGING_COMPARISON',
        'independent_expert_test':False,'independent_holdout':False,
        'original_private_books_used':False,'source_mode':'PUBLIC_PARAPHRASES_ONLY',
        'model_weights_updated':False,'actual_model_loads':len(loads),
        'consistent_resolved_cases':len(successful),
        'consistent_non_abstention_cases':sum(x['calls'][-1]['decision']['setup'] not in
            ('NO_SETUP','UNKNOWN') for x in successful),
        'results':results,'trading_quality_proven':False,
        'execution_allowed':False,'trade_action':None}
    output=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)
    Path('/tmp/qwen_correction_diagnostic.json').write_text(output,encoding='utf-8')
    print(output)


if __name__=='__main__':main()
