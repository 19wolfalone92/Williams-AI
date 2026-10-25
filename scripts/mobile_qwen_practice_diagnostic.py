"""Actual Qwen, shared mobile prompt, public source memory; never training."""
from pathlib import Path
import hashlib
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    from llama_cpp import Llama
    from importlib.metadata import version
    from williams_ai.mobile_tutor import practice_cases, public_source, prompt, parse_answer
    from williams_ai.real_gguf_acceptance import actual_gguf_sha256
    model=Path(os.environ["GGUF_MODEL_PATH"])
    expected="dbc27452971c440c0b94483fe6c101cf950ea50af6bcafe1282353c3886e6366"
    actual_gguf_sha256(model,expected)
    llm=Llama(model_path=str(model),n_ctx=8192,n_gpu_layers=0,verbose=False)
    cases=practice_cases()
    output=Path('/tmp/mobile_qwen_practice.json')
    report={"mode":"ACTUAL_QWEN_MOBILE_SOURCE_ASSISTED_PRACTICE",
        "source_commit":os.environ["SOURCE_HEAD"],"model_sha256":expected,
        "runtime":version('llama-cpp-python'),"actual_model_loads":1,"cases":len(cases),
        "original_private_books_used":False,"source_mode":"PUBLIC_PROJECT_PARAPHRASES",
        "project_practice_cases_not_independent_holdout":True,
        "same_questions_compared_with_and_without_source":True,
        "actual_phone_hardware_tested":False,"model_weights_updated":False,
        "lora_loaded":False,"execution_allowed":False,"trade_action":None,"results":[]}

    def save():
        temp=output.with_suffix('.part')
        temp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
        temp.replace(output)

    memory=Path('/tmp/mobile_qwen_source_memory.sqlite3')
    if memory.exists(): raise RuntimeError('FRESH_SOURCE_MEMORY_REQUIRED')
    # Persist originals of public project paraphrases, not model summaries.
    with sqlite3.connect(memory) as db:
        db.execute('CREATE TABLE source_memory(case_id TEXT PRIMARY KEY,model_sha TEXT,source_json TEXT,digest TEXT)')
        for c in cases:
            text=json.dumps(public_source(c),ensure_ascii=False,sort_keys=True)
            db.execute('INSERT INTO source_memory VALUES(?,?,?,?)',(c['id'],expected,text,hashlib.sha256(text.encode()).hexdigest()))
    save()
    for mode in ('without_source','with_reopened_source_memory'):
        for c in cases:
            source=None
            if mode!='without_source':
                with sqlite3.connect(memory) as db:
                    row=db.execute('SELECT source_json,digest FROM source_memory WHERE case_id=? AND model_sha=?',(c['id'],expected)).fetchone()
                assert hashlib.sha256(row[0].encode()).hexdigest()==row[1]
                source=json.loads(row[0])
            text=prompt(c,source)
            start=time.monotonic()
            response=llm.create_chat_completion(messages=[{'role':'user','content':text}],
                temperature=0.3,top_k=40,top_p=0.95,min_p=0.05,seed=1942,
                max_tokens=384,stream=False)
            choice=response['choices'][0]
            try:
                answer=parse_answer(choice['message']['content'],choice.get('finish_reason'))
                error=None
            except (ValueError,TypeError,KeyError):
                answer=None;error='MODEL_PROTOCOL_OR_LENGTH_FAILURE'
            report['results'].append({'mode':mode,'case':c['id'],'edition':c['edition'],
                'correct':answer is not None and answer['choice']==c['correct'],
                'answer':answer,'error':error,'finish_reason':choice.get('finish_reason'),
                'usage':response.get('usage'), 'elapsed_seconds':round(time.monotonic()-start,2),
                'prompt_sha256':hashlib.sha256(text.encode()).hexdigest()})
            save()
            print(mode,c['id'],report['results'][-1]['correct'],error,flush=True)
    scores={mode:sum(r['correct'] for r in report['results'] if r['mode']==mode)
            for mode in ('without_source','with_reopened_source_memory')}
    before={r['case']:r['correct'] for r in report['results'] if r['mode']=='without_source'}
    after={r['case']:r['correct'] for r in report['results'] if r['mode']!='without_source'}
    report.update(scores=scores,newly_correct=sum(after[k] and not before[k] for k in before),
        regressed=sum(before[k] and not after[k] for k in before),
        protocol_failures=sum(r['error'] is not None for r in report['results']),
        trading_skill_proven=False,book_mastery_percent=None)
    save();print(json.dumps({k:report[k] for k in ('scores','newly_correct','regressed','protocol_failures')},ensure_ascii=False))


if __name__=='__main__': main()
