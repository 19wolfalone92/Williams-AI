"""Export audited LoRA bytes with a portable, pinned base-model reference.

No original book text, private transport envelope or credentials are exported.
The original GPU configuration is retained as evidence; tensor bytes never change.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from williams_ai.deep_book_research import MODEL,REVISION
from williams_ai.restorative_book_practice import INITIAL_SHA

TOKENIZER_FILES={'tokenizer.json','tokenizer_config.json','special_tokens_map.json',
    'added_tokens.json','vocab.json','merges.txt','chat_template.jinja','generation_config.json'}
EXAM_FILES={'spread_market_predictions.json','spread_market_grade.json',
    'development_mode_selection.json','withheld_market_outcomes_cpu_only.json','spread_market_seal.json',
    'independent_final_exam_audit.json'}


def portable_config(raw:bytes):
    original=json.loads(raw)
    if (original.get('peft_type')!='LORA' or original.get('r')!=16 or original.get('lora_alpha')!=32
        or original.get('task_type')!='CAUSAL_LM' or original.get('bias')!='none'):
        raise ValueError('VERIFIED_RANK16_CAUSAL_LORA_CONFIG_REQUIRED')
    changed={**original,'base_model_name_or_path':MODEL,'revision':REVISION}
    delta={k:{'original':original.get(k),'portable':changed[k]} for k in changed if original.get(k)!=changed[k]}
    if set(delta)-{'base_model_name_or_path','revision'}:raise ValueError('ONLY_BASE_REFERENCE_MAY_CHANGE')
    return (json.dumps(changed,ensure_ascii=False,indent=2)+'\n').encode(),delta


def export(archive:Path,audit_file:Path,exam_directory:Path,readme:Path,destination:Path):
    if destination.exists():raise ValueError('NEW_PORTABLE_ARCHIVE_REQUIRED')
    audit=json.loads(audit_file.read_bytes());archive_sha=sha256(archive.read_bytes()).hexdigest()
    if (audit['archive_sha256']!=archive_sha or audit['weights']['source']['sha256']!=INITIAL_SHA
        or audit['execution_allowed'] is not False or audit['trading_qualification'] is not False
        or audit['development_selection_independently_recomputed'] is not True):
        raise ValueError('INDEPENDENT_RAW_GPU_ARCHIVE_AUDIT_REQUIRED')
    missing=[n for n in EXAM_FILES if not (exam_directory/n).is_file()]
    if missing:raise ValueError('COMPLETE_FROZEN_EXAM_EVIDENCE_REQUIRED:'+','.join(sorted(missing)))
    entries={}
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):
            raise ValueError('SAFE_UNIQUE_GPU_ARCHIVE_ENTRIES_REQUIRED')
        raw_config=z.read('candidate_adapter/adapter_config.json')
        config,delta=portable_config(raw_config)
        weights=z.read('candidate_adapter/adapter_model.safetensors')
        weights_sha=sha256(weights).hexdigest()
        if weights_sha!=audit['weights']['candidate']['sha256']:raise ValueError('AUDITED_TENSOR_BYTES_CHANGED')
        receipt=json.loads(z.read('research_receipt.json'))
        if receipt['candidate_adapter_sha256']!=weights_sha or receipt['source_commit']!=audit['source_commit']:
            raise ValueError('GPU_RECEIPT_AND_CPU_AUDIT_MUST_AGREE')
        entries['adapter/adapter_config.json']=config;entries['adapter/adapter_model.safetensors']=weights
        for name in TOKENIZER_FILES:
            key='candidate_adapter/'+name
            if key in names:entries['adapter/'+name]=z.read(key)
        if 'adapter/tokenizer.json' not in entries or 'adapter/chat_template.jinja' not in entries:
            raise ValueError('EXACT_SAVED_TOKENIZER_AND_CHAT_TEMPLATE_REQUIRED')
        for name in ('research_receipt.json','restorative_test_precommit.json'):
            entries['evidence/'+name]=z.read(name)
        entries['evidence/original_gpu_adapter_config.json']=raw_config
    prediction_bytes=(exam_directory/'spread_market_predictions.json').read_bytes()
    predictions=json.loads(prediction_bytes);grade=json.loads((exam_directory/'spread_market_grade.json').read_bytes())
    final_audit=json.loads((exam_directory/'independent_final_exam_audit.json').read_bytes())
    if (predictions['receipt']['adapter_sha256']['candidate']!=weights_sha
        or grade['predictions_file_sha256']!=sha256(prediction_bytes).hexdigest()
        or final_audit['candidate_adapter_sha256']!=weights_sha or final_audit['all_cpu_recomputations_agree'] is not True):
        raise ValueError('FINAL_EXAM_MUST_BELONG_TO_THE_EXPORTED_TENSORS')
    entries['evidence/independent_training_audit.json']=audit_file.read_bytes()
    for name in EXAM_FILES:entries['evidence/'+name]=(exam_directory/name).read_bytes()
    entries['README_RU.md']=readme.read_bytes()
    manifest={'format':'PORTABLE_RESEARCH_LORA_EXPORT_V1','base_model':MODEL,'base_revision':REVISION,
        'training_source_commit':audit['source_commit'],'raw_gpu_archive_sha256':archive_sha,
        'adapter_tensor_sha256':weights_sha,'all_safetensors_bytes_preserved':True,
        'configuration_delta_only':delta,'book_text_or_transport_envelopes_included':False,
        'base_weights_included':False,'execution_allowed':False,'trading_qualification':False,
        'files_sha256':{n:sha256(b).hexdigest() for n,b in sorted(entries.items())}}
    entries['EXPORT_MANIFEST.json']=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for name,raw in sorted(entries.items()):z.writestr(name,raw)
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None or set(z.namelist())!=set(entries):raise ValueError('EXPORT_ZIP_READBACK_FAILED')
        for name,expected in manifest['files_sha256'].items():
            if sha256(z.read(name)).hexdigest()!=expected:raise ValueError('EXPORT_FILE_HASH_FAILED')
        if sha256(z.read('adapter/adapter_model.safetensors')).hexdigest()!=weights_sha:
            raise ValueError('EXPORTED_ADAPTER_TENSOR_HASH_CHANGED')
    return {**manifest,'portable_archive_sha256':sha256(destination.read_bytes()).hexdigest(),
        'portable_archive_bytes':destination.stat().st_size,'export_readback_verified':True}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('archive','audit','exam-directory','readme','destination'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=export(a.archive,a.audit,a.exam_directory,a.readme,a.destination)
    print(json.dumps(result,ensure_ascii=False))
