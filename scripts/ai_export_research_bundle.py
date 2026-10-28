"""Bundle both verified research adapters, with tensor bytes unchanged."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.ai_export_portable_restorative import portable_config,TOKENIZER_FILES,EXAM_FILES
from williams_ai.deep_book_research import MODEL,REVISION
from williams_ai.source_consolidation import START_SHA


def export(parent:Path,source:Path,source_audit:Path,prior_exam:Path,final_exam:Path,readme:Path,destination:Path):
    if destination.exists():raise ValueError('NEW_RESEARCH_BUNDLE_REQUIRED')
    proof=json.loads(source_audit.read_bytes())
    if (proof['archive_sha256']!=sha256(source.read_bytes()).hexdigest() or proof['initial_adapter_sha256']!=START_SHA
        or proof['all_finite'] is not True or proof['matrices']!=504 or proof['parameters']!=33030144
        or proof['selected_on_development_only'] is not True or proof['execution_allowed'] is not False):
        raise ValueError('VERIFIED_SOURCE_TRAINING_AUDIT_REQUIRED')
    if sha256(parent.read_bytes()).hexdigest()!='6e18ff47570223a7f099c30401ca2e23b073e5670f0fef899944f74c78926841':
        raise ValueError('PREVIOUSLY_INDEPENDENTLY_AUDITED_PARENT_ARCHIVE_REQUIRED')
    entries={};models={}
    with zipfile.ZipFile(parent) as p,zipfile.ZipFile(source) as s:
        if p.read('candidate_adapter/adapter_model.safetensors')!=s.read('source_adapter/adapter_model.safetensors'):
            raise ValueError('EXACT_TRAINED_ADAPTER_LINEAGE_REQUIRED')
        for label,z,expected,root_jsons in (
            ('restorative_checkpoint768',p,START_SHA,('research_receipt.json','restorative_test_precommit.json')),
            ('source_consolidation_checkpoint1024',s,proof['candidate_adapter_sha256'],('research_receipt.json','source_validation_window_losses.json','practice_validation_window_losses.json','final_direct_development_predictions.json'))):
            names=z.namelist()
            if len(names)!=len(set(names)) or any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):raise ValueError('SAFE_TRAINING_ARCHIVE_ENTRIES_REQUIRED')
            weights=z.read('candidate_adapter/adapter_model.safetensors')
            if sha256(weights).hexdigest()!=expected:raise ValueError('EXPORTED_TENSOR_BYTES_MUST_MATCH_AUDITED_SHA')
            raw_config=z.read('candidate_adapter/adapter_config.json');config,delta=portable_config(raw_config)
            folder='adapters/'+label+'/'
            entries[folder+'adapter_model.safetensors']=weights;entries[folder+'adapter_config.json']=config
            for name in TOKENIZER_FILES:
                if 'candidate_adapter/'+name in names:entries[folder+name]=z.read('candidate_adapter/'+name)
            if folder+'tokenizer.json' not in entries or folder+'chat_template.jinja' not in entries:raise ValueError('EXACT_SAVED_TOKENIZER_REQUIRED')
            entries['evidence/'+label+'/original_gpu_adapter_config.json']=raw_config
            for name in root_jsons:entries['evidence/'+label+'/'+name]=z.read(name)
            receipt=json.loads(z.read('research_receipt.json'))
            if receipt['candidate_adapter_sha256']!=expected or receipt['model_revision']!=REVISION:raise ValueError('MODEL_RECEIPT_MUST_MATCH_EXPORTED_BYTES')
            models[label]={'tensor_sha256':expected,'training_source_commit':receipt['source_commit'],
                'actual_training_updates':receipt['joint_optimizer_steps'],'selected_checkpoint':receipt['selected_checkpoint'],
                'selected_updates':receipt['actual_updates_at_selected_checkpoint'],'configuration_delta_only':delta,
                'trading_qualification':False,'execution_allowed':False}
    entries['evidence/source_consolidation_independent_training_audit.json']=source_audit.read_bytes()
    entries['evidence/restorative_independent_training_summary.json']=Path('experiments/restorative-training-result-20261028.json').read_bytes()
    for directory,label in ((prior_exam,'prior_four_model_exam'),(final_exam,'final_five_model_exam')):
        for name in EXAM_FILES:
            if not (directory/name).is_file():raise ValueError('COMPLETE_FINAL_EXAM_EVIDENCE_REQUIRED:'+label+'/'+name)
            entries['evidence/'+label+'/'+name]=(directory/name).read_bytes()
        preds=json.loads((directory/'spread_market_predictions.json').read_bytes());grade=json.loads((directory/'spread_market_grade.json').read_bytes())
        audit=json.loads((directory/'independent_final_exam_audit.json').read_bytes())
        expected=START_SHA if label=='prior_four_model_exam' else proof['candidate_adapter_sha256']
        if (preds['receipt']['adapter_sha256']['candidate']!=expected or audit['candidate_adapter_sha256']!=expected
            or audit['all_cpu_recomputations_agree'] is not True
            or grade['predictions_file_sha256']!=sha256((directory/'spread_market_predictions.json').read_bytes()).hexdigest()):
            raise ValueError('EXAM_EVIDENCE_MUST_BELONG_TO_EXPORTED_WEIGHTS')
    entries['README_RU.md']=readme.read_bytes()
    manifest={'format':'TWO_VERIFIED_PORTABLE_RESEARCH_LORA_ADAPTERS_V1','base_model':MODEL,'base_revision':REVISION,
        'base_weights_included':False,'private_source_text_or_transport_envelopes_included':False,
        'all_safetensors_bytes_preserved':True,'models':models,'execution_allowed':False,'trading_qualification':False,
        'files_sha256':{n:sha256(b).hexdigest() for n,b in sorted(entries.items())}}
    entries['EXPORT_MANIFEST.json']=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for name,raw in sorted(entries.items()):z.writestr(name,raw)
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None or set(z.namelist())!=set(entries):raise ValueError('RESEARCH_BUNDLE_READBACK_FAILED')
        for name,digest in manifest['files_sha256'].items():
            if sha256(z.read(name)).hexdigest()!=digest:raise ValueError('READBACK_FILE_HASH_MISMATCH')
    return {**manifest,'bundle_sha256':sha256(destination.read_bytes()).hexdigest(),'bundle_bytes':destination.stat().st_size,'all_exported_files_readback_verified':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for n in ('parent','source','source-audit','prior-exam','final-exam','readme','destination','proof-output'):parser.add_argument('--'+n,type=Path,required=True)
    a=parser.parse_args()
    if a.proof_output.exists():raise ValueError('NEW_EXPORT_PROOF_REQUIRED')
    result=export(a.parent,a.source,a.source_audit,a.prior_exam,a.final_exam,a.readme,a.destination)
    a.proof_output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},ensure_ascii=False))
