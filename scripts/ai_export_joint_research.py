"""Portable research artifacts from a byte-verified GPU archive.

Export does not grant the runtime admission that failed models lack. Only
base-reference fields change; every safetensors byte and raw answer is kept.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.ai_export_portable_restorative import portable_config,TOKENIZER_FILES
from williams_ai.deep_book_research import MODEL,REVISION


def report(proof,receipt):
    lines=["# Williams AI — результат GPU-обучения", "",
        "Qwen3-4B; NF4 QLoRA rank16, два Tesla T4, 33 030 144 обучаемых параметра.", "",
        f"Фактически выполнено обновлений: **{proof['actual_updates']}**; выбран {proof['selected_checkpoint']} "
        f"(реальных обновлений в выбранной точке: {proof['selected_actual_updates']}).",
        f"Попыток оптимизатора: {receipt['optimizer_attempted_steps']}; AMP-пропусков: {receipt['amp_skipped_steps']}.", "",
        "Новые веса сохранены и проверены. Доля эпохи: "+str(receipt['actual_epoch'])+".", "",
        "## Одинаковый финальный протокол", "",
        "| Модель | Новые числовые условия | Новые пограничные условия | Историческая регрессия |",
        "|---|---:|---:|---:|"]
    stages=list(proof['grades']['numeric'])
    for stage in stages:
        cells=[]
        for group in ('numeric','boundary','historical'):
            grade=proof['grades'].get(group,{}).get(stage)
            cells.append(f"{grade['correct']}/{grade['total']}" if grade else "—")
        lines.append('| '+stage+' | '+' | '.join(cells)+' |')
    lines += ["", "Исторические вопросы раскрывались ранее: это регрессионная проверка. "
        "Новые числовые/пограничные условия являются авторскими вычислительными задачами, "
        "а не экспертным экзаменом графиков, волн или доходности.", "",
        "## Правила нового кандидата", "", "| Группа | Правило | Верно |", "|---|---|---:|"]
    for group,scores in proof['grades'].items():
        if group not in {'numeric','boundary','historical'}:continue
        for rule,value in scores['candidate']['per_rule'].items():
            lines.append(f"| {group} | {rule} | {value['correct']}/{value['total']} |")
    lines += ["", "## Сохранение материала книг", "",
        "| Проверка | Допуск |", "|---|---|"]
    for name,gate in proof['text_gates'].items():
        lines.append('| '+name+' | '+('Пройден' if gate['accepted'] else 'Отклонён: '+', '.join(gate['reasons']))+' |')
    lines += ["", "Это прежние development-страницы и упражнения четырёх книг; "
        "новым закрытым экзаменом книг они не являются.", "",
        "Совокупный исследовательский допуск: **"+('пройден' if proof['pilot_research_gates_passed'] else 'не пройден')+"**.",
        "Квалификация автономного трейдера не установлена; веса в торговле не активированы.", "",
        "## Происхождение", "", f"GPU-код: `{proof['source_commit']}`.",
        f"SHA256 нового адаптера: `{proof['candidate_adapter_sha256']}`.",
        f"Исходный архив: `{proof['archive_sha256']}`.", "",
        "Williams Bot 2.1 не изменён. Базовые веса Qwen, приватные тексты книг, "
        "транспортные конверты и учётные ключи в этот комплект не включены.", ""]
    return '\n'.join(lines).encode()


def export(archive,audit_file,destination,report_file):
    if destination.exists() or report_file.exists():raise ValueError('NEW_RESEARCH_EXPORT_PATHS_REQUIRED')
    proof=json.loads(audit_file.read_bytes())
    if (proof['archive_sha256']!=sha256(archive.read_bytes()).hexdigest()
        or proof['execution_allowed'] is not False or proof['trading_qualification'] is not False
        or proof['actual_updates']<=0 or proof['changed_values']<=0):
        raise ValueError('INDEPENDENT_REAL_TRAINING_AUDIT_REQUIRED')
    entries={};models={}
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):
            raise ValueError('SAFE_UNIQUE_RESEARCH_ARCHIVE_REQUIRED')
        receipt=json.loads(z.read('research_receipt.json'))
        if (receipt['source_commit']!=proof['source_commit'] or receipt['model_repo']!=MODEL
            or receipt['model_revision']!=REVISION or receipt['complete_training'] is not True
            or receipt['complete_test'] is not True or receipt['candidate_adapter_sha256']!=proof['candidate_adapter_sha256']):
            raise ValueError('EXPORTED_MODEL_MUST_MATCH_TRAINING_AND_EXAM_RECEIPT')
        for stage,expected in (('source',receipt['initial_adapter_sha256']),('candidate',proof['candidate_adapter_sha256'])):
            raw_config=z.read(stage+'_adapter/adapter_config.json');config,delta=portable_config(raw_config)
            weights=z.read(stage+'_adapter/adapter_model.safetensors')
            if sha256(weights).hexdigest()!=expected:raise ValueError('UNCHANGED_AUDITED_SAFETENSORS_REQUIRED')
            prefix='adapters/'+stage+'/'
            entries[prefix+'adapter_config.json']=config;entries[prefix+'adapter_model.safetensors']=weights
            for name in TOKENIZER_FILES:
                path='candidate_adapter/'+name
                if path in names:entries[prefix+name]=z.read(path)
            if prefix+'tokenizer.json' not in entries or prefix+'chat_template.jinja' not in entries:
                raise ValueError('EXACT_PINNED_MODEL_TOKENIZER_REQUIRED')
            entries['evidence/original_'+stage+'_gpu_adapter_config.json']=raw_config
            models[stage]={'tensor_sha256':expected,'configuration_delta_only':delta,'execution_allowed':False}
        for name in names:
            if '/' not in name and name.endswith('.json'):entries['evidence/'+name]=z.read(name)
        entries['evidence/independent_training_audit.json']=audit_file.read_bytes()
        content=report(proof,receipt);entries['GPU_RESULT_RU.md']=content
    entries['README_RU.md']=(
        '# Williams AI — исследовательские LoRA-адаптеры\n\n'
        'Комплект содержит исходную точку и новый кандидат; это не APK и не базовая модель.\n'
        'Исследовательский экспорт не активирует адаптер в Williams AI и не отменяет его допусков.\n\n'
        f'База: {MODEL}, строго ревизия {REVISION}. Для локального исследования загрузите\n'
        'базу через Transformers и подключите PeftModel.from_pretrained(base, "adapters/candidate").\n'
        'Токенизатор и chat_template берутся из того же каталога. База должна быть той же\n'
        'ревизии; используйте безопасный safetensors, без произвольного remote code.\n\n'
        'Рабочие версии GPU-стека: transformers 4.57.1, peft 0.17.1, accelerate 1.10.1,\n'
        'bitsandbytes 0.48.1, safetensors 0.6.2 и CUDA-совместимый PyTorch.\n'
        'Перед загрузкой сверьте каждый файл с EXPORT_MANIFEST.json.\n'
        'Значения native-экзаменов, ошибки и ограничения находятся в GPU_RESULT_RU.md;\n'
        'сырые ответы и независимый CPU-аудит — в evidence.\n').encode()
    manifest={'format':'PORTABLE_JOINT_RESEARCH_LORA_20261106','base_model':MODEL,'base_revision':REVISION,
        'source_commit':proof['source_commit'],'all_safetensors_bytes_preserved':True,'models':models,
        'base_weights_or_private_book_text_included':False,'trading_qualification':False,'execution_allowed':False,
        'files_sha256':{n:sha256(b).hexdigest() for n,b in sorted(entries.items())}}
    entries['EXPORT_MANIFEST.json']=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for name,raw in sorted(entries.items()):z.writestr(name,raw)
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None or set(z.namelist())!=set(entries):raise ValueError('RESEARCH_EXPORT_READBACK_FAILED')
        for name,digest in manifest['files_sha256'].items():
            if sha256(z.read(name)).hexdigest()!=digest:raise ValueError('RESEARCH_EXPORT_HASH_MISMATCH')
    report_file.write_bytes(content)
    return {'archive_sha256':sha256(destination.read_bytes()).hexdigest(),'bytes':destination.stat().st_size,
            'models':models,'files':len(entries),'all_exported_files_readback_verified':True,'execution_allowed':False}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('archive','audit','destination','report'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(export(a.archive,a.audit,a.destination,a.report)))
