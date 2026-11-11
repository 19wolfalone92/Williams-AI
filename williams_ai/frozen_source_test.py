"""Final-only LM evaluation on the 96 original pages withheld from training.

The training-payload schema rejects this separate test-only payload. Sending
these texts to inference does not authorize optimizer updates or reuse as an
unseen test after the first result is revealed.
"""
from __future__ import annotations
import base64
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import math

from .deep_book_research import EDITIONS
from .book_rules import registry
from .private_book_source import FIELDS
from .lora_integrity import validation_admission

FORMAT='PRIVATE_FROZEN_96_SOURCE_TEST_ONLY_NO_TRAINING_V1'


def manifest_digest(manifest):
    return sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()


def checked_test_payload(payload,*,expected_manifest_sha256):
    fields={'format','manifest','test_file_base64','training_page_ids','validation_page_ids','training_allowed','candidate_selection_allowed','execution_allowed'}
    if (set(payload)!=fields or payload['format']!=FORMAT or any(payload[k] is not False for k in ('training_allowed','candidate_selection_allowed','execution_allowed'))):
        raise ValueError('FROZEN_SOURCE_TEST_ONLY_NO_UPDATES_REQUIRED')
    m=payload['manifest']
    if (manifest_digest(m)!=expected_manifest_sha256 or m.get('page_boundary_embargo')!=2 or m.get('status')!='SOURCE_ONLY_NOT_EXPERT_APPROVED'
        or m.get('all_figures_understood') is not False or m.get('market_training_or_profitability_verified') is not False):
        raise ValueError('EXACT_FROZEN_TRAINING_MANIFEST_REQUIRED')
    books={b['id']:b for b in registry()['books']}
    if set(m['book_registry'])!=set(EDITIONS) or any(m['book_registry'][e]!={'sha256':books[e]['sha256'],'pages':books[e]['pages']} for e in EDITIONS):
        raise ValueError('ORIGINAL_FOUR_BOOK_IDENTITIES_REQUIRED')
    raw=base64.b64decode(payload['test_file_base64'],validate=True)
    if len(raw)>8_000_000 or sha256(raw).hexdigest()!=m['splits']['test']['sha256']:raise ValueError('FROZEN_SOURCE_TEST_TEXT_HASH_FAILED')
    rows=[json.loads(line) for line in raw.splitlines() if line.strip()];ids={e:set() for e in EDITIONS}
    for row in rows:
        e=row.get('edition');p=row.get('pdf_page')
        if (set(row)!=FIELDS or e not in books or type(p) is not int or not 1<=p<=books[e]['pages'] or p in ids[e]
            or row['source_sha256']!=books[e]['sha256'] or not isinstance(row['text'],str) or not 30<=len(row['text'])<=250000
            or row['extraction_status']!=('UNVERIFIED_OCR' if e=='COURSE' else 'PDF_TEXT_NOT_EXPERT_REVIEWED')):
            raise ValueError('VALID_UNIQUE_ORIGINAL_TEST_PAGE_REQUIRED')
        ids[e].add(p)
    if len(rows)!=96 or m['splits']['test']['rows']!=96 or {e:len(v) for e,v in ids.items()}!=m['splits']['test']['books']:
        raise ValueError('EXACT_FROZEN_NINETY_SIX_TEST_PAGES_REQUIRED')
    for key,split in (('training_page_ids','train'),('validation_page_ids','validation')):
        pages=payload[key]
        if set(pages)!=set(EDITIONS):raise ValueError('FOUR_BOOK_PAGE_EXCLUSIONS_REQUIRED')
        for e,values in pages.items():
            if (not isinstance(values,list) or len(values)!=len(set(values)) or any(type(v) is not int or not 1<=v<=books[e]['pages'] for v in values)
                or len(values)!=m['splits'][split]['books'][e] or any(abs(t-v)<=2 for t in ids[e] for v in values)):
                raise ValueError('SOURCE_TEST_TWO_PAGE_EMBARGO_BROKEN')
    return rows,{'manifest_sha256':expected_manifest_sha256,'test_file_sha256':sha256(raw).hexdigest(),
        'test_pages':{e:len(v) for e,v in ids.items()},'total_pages':96,'test_text_was_excluded_from_training':True,
        'evaluation_only_after_candidate_frozen':True,'training_performed_here':False,'ocr_and_figures_not_expert_approved':True,'execution_allowed':False}


def build_test_payload(directory:Path):
    from .private_book_source import load_private_source_corpus
    load_private_source_corpus(directory)
    m=json.loads((directory/'manifest.json').read_bytes());ids={}
    for split in ('train','validation'):
        ids[split]={e:[] for e in EDITIONS}
        for line in (directory/(split+'.jsonl')).read_bytes().splitlines():
            if line.strip():
                row=json.loads(line);ids[split][row['edition']].append(row['pdf_page'])
    payload={'format':FORMAT,'manifest':m,'test_file_base64':base64.b64encode((directory/'test.jsonl').read_bytes()).decode(),
        'training_page_ids':ids['train'],'validation_page_ids':ids['validation'],'training_allowed':False,
        'candidate_selection_allowed':False,'execution_allowed':False}
    _,audit=checked_test_payload(payload,expected_manifest_sha256=manifest_digest(m))
    raw=json.dumps(payload,sort_keys=True,separators=(',',':')).encode()
    design={**audit,'payload_sha256':sha256(raw).hexdigest(),'format':FORMAT,'loss_windows_max_tokens':384,
        'minimum_relative_loss_improvement':0.01,'maximum_book_loss_regression':0.02,
        'reference_models_for_gate':['previous40','focused69'],'after_first_evaluation_this_source_test_is_revealed':True}
    return raw,design


def grade_source_test(scores,*,include_restorative_parent=False):
    expected={'base','previous40','focused69','candidate'}|({'restorative768'} if include_restorative_parent else set())
    if type(include_restorative_parent) is not bool or set(scores)!=expected:raise ValueError('EXACT_SOURCE_TEST_MODELS_REQUIRED')
    for values in scores.values():
        if set(values)!=set(EDITIONS)|{'ALL'} or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values.values()):
            raise ValueError('FINITE_FULL_FOUR_BOOK_TEST_LOSSES_REQUIRED')
    refs=('previous40','focused69','restorative768') if include_restorative_parent else ('previous40','focused69')
    gates={name:validation_admission(scores[name],scores['candidate']) for name in refs}
    return {'mode':'FROZEN_96_ORIGINAL_PAGE_FINAL_LM_TEST_NOT_EXPERT_UNDERSTANDING','scores':scores,'gates':gates,
        'source_test_gate_passed':all(g['accepted'] for g in gates.values()),'checkpoint_selected_on_this_test':False,
        'text_loss_is_not_market_profitability':True,'trading_qualification':False,'execution_allowed':False}


def recompute_source_scores(window_losses,window_counts,*,include_restorative_parent=False):
    if type(include_restorative_parent) is not bool:raise ValueError('EXPLICIT_SOURCE_PROTOCOL_REQUIRED')
    stages=('base','previous40','focused69','restorative768','candidate') if include_restorative_parent else ('base','previous40','focused69','candidate')
    if set(window_counts)!=set(EDITIONS) or any(type(v) is not int or v<=0 for v in window_counts.values()):
        raise ValueError('POSITIVE_FOUR_BOOK_TEST_WINDOW_COUNTS_REQUIRED')
    seen=set();groups={s:{e:[] for e in EDITIONS} for s in stages}
    for stage,edition,index,loss in window_losses:
        key=(stage,edition,index)
        if (stage not in stages or edition not in EDITIONS or type(index) is not int or not 0<=index<window_counts[edition]
            or key in seen or type(loss) not in (int,float) or not math.isfinite(loss) or loss<0):
            raise ValueError('EXACT_FINITE_UNIQUE_SOURCE_TEST_WINDOW_REQUIRED')
        seen.add(key);groups[stage][edition].append((index,loss))
    if len(seen)!=len(stages)*sum(window_counts.values()):raise ValueError('ALL_MODELS_ALL_SOURCE_TEST_WINDOWS_REQUIRED')
    result={}
    for stage,books in groups.items():
        values={e:sum(v for _,v in sorted(rows))/len(rows) for e,rows in books.items()}
        values['ALL']=sum(values[e]*window_counts[e] for e in EDITIONS)/sum(window_counts.values())
        result[stage]=values
    return result
