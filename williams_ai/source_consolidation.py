"""Corrective source stage fixed before final source scores are read."""
from __future__ import annotations
from .restorative_book_practice import restorative_cases
from .contrastive_book_practice import question_rows,questions_digest
from .focused_book_practice import checkpoint_selection
from .lora_integrity import validation_admission

START_SHA='6fb8a52a31c49ed789e2efb193cba406851f7a7d05850f8e48a3a14ae8d076a9'
PARENT_RECEIPT_SHA='f0802e08ef946bd7a8afadf60f1a4eca4dee1c8eb9d964bd237fc4f25824b40c'
RUN_SLUG='williams-ai-source-preserve-2t4-20261028'


def selection(grade,source,practice,baselines):
    gates={name:validation_admission(baselines[before],after) for name,before,after in (
        ('source_previous40','source_baseline_loss',source),('source_focused69','initial_focused_source_loss',source),
        ('practice_previous40','practice_baseline_loss',practice),('practice_focused69','initial_focused_practice_loss',practice),
        ('source_parent768','parent_source_loss',source),('practice_parent768','parent_practice_loss',practice))}
    semantic=checkpoint_selection(grade,0.)
    source_regression=max(source[e]/baselines['source_baseline_loss'][e]-1 for e in ('TC1','NTD','TC2','COURSE'))
    practice_regression=max(practice[e]/baselines['practice_baseline_loss'][e]-1 for e in ('TC1','NTD','TC2','COURSE'))
    regressions=[after[e]/baselines[before][e]-1 for before,after in (
        ('source_baseline_loss',source),('initial_focused_source_loss',source),
        ('practice_baseline_loss',practice),('initial_focused_practice_loss',practice),
        ('parent_source_loss',source),('parent_practice_loss',practice)) for e in ('TC1','NTD','TC2','COURSE')]
    deficit=sum(max(0.,.01-g['relative_improvement']) for g in gates.values())
    if max(regressions)>.02:score=-100.-max(regressions)
    elif all(g['accepted'] for g in gates.values()):score=5.+semantic['score']
    else:score=-3.-deficit
    return {**semantic,'score':score,'max_source_regression':source_regression,'max_practice_regression':practice_regression,
        'all_text_gates_accepted':all(g['accepted'] for g in gates.values()),'text_gates':gates}


def fresh_test_cases():
    return restorative_cases('test',seed_override=71435073,case_namespace='source-consolidation-20261028')


def design():
    return {'mode':'SOURCE_CONSOLIDATION_NO_FINAL_TEST_SCORES_USED','initial_adapter_sha256':START_SHA,
        'parent_receipt_sha256':PARENT_RECEIPT_SHA,'source_train_repetitions':3,'extra_ntd_source_repetitions':1,
        'legacy_ntd_extra_repeats':5,'extra_typed_risk_repeats':2,'extra_typed_ac_repeats':1,
        'reasoned_training_rows':0,'maximum_epochs':2,'training_wall_seconds':8400,'learning_rate':8e-6,
        'development_mode':'greedy_direct','development_max_new_tokens':128,'development_seconds':10,
        'evaluation_every_attempted_steps':512,'source_validation_windows':178,'practice_validation_examples':122,
        'reference_models_for_all_text_gates':['previous40','focused69','restorative768'],
        'validation_pages':70,'source_test_pages_used_for_training_or_selection':0,
        'fresh_direct_test_questions_sha256':questions_digest(question_rows(fresh_test_cases())),
        'fresh_direct_test_cases':112,'source_and_market_test_scores_unread_before_precommit':True,
        'selection':'text preservation and >=1% improvement gates precede weakest-rule score; >2% any-reference book regression vetoes',
        'candidate_selection_uses_only_development':True,'execution_allowed':False,'trading_qualification':False}
