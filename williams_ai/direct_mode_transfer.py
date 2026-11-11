"""Select inference mode on development only, then score fresh direct cases.

The earlier thinking exam remains recorded. The new direct exam uses different
numeric operands; its percentages are not a paired causal mode comparison.
"""
from __future__ import annotations
from .restorative_book_practice import restorative_cases
from .contrastive_book_practice import RULES,question_rows,questions_digest,score_responses
from .focused_book_practice import checkpoint_selection


def direct_test_cases():
    return restorative_cases('test',seed_override=71435071,case_namespace='direct-transfer-20261028')


def select_development_mode(thinking_grade,direct_grade):
    cohorts=[]
    for g in (thinking_grade,direct_grade):
        if (g.get('total')!=112 or set(g['per_rule'])!=set(RULES) or any(v['total']!=16 for v in g['per_rule'].values())
            or abs(g['accuracy']-sum(v['correct'] for v in g['per_rule'].values())/112)>1e-12):
            raise ValueError('CONSISTENT_SEVEN_RULE_DEVELOPMENT_COUNTS_REQUIRED')
        ids=[r['case_id'] for r in g['cases']]
        if len(ids)!=112 or len(set(ids))!=112:raise ValueError('EXACT_DEVELOPMENT_COHORT_REQUIRED')
        cohorts.append(set(ids))
    if cohorts[0]!=cohorts[1]:raise ValueError('SAME_DEVELOPMENT_QUESTIONS_REQUIRED_FOR_MODE_SELECTION')
    a=checkpoint_selection(thinking_grade,0.)['score'];b=checkpoint_selection(direct_grade,0.)['score']
    return {'selected_mode':'direct' if b>a else 'thinking','thinking_development_selection_score':a,
        'direct_development_selection_score':b,'same_development_questions':True,'test_scores_used_for_selection':False,
        'rule':'minimum rule accuracy + 0.1 overall accuracy - 0.1 collapsed rules; ties keep thinking',
        'trading_qualification':False,'execution_allowed':False}


def grade_direct_test(cases,predictions,*,include_restorative_parent=False):
    expected={'base','previous40','focused69','candidate'}|({'restorative768'} if include_restorative_parent else set())
    if type(include_restorative_parent) is not bool or set(predictions)!=expected:raise ValueError('ALL_SAME_MODE_MODELS_REQUIRED')
    grades={s:score_responses(cases,rows) for s,rows in predictions.items()}
    return {'mode':'FRESH_PARAMETER_DIRECT_TRANSFER_NOT_EXPERT_MARKETS','stages':grades,'transfer_gate_passed':transfer_gate(grades,include_restorative_parent=include_restorative_parent),
        'same_questions_and_response_budget_each_model':True,'execution_allowed':False,'trading_qualification':False}


def transfer_gate(grades,*,include_restorative_parent=False):
    expected={'base','previous40','focused69','candidate'}|({'restorative768'} if include_restorative_parent else set())
    if type(include_restorative_parent) is not bool or set(grades)!=expected:raise ValueError('ALL_COMPARABLE_MODEL_GRADES_REQUIRED')
    c=grades['candidate']
    return (c['accuracy']>=.8 and c['accuracy']>=max(g['accuracy'] for s,g in grades.items() if s!='candidate')+.05
        and all(v['correct']/v['total']>=.75 for v in c['per_rule'].values()) and not c['constant_fields_despite_varied_expected_values'])


def direct_design():
    return {'mode':'DEVELOPMENT_MODE_SELECTION_THEN_FRESH_DIRECT_PARAMETER_TEST','cases':112,
        'questions_sha256':questions_digest(question_rows(direct_test_cases())),
        'development_questions_sha256':questions_digest(question_rows(restorative_cases('validation'))),
        'original_thinking_test_retained':True,'new_numeric_parameters':True,'new_phrasing_templates_claimed':False,
        'inference_mode':'direct','maximum_new_tokens':128,'seconds_per_response':10,
        'minimum_overall_accuracy':.8,'minimum_each_rule_accuracy':.75,'minimum_gain_over_all_references':.05,
        'checkpoint_selected_on_this_test':False,'expected_test_keys_sent_to_gpu':False,'execution_allowed':False}
