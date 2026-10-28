"""Frozen 30-case numerical market comparison over twenty UTC dates.

This is an archived, correlated directional diagnostic. It is not expert wave
supervision or a complete Williams campaign with exchange fills and stops.
"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path

from .independent_chart_holdout import build_blind_market_cases, grade_blind_predictions

ARCHIVE_SHA='d5dfe1fc49df84c8b8ea7b3e095f676b462ae455fe204c3f8dd43eaea41759f7'
SYMBOLS=('ADAUSDT','AVAXUSDT','BCHUSDT','BNBUSDT','BTCUSDT','DOGEUSDT','DOTUSDT','ETHUSDT','LINKUSDT','LTCUSDT','NEARUSDT','SOLUSDT','SUIUSDT','TRXUSDT','XRPUSDT')
FEE_PER_SIDE=0.0004
SLIPPAGE_PER_SIDE=0.0002


def frozen_market_cohort(path:Path):
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=ARCHIVE_SHA:raise ValueError('FROZEN_EXISTING_MARKET_ARCHIVE_CHANGED')
    archive=json.loads(raw)
    if tuple(sorted(archive['pairs']))!=SYMBOLS:raise ValueError('FROZEN_FIFTEEN_MARKETS_REQUIRED')
    cuts={s:(300+48*(i%10),612+48*(i%10)) for i,s in enumerate(SYMBOLS)}
    questions,keys,seal=build_blind_market_cases(archive,cuts_by_symbol=cuts)
    dates={q['market_context']['cutoff'][:10] for q in questions}
    if len(dates)!=20:raise ValueError('EXACT_TWENTY_UTC_DATES_REQUIRED')
    qbytes=('\n'.join(json.dumps(q,ensure_ascii=False,separators=(',',':')) for q in questions)+'\n').encode()
    seal.update({'mode':'FROZEN_30_NUMERIC_H1_CASES_TWENTY_DATES_RESEARCH_ONLY',
        'archive_sha256':ARCHIVE_SHA,'questions_file_sha256':sha256(qbytes).hexdigest(),'utc_dates':sorted(dates),
        'previous_two_cut_cohort_reused':False,'archive_previously_available_to_developers':True,
        'forecast_cases_not_statistically_independent':True,'expert_wave_labels':False,
        'class_score_method':'normalized_full_label_plus_eos_likelihood',
        'free_generation_protocol':{'mode':'greedy_non_thinking','maximum_new_tokens':64,'seconds_per_response':15,'strict_one_word_class':True},
        'precommitted_fee_per_side_scenario':FEE_PER_SIDE,'precommitted_slippage_per_side_scenario':SLIPPAGE_PER_SIDE,
        'fees_are_assumptions_not_verified_exchange_rates':True,'candidate_selection_on_this_exam':False})
    return questions,keys,qbytes,seal


def friction_diagnostic(keys,predictions):
    lookup={r['case_id']:r for r in keys}
    grade_blind_predictions(keys,predictions)
    returns=[];directions=[]
    for row in predictions:
        p=row['probabilities'];direction=max(('UP','DOWN','RANGE'),key=lambda k:p[k]);directions.append(direction)
        if direction!='RANGE':
            gross=lookup[row['case_id']]['future_return']*(1 if direction=='UP' else -1)
            returns.append(gross-2*(FEE_PER_SIDE+SLIPPAGE_PER_SIDE))
    return {'conditional_24h_directional_positions':len(returns),'range_abstentions':directions.count('RANGE'),
        'mean_net_return_per_conditional_position':sum(returns)/len(returns) if returns else None,
        'mean_net_return_per_case':sum(returns)/len(predictions),'positive_positions':sum(r>0 for r in returns),
        'round_trip_friction_assumption':2*(FEE_PER_SIDE+SLIPPAGE_PER_SIDE),
        'not_a_full_williams_strategy_backtest':True,'funding_fills_intrabar_stops_not_modeled':True,
        'overlapping_assets_and_dates_not_independent':True,'execution_allowed':False}


def grade_market_comparison(keys,predictions):
    stages={s:{**grade_blind_predictions(keys,rows),'friction_diagnostic':friction_diagnostic(keys,rows),
        'free_generation':free_generation_grade(keys,rows)} for s,rows in predictions.items()}
    uniform=[{'case_id':k['case_id'],'probabilities':dict.fromkeys(('UP','DOWN','RANGE'),1/3)} for k in keys]
    return {'mode':'ACTUAL_QWEN_ARCHIVED_DIRECTIONAL_DIAGNOSTIC_NOT_TRADING_QUALIFICATION','stages':stages,
        'uniform_probability_baseline':grade_blind_predictions(keys,uniform),'cases':len(keys),
        'trading_qualification':False,'execution_allowed':False,'candidate_selection_on_this_exam':False}


def free_generation_grade(keys,rows):
    lookup={k['case_id']:k for k in keys}
    if len(rows)!=len(lookup) or len({r['case_id'] for r in rows})!=len(rows) or set(lookup)!={r['case_id'] for r in rows}:
        raise ValueError('EXACT_GENERATIVE_COHORT_REQUIRED')
    correct=valid=0;labels={};by_case=[];position_returns=[];range_count=0
    for row in rows:
        if type(row.get('generated_complete')) is not bool or type(row.get('generation_time_limit_reached')) is not bool:
            raise ValueError('EXPLICIT_GENERATION_COMPLETION_REQUIRED')
        label=row.get('generated_response','').strip()
        good=label in ('UP','DOWN','RANGE') and row['generated_complete'] and not row['generation_time_limit_reached']
        ok=good and label==lookup[row['case_id']]['true_class'];correct+=ok;valid+=good
        if good:
            labels[label]=labels.get(label,0)+1
            if label=='RANGE':range_count+=1
            else:position_returns.append(lookup[row['case_id']]['future_return']*(1 if label=='UP' else -1)-2*(FEE_PER_SIDE+SLIPPAGE_PER_SIDE))
        by_case.append({'case_id':row['case_id'],'valid_final_class':good,'correct':ok})
    return {'cases':len(rows),'correct':correct,'accuracy':correct/len(rows),'valid_completed_class_answers':valid,
        'invalid_or_incomplete_answers':len(rows)-valid,'class_counts':labels,'range_abstentions':range_count,
        'invalid_answers_are_abstentions_in_friction_only':True,'conditional_24h_positions':len(position_returns),
        'mean_scenario_net_return_per_case':sum(position_returns)/len(rows),'scenario_round_trip_friction':2*(FEE_PER_SIDE+SLIPPAGE_PER_SIDE),
        'not_a_complete_strategy_backtest':True,'per_case':by_case,'trading_qualification':False,'execution_allowed':False}
