"""Evaluate REAL local GGUF book reasoning on one HISTORICAL CLOSED prefix.

Usage:
python -m williams_ai.reasoning_trial --symbol ETHUSDT --bar-index 900 \
  --edition TC2 --local-model /private/real-model.gguf

Or inspect a saved proposal without a GGUF:
python -m williams_ai.reasoning_trial --symbol ETHUSDT --bar-index 900 \
  --edition TC2 --proposal-file /private/hypothesis.json

The next candle is NEVER exposed to the GGUF or the rule critic. No orders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .algorithm_guard import audit_reasoning
from .causal_paper_lab import FROZEN_DATA,frozen_markets
from .chart_reader import analyze_chart
from .causal_timeframes import aggregate_closed_h1


def evaluate_historical_thought(
    symbol:str, rows:list[dict[str,Any]], *,
    tick_size:float,bar_index:int,edition:str="TC2",
    model:Any=None,proposal:dict[str,Any]|None=None,
    include_topdown:bool=True,
    book_index:Path|None=None,
    coach_notebook:Path|None=None,
    model_id:str|None=None,
    study_memory:Path|None=None,
    experience_journal:Path|None=None,
    deliberate_study:bool=True,
    campaign_journal:dict[str,Any]|None=None,
    public_rule_learning:bool=False,
    research_memory:Path|None=None,
    research_question:str|None=None,
)->dict[str,Any]:
    """bar_index is a 0-based last closed bar, not next bar to trade.

    OHLCV that follows bar_index is excluded before ANY analysis call.
    """
    if type(public_rule_learning) is not bool or (public_rule_learning and (
            book_index is not None or study_memory is None or model is None or not deliberate_study)):
        raise ValueError("Public rule learning requires model/memory and no private book index")
    if (not isinstance(bar_index,int) or isinstance(bar_index,bool)
            or not 60<=bar_index<len(rows)):
        raise ValueError("Need an index of a closed bar after warmup")
    if (model is None)==(proposal is None):
        raise ValueError("Provide exactly one of real local GGUF or saved proposal")
    prefix=rows[:bar_index+1]
    as_of=prefix[-1]["close_time"]
    evidence=analyze_chart(symbol,"H1",prefix,as_of=as_of,tick_size=tick_size)
    if evidence["gaps"] or evidence["execution_allowed"] is not False:
        raise ValueError("No causal and contiguous market history")
    if include_topdown:
        # Efficiently reconstruct only completely CLOSED higher bars. Reuse
        # already calculated H1 observation, avoiding an expensive duplicate
        # full H1 chart analysis on every historical checkpoint.
        higher={}
        for tf in ("D1","H4"):
            closed=aggregate_closed_h1(prefix,tf)
            if len(closed)<45:
                higher[tf]={"status":"INSUFFICIENT_CLOSED_HISTORY",
                            "bars":len(closed),"wave_degree":"UNKNOWN"}
            else:
                tf_view=analyze_chart(
                    symbol,tf,closed,as_of=as_of,tick_size=tick_size)
                higher[tf]={
                    "status":"OBSERVED_CLOSED_BARS",
                    "bars":len(closed),
                    "closed_through":closed[-1]["close_time"],
                    "market_context":tf_view["evidence"]["market_context"],
                    "ao":tf_view["evidence"]["ao"],
                    "ac":tf_view["evidence"]["ac"],
                    "wave_degree":tf_view["wave_structure"]["wave_degree"],
                    "wave_hypotheses_unverified":True,
                    "recent_levels":tf_view["market_position"]["recent_levels"],
                    "nonlinear_structure":tf_view["principle_evidence"]
                        ["observable_facts"]["multi_resolution_price_structure"],
                    "supported_not_proven_market_scenarios":[
                        q["hypothesis"] for q in tf_view["principle_evidence"]
                            ["competing_market_scenarios"]
                        if q["status"]=="EVIDENCE_COMPATIBLE_NOT_PREDICTION"
                    ],
                }
        from .market_hierarchy import reconcile_market_hierarchy
        frames_for_relations={**higher,"H1":{
            "status":"OBSERVED_CLOSED_BARS",
            "closed_through":evidence["last_closed_at"],
            "market_context":evidence.get("evidence",{}).get("market_context","UNKNOWN"),
            "ao":evidence.get("evidence",{}).get("ao"),
            "ac":evidence.get("evidence",{}).get("ac"),
            "recent_levels":evidence["market_position"]["recent_levels"],
        }}
        higher["conditional_parent_child_relations"] = (
            reconcile_market_hierarchy(frames_for_relations)["parent_child_relations"]
        )
        from .cross_scale_wave_theses import cross_scale_wave_theses
        # These hypotheses remain competing and falsifiable. Never claim
        # that AO near zero proves wave four or that H1 correction is a
        # certified higher degree. All TF passports are fully CLOSED.
        higher["cross_scale_wave_theses"] = cross_scale_wave_theses(
            frames_for_relations,as_of=evidence["as_of"])
        evidence["higher_timeframe_context"]=higher
    # Teach the genuine local Qwen prompt distinct TC2 entry, campaign
    # management and exit rules, using ONLY the historical H1 prefix. This
    # always remains a source-grounded lesson, NOT an execution signal.
    if campaign_journal is not None and edition!="TC2":
        raise ValueError("Campaign stop review currently requires TC2")
    if edition=="TC2":
        from .tc2_trade_lifecycle_coach import teach_tc2_trade_lifecycle
        evidence["tc2_trade_lifecycle_lesson"]=teach_tc2_trade_lifecycle(
            evidence,prefix,tick_size=tick_size,journal=campaign_journal)
    # Independently reconstruct a practical, source-specific "what facts
    # would confirm or falsify my hypothesis?" lesson for the SAME chart
    # prefix. This is how book knowledge reaches the real Qwen decision
    # step, rather than staying a detached theoretical curriculum.
    if model is not None:
        from .practical_market_teacher import rehearse_trade_reasoning
        evidence["practical_market_teacher"]=rehearse_trade_reasoning(
            symbol,"H1",prefix,edition=edition,tick_size=tick_size)
    # The GGUF can review only PAPER outcome feedback already resolved by
    # this closed H1 timestamp; the notebook must never act as book proof.
    experience_notes=None
    if experience_journal is not None:
        if edition!="TC2" or model is None:
            raise ValueError("Only TC2 local GGUF can consume TC2 paper feedback")
        from .continual_trader import PrivateExperience
        experience_notes=PrivateExperience(experience_journal).past_feedback(
            symbol=symbol,as_of=as_of)
        evidence["past_experience"]=experience_notes
    sources=[]
    if book_index is not None:
        if model is None:
            raise ValueError("Private source pages require a locally running model")
        from .grounded_inference import verified_private_book_pages
        current=evidence.get("candidates",[])
        latest=current[-1]["type"] if current else None
        sources=verified_private_book_pages(book_index,edition,selected_setup=latest)
    prior=[]
    if coach_notebook is not None:
        if model is None or model_id is None:
            raise ValueError("Private lessons require an actual model fingerprint")
        from .adaptive_coach import recall_model_corrections
        related=[]
        names={"WM1":("TC2.WM1.BAR","TC2.WM1.ANG","TC2.WM1.TRIGGER"),
               "WM2_SUPER_AO":("TC2.WM2",),
               "WM3_FRACTAL":("TC2.WM3",)}
        for candidate in evidence.get("candidates",[])[-12:]:
            if isinstance(candidate,dict):
                related.extend(names.get(candidate.get("type"),()))
        if edition=="TC1":
            related.append("TC1.FRACTAL")
        if edition=="NTD":
            related.extend(("NTD.AO","NTD.AC","NTD.FRACTAL"))
        focus=tuple(sorted(set(
            x for x in related if x.startswith(edition+".")
        )))
        prior=recall_model_corrections(coach_notebook,model_id,edition,
                                       focus_rule_ids=focus,limit=2)
        # If the active signals have no matching corrected rules, still
        # recall a single general lesson in the SAME edition. Otherwise a
        # model that studied TC1 wave degree but sees only TC1 fractals
        # would never receive its prior book knowledge.
        if not prior and focus:
            prior=recall_model_corrections(coach_notebook,model_id,edition,
                                           limit=1)
    studied=[]
    materials=[]
    if research_memory is not None:
        if model is None:
            raise ValueError("RESEARCH_MEMORY_REQUIRES_LOCAL_MODEL")
        from .unified_memory import UnifiedMemory
        query=research_question or "фрактал волны AO вход сопровождение стоп выход риск"
        materials=UnifiedMemory(research_memory).retrieve(query,edition=edition,as_of=as_of)
    if study_memory is not None:
        if model is None or model_id is None:
            raise ValueError("Book study recall requires actual model and model ID")
        from .study import recall
        vocabulary={
            "TC1":"волна коррекция дивергенция фрактал целевая зона",
            "NTD":"фрактал Аллигатор AO AC зона Линия Баланса",
            "TC2":"Мудрец ангуляция фрактал Super AO стоп сопровождение",
            "COURSE":"волны Hump Thumb Profitunity структура ритм",
        }
        observed=" ".join(
            x.get("type","") for x in evidence.get("candidates",[])
            if isinstance(x,dict)
        )[:220]
        if not public_rule_learning:
            studied=recall(study_memory,vocabulary[edition]+" "+observed,
                           edition=edition,limit=2,model_id=model_id,
                           book_index=book_index)
    if model is not None:
        args={"edition":edition}
        if book_index is not None:
            args["primary_pages"]=sources
        if coach_notebook is not None:
            args["prior_lessons"]=prior
        if study_memory is not None:
            args["study_lessons"]=studied
            args["model_id"]=model_id
        if research_memory is not None:
            args["research_materials"]=materials
        learner=None
        if deliberate_study and (book_index is not None or public_rule_learning) and study_memory is not None:
            from .deliberate_learning import DeliberateBookModel
            learner=DeliberateBookModel(model,book_index,study_memory,model_id=model_id,
                                       public_rules_only=public_rule_learning)
        submitted=(learner or model).propose_chart_reasoning(evidence,**args)
    else:
        submitted=proposal
    result=audit_reasoning(evidence,submitted)
    stop_review=None
    if campaign_journal is not None:
        lesson=evidence["tc2_trade_lifecycle_lesson"]
        context=lesson["management"]["stop_review_context"]
        if context is not None and model is not None:
            from .stop_review import review_with_model
            stop_review=review_with_model(model,context)
            if learner is not None:
                stop_review["next_reading_agenda_updated"] = learner.remember_stop_review(
                    evidence, context, stop_review)
        else:
            critical=lesson["management"].get("campaign_audit_critical_findings",[])
            flat=(lesson["campaign_state"]=="CLAIMED_FLAT_REQUIRES_EXCHANGE_RECONCILIATION"
                  and not critical)
            stop_review={"mode":"TC2_CAMPAIGN_STOP_REVIEW_NOT_CALLED",
                "model_called":False,
                "campaign_audit_critical_findings":critical,
                "audit":{"hard_failures":([] if flat else
                    ["CAMPAIGN_AUDIT_HAS_CRITICAL_FINDINGS"] if critical else
                    ["CAMPAIGN_REQUIRES_PROTECTION_OR_MODEL_REVIEW"]),
                         "verdict":"NO_OPEN_CAMPAIGN" if flat else "HALT_FOR_RECONCILIATION"},
                "execution_allowed":False,"trade_action":None}
    reason=submitted.get("reason","") if isinstance(submitted,dict) else ""
    return {
        "mode":"LOCAL_BLINDED_REASONING_TRIAL",
        "symbol":symbol,"edition":edition,
        "last_visible_bar_index":bar_index,
        "last_visible_at":as_of,
        "private_book_pages_used":[page["source"] for page in sources],
        "public_rule_learning_only":public_rule_learning,
        "past_learning_feedback_used":(
            experience_notes["past_examples_used"] if experience_notes else 0),
        "past_learning_feedback_only_from_previously_resolved_simulations":True,
        "private_prior_mistake_lessons_used":[note["source"] for note in prior],
        "private_research_material_refs_used":[item["source"] for item in materials],
        "same_model_unverified_book_lessons_used":[x["source"] for x in studied],
        "book_study_notes_checked_against_current_source":(
            study_memory is not None and book_index is not None),
        "private_book_passages_exported":False,
        "private_coaching_text_exported":False,
        "top_down_context":evidence.get("higher_timeframe_context",{}),
        "model_saw_future_bars":False,
        "model_weights_trained_here":False,
        "book_mastery_percent":None,
        "tc2_entry_management_exit_source_lesson_supplied":(
            edition=="TC2" and model is not None),
        "independent_practical_decision_teacher_supplied":model is not None,
        "competing_market_hypotheses_rehearsed":5 if model is not None else 0,
        "model_explanation":reason[:600] if isinstance(reason,str) else "",
        "audit":result,
        "campaign_stop_review":stop_review,
        "book_question_resolved":submitted.get("learning_need", "NONE") == "NONE"
            and submitted.get("setup") != "UNKNOWN",
        "research_reasoning_consistent":not result["hard_failures"] and
            submitted.get("learning_need", "NONE") == "NONE" and submitted.get("setup") != "UNKNOWN" and
            not (model is not None and learner and learner.latest and
                 learner.latest.get("needs_independent_topic_review")) and
            (stop_review is None or not stop_review["audit"]["hard_failures"]),
        "deliberate_book_learning":learner.latest if model is not None and learner else None,
        "trade_action":None,"execution_allowed":False,
    }


def main()->None:
    p=argparse.ArgumentParser(description="Blind by timestamp, source-scoped AI reasoning trial")
    p.add_argument("--dataset",type=Path,default=FROZEN_DATA)
    p.add_argument("--symbol",required=True)
    p.add_argument("--bar-index",required=True,type=int)
    p.add_argument("--edition",choices=["TC1","NTD","TC2","COURSE"],default="TC2")
    p.add_argument("--no-topdown",action="store_true",help="Disable complete H4/D1 context")
    p.add_argument("--ctx",type=int,default=8192,
                   help="Local GGUF context, including source text and bounded reasoning")
    p.add_argument("--reasoning-budget",type=int,default=2048,
                   help="Bounded actual Qwen reasoning tokens per learning call")
    p.add_argument("--book-index",type=Path,help="Private indexed four-book SQLite outside repository")
    p.add_argument("--lora-adapter",type=Path)
    p.add_argument("--lora-manifest",type=Path)
    p.add_argument("--public-rule-learning",action="store_true",
                   help="Explicitly use reviewed public paraphrases, NOT original private books")
    p.add_argument("--coach-notebook",type=Path,
                   help="Private GGUF-specific lesson notebook generated by adaptive_coach")
    p.add_argument("--study-memory",type=Path,
                   help="Private book study notebook for this EXACT GGUF fingerprint")
    p.add_argument("--no-deliberate-study",action="store_true",
                   help="Disable bounded read-study-reconsider loop for an unassisted baseline")
    p.add_argument("--experience-journal",type=Path,
                   help="Private resolved TC2 WM1/2/3 feedback for real local GGUF")
    p.add_argument("--campaign-journal",type=Path,
                   help="Private bounded TC2 campaign JSON, claims not broker authentication")
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--local-model",type=Path)
    group.add_argument("--proposal-file",type=Path)
    a=p.parse_args()
    if (a.lora_adapter or a.lora_manifest) and not a.local_model:
        p.error("LoRA options require --local-model")
    if a.public_rule_learning and (a.book_index or a.no_deliberate_study or not a.local_model):
        p.error("--public-rule-learning requires local model, no private index and enabled study")
    if a.public_rule_learning and a.study_memory is None:
        p.error("--public-rule-learning requires an explicit private --study-memory path")
    markets=frozen_markets(a.dataset)
    if a.symbol not in markets:
        p.error("Unknown symbol in selected frozen dataset")
    model=None
    proposal=None
    if a.local_model:
        from .local_model import LocalWilliamsModel
        model=LocalWilliamsModel(a.local_model,n_ctx=a.ctx,
                                use_decision_schema=True,learning_mode="auto",
                                reasoning_budget=a.reasoning_budget,
                                lora_adapter=a.lora_adapter,lora_manifest=a.lora_manifest)
    else:
        proposal=json.loads(a.proposal_file.read_text(encoding="utf-8"))
    x=markets[a.symbol]
    campaign=None
    if a.campaign_journal:
        from .book_index import ROOT
        path=a.campaign_journal.expanduser().resolve()
        if (a.edition!="TC2" or ROOT in path.parents or path==ROOT
                or not path.is_file() or path.stat().st_size>1_000_000):
            p.error("--campaign-journal must be a private TC2 JSON outside the repository, <=1MB")
        campaign=json.loads(path.read_text(encoding="utf-8"))
    if a.coach_notebook and a.local_model is None:
        p.error("--coach-notebook requires --local-model")
    if a.study_memory and a.local_model is None:
        p.error("--study-memory requires --local-model")
    if a.experience_journal and (a.local_model is None or a.edition!="TC2"):
        p.error("--experience-journal requires local GGUF and TC2")
    model_id=None
    if a.book_index and a.local_model and not a.no_deliberate_study and a.study_memory is None:
        a.study_memory=a.book_index.parent/"Williams_AI_Decision_Study.sqlite3"
    if a.coach_notebook or a.study_memory:
        from .study_cli import _model_identity
        model_id=_model_identity(a.local_model,lora_adapter=a.lora_adapter,
                                lora_manifest=a.lora_manifest)
    output=evaluate_historical_thought(
        a.symbol,x["bars"],tick_size=x["tick_size"],bar_index=a.bar_index,
        edition=a.edition,model=model,proposal=proposal,
        include_topdown=not a.no_topdown,book_index=a.book_index,
        coach_notebook=a.coach_notebook,model_id=model_id,
        study_memory=a.study_memory,
        experience_journal=a.experience_journal,
        deliberate_study=not a.no_deliberate_study,
        campaign_journal=campaign,public_rule_learning=a.public_rule_learning)
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
