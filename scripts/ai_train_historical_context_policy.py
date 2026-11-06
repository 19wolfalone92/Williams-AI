"""Train existing numeric research heads on the declared 2022/24/25 packs.

This does not modify Qwen. It reuses the established TC2 proxy replay, journal,
cost model and frozen-selection gates without adding a new trading strategy.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from williams_ai.historical_rule_curriculum import canonical, iso, load_pack, check_partitions, INTERVALS
from williams_ai.williams_signal_experience import signal_conditioned_outcomes
from williams_ai.continual_trader import PrivateExperience, learn_from_private_experience
from williams_ai.prequential_skill import causal_incremental_study
from williams_ai.prequential_net_skill import causal_expected_net_study

DESIGN = {"mode":"THREE_YEAR_TC2_PROXY_CONTEXT_POLICY_RESEARCH_20261106",
          "train_until":"2023-01-01T00:00:00+00:00",
          "validation_until":"2025-01-01T00:00:00+00:00",
          "observe_until":"2026-01-01T00:00:00+00:00",
          "horizon":12,"fee_bps_each_side":4,"slippage_bps_each_side":2,
          "quantity_or_leverage_not_simulated":True,
          "ticks_are_declared_research_assumptions_not_historical_exchange_filters":True,
          "training_epochs_batch_head":100,"online_learning_rate":0.02,"online_l2":0.002,
          "threshold_for_research_selection":0.6,
          "validation_and_test_do_not_update_weights":True,
          "model_hyperparameters_not_selected_on_test":True,
          "qwen_weight_training_claimed":False,"expert_tc2_entries_claimed":False,"execution_allowed":False}


def train(output):
    if output.exists():raise ValueError("NEW_CONTEXT_POLICY_OUTPUT_REQUIRED")
    output.mkdir(parents=True,mode=0o700)
    (output/"frozen_design.json").write_bytes(canonical(DESIGN))
    packs={s:load_pack(Path("examples/historical_rule_20261029")/(s+".json.gz")) for s in ("train","validation","test")}
    periods=check_partitions(packs)
    ledger=PrivateExperience(output/"experience.sqlite3")
    replay=[];source_packs={};counts={};total=0
    for split,pack in packs.items():
        source_packs[split]=sha256(Path("examples/historical_rule_20261029",split+".json.gz").read_bytes()).hexdigest()
        collected=[]
        for d in pack["datasets"]:
            symbol=d["symbol"];tf=INTERVALS[d["interval"]][2]
            rows=[{**dict(zip(("close_time","open","high","low","close","volume"),
                              (iso(r[0]),*map(float,r[1:6])))),"closed":True} for r in d["bars"]]
            tick=0.1 if symbol in {"BTCUSDT","ETHUSDT","BNBUSDT"} else 0.001
            result=signal_conditioned_outcomes(symbol,rows,tf,tick_size=tick,horizon=DESIGN["horizon"],
                fee_bps=DESIGN["fee_bps_each_side"],slippage_bps=DESIGN["slippage_bps_each_side"])
            collected.extend(result["outcomes"])
            replay.append({"split":split,"symbol":symbol,"timeframe":tf,"research_tick":tick,
                "candidate_checks":result["research_approximated_wm_candidates"],"resolved_paper_cases":len(result["outcomes"]),
                "setup_counts":result["setup_counts"],"source_sha256":result["source_sha256"]})
        ledger.append(collected,knowledge_as_of=DESIGN["observe_until"])
        counts[split]=len(collected);total+=len(collected)
        print("CONTEXT_CLOSED_HISTORY_REPLAY",json.dumps({"split":split,"resolved_paper_cases":len(collected)}),flush=True)
    cutoffs={k:DESIGN[k] for k in ("train_until","observe_until")}
    batch=learn_from_private_experience(ledger,validate_until=DESIGN["validation_until"],**cutoffs)
    online=causal_incremental_study(ledger,validation_until=DESIGN["validation_until"],**cutoffs,
        learning_rate=DESIGN["online_learning_rate"],l2=DESIGN["online_l2"])
    expected_net=causal_expected_net_study(ledger,validation_until=DESIGN["validation_until"],**cutoffs)
    receipt={"design":DESIGN,"design_sha256":sha256(canonical(DESIGN)).hexdigest(),
        "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "runner_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_packs_sha256":source_packs,"global_partition_ranges":periods,"resolved_cases_by_split":counts,
        "replay":replay,"batch_numeric_model":batch,"online_numeric_model":online,"online_expected_net_model":expected_net,
        "limitations":["Existing WM1/WM2/WM3 approximations, not expert reviewed Williams trades.",
            "Public historical regimes already available to developers; not a prospective market test.",
            "Overlapping hypothetical observations are not independent trades or a financed portfolio.",
            "Funding, partial fills, actual stop orders and market impact are not modeled.",
            "Probabilities have no certified empirical calibration outside this historical protocol."],
        "qwen_weights_changed":False,"execution_allowed":False,"trading_qualification":False}
    (output/"context_policy_result.json").write_bytes(canonical(receipt))
    print("CONTEXT_MODELS_TRAINED_RESEARCH_ONLY",json.dumps({"cases":total,"batch_state":batch["validation_state"],
        "batch_validation":batch["validation"],"batch_test":batch["frozen_test"],"execution_allowed":False}),flush=True)
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--new-output",type=Path,required=True)
    train(p.parse_args().new_output)
