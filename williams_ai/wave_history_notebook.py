"""Incremental retrospective wave-pattern *study* of the ENTIRE cached archive.

The data collector stores ALL fetched historical Binance market bars; recent
chart inference is NOT equivalent to studying all of them. This bounded,
resumable notebook scans from the earliest cached bars to the newest at
successive historical, strictly CAUSAL checkpoints. The notebook records
only independently observable structure, AO/AC, indicator state, valid
fractal candidates and competing Elliott skeletons. It does not relabel
past pivots with future bars, train GGUF weights or authorize any order.

The chosen stride is a sampling design: skipping checkpoint timestamps
does NOT mean skipping the candles available inside their history windows.
"""
from __future__ import annotations

from datetime import datetime,timezone
import json
import math
from typing import Any

from .binance_wave_history import (
    ANALYSIS_TFS,DurableWaveHistory,HistoryUnavailable,next_open_ms,
)
from .chart_reader import analyze_chart

ALGORITHM_REVISION="WILLIAMS_ALL_BINANCE_TF_CAUSAL_HIST_REVIEW_V1"
MIN_CLOSED_BARS=45


class WaveArchiveNotebook:
    def __init__(self,store:DurableWaveHistory):
        self.store=store
        with store._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS wave_review(
                venue TEXT NOT NULL, symbol TEXT NOT NULL,
                interval TEXT NOT NULL, review_revision TEXT NOT NULL,
                asof_open_ms INTEGER NOT NULL, review_json TEXT NOT NULL,
                PRIMARY KEY(venue,symbol,interval,review_revision,asof_open_ms))""")

    def _last_reviewed(self,venue:str,symbol:str,interval:str)->int|None:
        with self.store._connect() as conn:
            r=conn.execute(
                "SELECT MAX(asof_open_ms) FROM wave_review "
                "WHERE venue=? AND symbol=? AND interval=? AND review_revision=?",
                (venue,symbol,interval,ALGORITHM_REVISION)).fetchone()
        return r[0]

    def _cutoff_for_next(self,venue:str,symbol:str,interval:str,
                         stride_bars:int)->int|None:
        last=self._last_reviewed(venue,symbol,interval)
        with self.store._connect() as conn:
            if last is None:
                # First indicator-capable reference: first 45 fully closed
                # Binance candles, no future and no synthetic prefix.
                rows=conn.execute(
                    "SELECT open_ms FROM candles "
                    "WHERE venue=? AND symbol=? AND interval=? "
                    "ORDER BY open_ms ASC LIMIT 1 OFFSET ?",
                    (venue,symbol,interval,MIN_CLOSED_BARS-1)).fetchone()
            else:
                rows=conn.execute(
                    "SELECT open_ms FROM candles "
                    "WHERE venue=? AND symbol=? AND interval=? AND open_ms>? "
                    "ORDER BY open_ms ASC LIMIT 1 OFFSET ?",
                    (venue,symbol,interval,last,stride_bars-1)).fetchone()
        return rows[0] if rows else None

    def _asof_rows(self,venue:str,symbol:str,interval:str,
                   at_ms:int,window:int)->list[dict[str,Any]]:
        with self.store._connect() as conn:
            x=conn.execute(
                "SELECT open_ms,close_ms,open,high,low,close,volume,quote_volume,"
                "trades,taker_buy_base_volume,taker_buy_quote_volume "
                "FROM candles WHERE venue=? AND symbol=? AND interval=? "
                "AND open_ms<=? ORDER BY open_ms DESC LIMIT ?",
                (venue,symbol,interval,at_ms,window)).fetchall()
        x.reverse()
        rows=[]
        prev=None
        for line in x:
            op,cl=line[0],line[1]
            if prev is not None and op!=next_open_ms(prev,interval):
                raise HistoryUnavailable("WAVE_ARCHIVE_HOLE_NEVER_INFER_WAVE")
            if cl!=next_open_ms(op,interval)-1:
                raise HistoryUnavailable("WAVE_ARCHIVE_CALENDAR_MISMATCH")
            rows.append({
                "close_time":datetime.fromtimestamp(cl/1000,timezone.utc).isoformat(),
                "open":float(line[2]),"high":float(line[3]),
                "low":float(line[4]),"close":float(line[5]),
                "volume":float(line[6]),"quote_volume":float(line[7]),
                "number_of_trades":line[8],
                "taker_buy_base_volume":float(line[9]),
                "taker_buy_quote_volume":float(line[10]),
                "closed":True,
            })
            prev=op
        return rows

    def review(self,*,venue:str,symbol:str,interval:str,
               checkpoint_budget:int=10,stride_bars:int=120,
               window_bars:int=1400)->dict[str,Any]:
        if (interval not in ANALYSIS_TFS
            or not isinstance(checkpoint_budget,int) or isinstance(checkpoint_budget,bool)
            or not 1<=checkpoint_budget<=1000
            or not isinstance(stride_bars,int) or isinstance(stride_bars,bool)
            or not 1<=stride_bars<=1000
            or not isinstance(window_bars,int) or isinstance(window_bars,bool)
            or not 45<=window_bars<=5000):
            raise ValueError("INVALID_WAVE_ARCHIVE_REVIEW_BUDGET")
        examined=0
        last=None
        meta=self.store.status(venue,symbol,interval)
        if meta["stored_closed_bars"]<MIN_CLOSED_BARS:
            return {
                "interval":interval,"status":"INSUFFICIENT_CACHED_BARS",
                "book_accurate_wave_degree_not_proven":True,
                "new_checkpoints":0,"trade_action":None,"execution_allowed":False,
            }
        for _ in range(checkpoint_budget):
            cutoff=self._cutoff_for_next(venue,symbol,interval,stride_bars)
            if cutoff is None:
                break
            bars=self._asof_rows(venue,symbol,interval,cutoff,window_bars)
            view=analyze_chart(symbol,ANALYSIS_TFS[interval],bars,
                as_of=bars[-1]["close_time"],tick_size=float(meta["tick_size"]))
            if (view["gaps"] or view["last_closed_at"]!=view["as_of"]
                    or view["execution_allowed"] is not False):
                raise HistoryUnavailable("WAVE_ARCHIVE_FUTURE_OR_GAP_REJECTED")
            waves=view["wave_structure"]
            item={
                "mode":"BINANCE_CAUSAL_ARCHIVE_WAVE_NOTE",
                "review_revision":ALGORITHM_REVISION,
                "venue":venue,"symbol":symbol,"interval":interval,
                "as_of":view["as_of"],
                "asof_open_ms":cutoff,
                "closed_bars_visible_at_cutoff":len(bars),
                "market_context":view["evidence"]["market_context"],
                "ao":view["evidence"]["ao"],
                "ac":view["evidence"]["ac"],
                "alligator_jaw":view["evidence"]["jaw_visible"],
                "alligator_teeth":view["evidence"]["teeth_visible"],
                "alligator_lips":view["evidence"]["lips_visible"],
                "zone":view["evidence"]["zone"],
                "confirmed_pivots":[
                    {k:v.get(k) for k in ("index","price","type","time","confirmed_at_index")}
                    for v in waves.get("recent_confirmed_swings",[])[-6:]
                    if isinstance(v,dict)
                ],
                "possible_impulse_structures":len(waves.get("impulse_alternatives",[])),
                "possible_corrective_structures":len(waves.get("corrective_alternatives",[])),
                "wave_degree": "UNKNOWN",
                "all_hypotheses_unverified":True,
                "100_140_bar_scale_is_a_heuristic_not_proof":True,
                "future_bars_excluded":True,
                "book_image_expert_label_unavailable":True,
                "model_weights_changed":False,
                "trade_action":None,"execution_allowed":False,
            }
            data=json.dumps(item,sort_keys=True,ensure_ascii=False,allow_nan=False)
            with self.store._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    conn.execute(
                        "INSERT INTO wave_review VALUES(?,?,?,?,?,?)",
                        (venue,symbol,interval,ALGORITHM_REVISION,cutoff,data))
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
            last=cutoff
            examined+=1
        with self.store._connect() as conn:
            record=conn.execute(
                "SELECT COUNT(*),MIN(asof_open_ms),MAX(asof_open_ms) "
                "FROM wave_review WHERE venue=? AND symbol=? AND interval=? "
                "AND review_revision=?",
                (venue,symbol,interval,ALGORITHM_REVISION)).fetchone()
        state=(
            "ARCHIVE_FULLY_STUDIED_AT_DECLARED_STRIDE" if self._cutoff_for_next(
                venue,symbol,interval,stride_bars) is None else
            "ARCHIVE_MORE_CLOSED_HISTORY_TO_REVIEW"
        )
        return {
            "interval":interval,"venue":venue,"symbol":symbol,
            "study_status":state,
            "first_reviewed_open_ms":record[1],
            "last_reviewed_open_ms":record[2],
            "checkpoints_stored":record[0],
            "new_checkpoints":examined,
            "step_between_checkpoints_bars":stride_bars,
            "window_bars_per_checkpoint":window_bars,
            "history_between_checkpoints_not_ignored":True,
            "book_accurate_wave_degree_not_proven":True,
            "local_GGUF_weights_trained_by_review":False,
            "trade_action":None,"execution_allowed":False,
        }
