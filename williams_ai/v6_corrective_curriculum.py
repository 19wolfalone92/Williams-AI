"""Adaptive corrective SFT tasks built from verified Williams source rule cards."""
from __future__ import annotations
from collections import Counter
from decimal import Decimal
import hashlib,json,random
from typing import Any
from williams_ai.book_rules import load_rule_cards

SEED=20261125
GENERATOR_VERSION="v6-corrective-1.0"
SPLIT_COUNTS={"train":16,"validation":4,"development":4,"locked_holdout":4}
SPLIT_ORDER=("train","validation","development","locked_holdout")
RULES={
 "ao_difference":{"edition":"NTD","rule_id":"NTD.AO","pages":"55-57"},
 "zone":{"edition":"NTD","rule_id":"NTD.ZONE","pages":"91-92"},
 "fractal":{"edition":"TC1","rule_id":"TC1.FRACTAL","pages":"68-70"},
 "wm1_shape":{"edition":"TC2","rule_id":"TC2.WM1.BAR","pages":"121-122"},
 "wm1_stop":{"edition":"TC2","rule_id":"TC2.WM1.STOP","pages":"131"},
 "wm2":{"edition":"TC2","rule_id":"TC2.WM2","pages":"135-139"},
 "wm3":{"edition":"TC2","rule_id":"TC2.WM3","pages":"147-149"},
 "course_momentum":{"edition":"COURSE","rule_id":"COURSE.AO5345","pages":"42-45"}}
SYSTEM_PROMPT=("Analyse only the supplied closed-bar data and the stated Williams rule. "
 "Do not infer missing prices or future bars. Compute signs and comparisons explicitly. "
 "Return one JSON object with keys answer, entry_stop, rule_id, calculation, reason, "
 "counterexample, uncertainty, cancel_if. Use null for entry_stop when no order is justified.")

def _d(x): return format(Decimal(x).normalize(),"f")
def _answer(answer,rule_id,calculation,reason,counterexample,uncertainty,cancel_if,entry_stop=None):
 return json.dumps({"answer":answer,"entry_stop":None if entry_stop is None else str(entry_stop),"rule_id":rule_id,
  "calculation":calculation,"reason":reason,"counterexample":counterexample,"uncertainty":uncertainty,"cancel_if":cancel_if},
  ensure_ascii=False,sort_keys=True,separators=(",",":"))
def _row(split,family,i,prompt,answer,inputs):
 src=RULES[family]
 return {"case_id":f"corr_{family}_{split}_{i:03d}","split":split,"family":family,"edition":src["edition"],
  "rule_id":src["rule_id"],"source_pages":src["pages"],"inputs":inputs,
  "prompt":prompt.strip()+"\n\nUse only the supplied closed-bar values. Show the calculation and state what invalidates the conclusion.","answer":answer}

def _ao(rng,split,i,t=0):
 positive=i%2==0;gap=Decimal(rng.randint(125,975))/100;base=Decimal(rng.randint(2500,8500))/100
 s34=base+(0 if positive else gap);s5=base+(gap if positive else 0);ao=s5-s34
 p=[f"SMA(5)={_d(s5)} and SMA(34)={_d(s34)} on the same closed median-price bar. Calculate signed AO.",
 f"Do not use an absolute value. Given the two closed-bar averages SMA5 {_d(s5)} and SMA34 {_d(s34)}, calculate AO.",
 f"A calculation sheet gives the 5-period average {_d(s5)} and 34-period average {_d(s34)}. What is their AO difference, including sign?",
 f"The median-price SMA(5) is {_d(s5)}; SMA(34) is {_d(s34)}. Return the exact signed Awesome Oscillator value."]
 return _row(split,"ao_difference",i,p[t],_answer(_d(ao),RULES["ao_difference"]["rule_id"],
  f"AO = SMA5 - SMA34 = {_d(s5)} - {_d(s34)} = {_d(ao)}.","AO is a signed difference; preserve a negative sign rather than converting to magnitude.",
  f"Using {_d(abs(ao))} would discard direction and is incorrect.","None; both averages use the same completed bar.",
  "Recalculate if the averages come from different bars or price bases."),
  {"sma5":_d(s5),"sma34":_d(s34),"prompt_template_id":t})

def _zone(rng,split,i,t=0):
 kind=("GREEN","RED","GRAY","GRAY")[i%4];a0=Decimal(rng.randint(-800,800))/100;c0=Decimal(rng.randint(-800,800))/100
 da=Decimal(rng.randint(15,220))/100;dc=Decimal(rng.randint(15,220))/100
 if kind=="GREEN":a1,c1=a0+da,c0+dc
 elif kind=="RED":a1,c1=a0-da,c0-dc
 elif i%2==0:a1,c1=a0+da,c0-dc
 else:a1,c1=a0-da,c0+dc
 p=[f"Two consecutive closed bars: AO {_d(a0)} -> {_d(a1)}; AC {_d(c0)} -> {_d(c1)}. Classify GREEN, RED, or GRAY.",
 f"Compare the current AO {_d(a1)} and AC {_d(c1)} columns with prior values {_d(a0)} and {_d(c0)}. Which zone applies?",
 f"AO changed {_d(a0)} to {_d(a1)} and AC changed {_d(c0)} to {_d(c1)}. Determine the Profitunity zone from direction only.",
 f"On this closed bar, previous AO/AC were {_d(a0)}/{_d(c0)} and current AO/AC are {_d(a1)}/{_d(c1)}. Return the zone."]
 return _row(split,"zone",i,p[t],_answer(kind,RULES["zone"]["rule_id"],f"AO change={_d(a1-a0)}; AC change={_d(c1-c0)}.",
  "Both columns rising means GREEN; both falling means RED; opposite directions mean GRAY.",
  "The absolute sign of AO or AC is not the zone rule; compare each column with its prior column.",
  "None; consecutive closed-bar values are given.","Do not use an unclosed column or infer from zero-line position."),
  {"previous_ao":_d(a0),"current_ao":_d(a1),"previous_ac":_d(c0),"current_ac":_d(c1),"prompt_template_id":t})

def _fractal(rng,split,i,t=0):
 kind=("UP","DOWN","NONE","NONE")[i%4]
 if kind=="UP":
  m=Decimal(rng.randint(145,210));h=[m-7,m-3,m,m-5,m-2];l=[m-20,m-15,m-9,m-16,m-12]
 elif kind=="DOWN":
  b=Decimal(rng.randint(90,130));l=[b+8,b+4,b,b+6,b+2];h=[x+12 for x in l]
 elif i%4==2:
  m=Decimal(rng.randint(145,210));h=[m-5,m-2,m,m+1,m-3];l=[m-20,m-15,m-9,m-16,m-12]
 else:
  m=Decimal(rng.randint(145,210));h=[m-7,m-3,m,m-5,m-2];l=[m-20,m-15,m-9,m-16,m-12]
 closed=i%4!=3
 if not closed:kind="NONE"
 hs=[_d(x) for x in h];ls=[_d(x) for x in l]
 p=[f"Five bars are chronological. Highs={hs}; lows={ls}. Bar 3 is the candidate; both right bars are {'closed' if closed else 'not closed'}. Return UP, DOWN, or NONE.",
 f"Assess a five-bar fractal. Highs={hs}; lows={ls}. Candidate is bar 3; right-side bars closed={closed}. Classify UP, DOWN, or NONE.",
 f"With five chronological bars, compare bar 3 to two bars on either side. Highs={hs}; lows={ls}; right bars closed={closed}.",
 f"Only the five bars shown may be used. High sequence={hs}; low sequence={ls}; two right bars closed={closed}. Is the middle a confirmed fractal?"]
 comparison="middle high exceeds all four surrounding highs" if kind=="UP" else "middle low is below all four surrounding lows" if kind=="DOWN" else "the middle is not a strict extreme or the two right bars are not both closed"
 return _row(split,"fractal",i,p[t],_answer(kind,RULES["fractal"]["rule_id"],
  f"Bar-3 high={_d(h[2])}; bar-3 low={_d(l[2])}; {comparison}.",
  "A confirmed standard fractal needs the middle extreme relative to two bars on each side and two closed bars to its right.",
  "A middle high above the two left highs is insufficient if a right high exceeds it.",
  "Strict unequal highs/lows are used here; equal-bar counting is a separate canonical case.",
  "Do not confirm before both right-side bars close; never inspect future bars."),
  {"highs":hs,"lows":ls,"right_bars_closed":closed,"prompt_template_id":t})

def _wm1_shape(rng,split,i,t=0):
 direction="LONG" if i%2==0 else "SHORT";valid=(i//2)%2==0
 low=Decimal(rng.randint(80,160));high=low+Decimal(rng.randint(10,30));mid=(low+high)/2
 if direction=="LONG":
  prev=low+Decimal(rng.randint(1,6)) if valid else low-Decimal(rng.randint(1,6));close=mid+1 if valid else mid-1
  calc=f"LONG: current low {_d(low)} {'<' if low<prev else '>='} prior low {_d(prev)}; midpoint={_d(mid)}; close={_d(close)}."
  reason="A bullish reversal bar needs a lower low than the prior bar and a close in the upper half."
 else:
  prev=high-Decimal(rng.randint(1,6)) if valid else high+Decimal(rng.randint(1,6));close=mid-1 if valid else mid+1
  calc=f"SHORT: current high {_d(high)} {'>' if high>prev else '<='} prior high {_d(prev)}; midpoint={_d(mid)}; close={_d(close)}."
  reason="A bearish reversal bar needs a higher high than the prior bar and a close in the lower half."
 answer="YES" if valid else "NO"
 p=[f"Closed {direction} bar: prior relevant extreme={_d(prev)}, current L={_d(low)}, H={_d(high)}, C={_d(close)}. Does it satisfy only the TC2 reversal-bar shape?",
 f"Check the {direction} reversal-bar geometry. Previous comparison extreme={_d(prev)}; candidate L/H/C={_d(low)}/{_d(high)}/{_d(close)}. Valid shape?",
 f"No entry decision is requested. For a completed {direction} bar, compare its extreme with {_d(prev)} and its close with midpoint {_d(mid)}. Values L={_d(low)}, H={_d(high)}, C={_d(close)}.",
 f"A {direction} candidate has range [{_d(low)}, {_d(high)}], close {_d(close)}, prior comparison extreme {_d(prev)}. Is the bar shape valid?"]
 return _row(split,"wm1_shape",i,p[t],_answer(answer,RULES["wm1_shape"]["rule_id"],calc,reason,
  "A close in the wrong half fails even if the lower-low/higher-high condition is satisfied.",
  "None; bar is closed and all values are supplied.","This isolates bar geometry; it does not by itself prove Alligator context or a complete WM1 entry."),
  {"direction":direction,"previous_extreme":_d(prev),"low":_d(low),"high":_d(high),"close":_d(close),"prompt_template_id":t})

def _wm1_stop(rng,split,i,t=0):
 direction="LONG" if i%2==0 else "SHORT";low=Decimal(rng.randint(9000,18000))/100
 high=low+Decimal(rng.randint(300,1600))/100;tick=Decimal(rng.choice((1,2,5)))/100
 stop=low-tick if direction=="LONG" else high+tick
 p=[f"Closed {direction} reversal bar: low={_d(low)}, high={_d(high)}. Explicit protective offset={_d(tick)}. Calculate the initial stop beyond the opposite extreme.",
 f"Set only the original WM1 protective STOP for {direction}; signal-bar range [{_d(low)}, {_d(high)}], one tick={_d(tick)}.",
 f"For a {direction} reversal bar with L={_d(low)} and H={_d(high)}, apply an outward stop offset of {_d(tick)}. What is the stop price?",
 f"No sizing is requested. Direction={direction}; low={_d(low)}; high={_d(high)}; explicit offset={_d(tick)}. Compute the initial protection."]
 calc=f"{'LONG: low - offset' if direction=='LONG' else 'SHORT: high + offset'} = {_d(stop)}."
 return _row(split,"wm1_stop",i,p[t],_answer(_d(stop),RULES["wm1_stop"]["rule_id"],calc,
  "Place the original stop just beyond the opposite extreme of the closed reversal bar; apply the given offset outward.",
  "A LONG stop above the signal low or a SHORT stop below the signal high is on the wrong side.",
  "None; direction, extremes and tick offset are explicit.","Recalculate if the bar is not closed or the actual tick size changes."),
  {"direction":direction,"low":_d(low),"high":_d(high),"tick":_d(tick),"prompt_template_id":t})

def _wm2(rng,split,i,t=0):
 direction="LONG" if i%2==0 else "SHORT";valid=(i//2)%2==0
 start=Decimal(rng.randint(-500,500))/100;step=Decimal(rng.randint(20,180))/100
 values=[start+j*step*(1 if direction=="LONG" else -1) for j in range(4)]
 if not valid:values[3]=values[2]-step if direction=="LONG" else values[2]+step
 hi=Decimal(rng.randint(14000,26000))/100;lo=hi-Decimal(rng.randint(100,500))/100;tick=Decimal(rng.choice((1,2,5)))/100
 ok=all(values[j]<values[j+1] for j in range(3)) if direction=="LONG" else all(values[j]>values[j+1] for j in range(3))
 entry=hi+tick if ok and direction=="LONG" else lo-tick if ok else None
 answer=direction if ok else "NO_SIGNAL";ao=[_d(x) for x in values]
 p=[f"Four consecutive closed AO columns are {ao}. The price bar aligned with the third directional change has H={_d(hi)}, L={_d(lo)}; tick={_d(tick)}. Evaluate the TC2 WM2 signal and STOP entry for {direction}.",
 f"For a {direction} WM2 candidate, AO values in time order are {ao}. The third-change price bar has high {_d(hi)}, low {_d(lo)}, tick {_d(tick)}. Is there a signal, and what is the stop entry?",
 f"Test three consecutive AO changes in the {direction} direction: {ao}. If valid, use the aligned third-change bar [{_d(lo)}, {_d(hi)}] and one tick {_d(tick)} for the entry.",
 f"Do not infer from AO sign. Direction={direction}; AO sequence={ao}; third-change bar H={_d(hi)}, L={_d(lo)}, tick={_d(tick)}. Determine the WM2 entry."]
 deltas=[values[j+1]-values[j] for j in range(3)]
 calc=f"AO deltas={[ _d(x) for x in deltas ]}. "+(f"Entry={'high + tick' if direction=='LONG' else 'low - tick'}={_d(entry)}." if ok else "At least one change reverses; no entry.")
 return _row(split,"wm2",i,p[t],_answer(answer,RULES["wm2"]["rule_id"],calc,
  "Three consecutive AO columns must change in the trade direction; the stop is one tick beyond the price bar aligned with the third change.",
  "AO being above or below zero is not a substitute for three consecutive rising/falling columns.",
  "None; four closed AO values and the aligned price-bar range are supplied.","Invalidate if any AO or aligned price bar is not closed or the bars are misaligned.",_d(entry) if entry is not None else None),
  {"direction":direction,"ao_values":[_d(x) for x in values],"third_bar_high":_d(hi),"third_bar_low":_d(lo),"tick":_d(tick),"prompt_template_id":t})

def _wm3(rng,split,i,t=0):
 direction="LONG" if i%2==0 else "SHORT";active=(i//2)%2==0
 level=Decimal(rng.randint(12000,22000))/100;tick=Decimal(rng.choice((1,2,5)))/100
 teeth=level-Decimal(rng.randint(20,90))/100 if direction=="LONG" else level+Decimal(rng.randint(20,90))/100
 trigger=level+tick if direction=="LONG" else level-tick
 if not active:teeth=trigger+Decimal("0.20") if direction=="LONG" else trigger-Decimal("0.20")
 confirmed=i%4!=3
 beyond=(trigger>level and trigger>teeth) if direction=="LONG" else (trigger<level and trigger<teeth)
 answer="ACTIVE" if confirmed and beyond else "NOT_ACTIVE";entry=trigger if answer=="ACTIVE" else None
 p=[f"A {direction} fractal at {_d(level)} is {'confirmed' if confirmed else 'not confirmed'}. At the possible trigger, price={_d(trigger)} and current Alligator Teeth={_d(teeth)}. Is WM3 active?",
 f"Evaluate WM3 at trigger, not at fractal formation: direction={direction}; confirmed={confirmed}; fractal level={_d(level)}; trigger price={_d(trigger)}; current Teeth={_d(teeth)}.",
 f"The {direction} fractal level is {_d(level)} and its confirmation state is {confirmed}. Trigger price={_d(trigger)}, current Teeth={_d(teeth)}. Has price cleared both required levels?",
 f"Use current Teeth at activation. Direction={direction}; fractal confirmed={confirmed}; level={_d(level)}; trigger={_d(trigger)}; Teeth now={_d(teeth)}. State whether WM3 is active."]
 calc=f"{'LONG requires price > fractal and price > Teeth' if direction=='LONG' else 'SHORT requires price < fractal and price < Teeth'}; trigger={_d(trigger)}, fractal={_d(level)}, Teeth={_d(teeth)}, confirmed={confirmed}."
 return _row(split,"wm3",i,p[t],_answer(answer,RULES["wm3"]["rule_id"],calc,
  "WM3 requires a confirmed fractal and a trigger beyond both the fractal level and current Teeth in the trade direction.",
  "A fractal formed on the correct side of Teeth does not prove price remains beyond current Teeth at activation.",
  "None; trigger price, fractal level, current Teeth and confirmation state are explicit.","Re-evaluate if the fractal is unconfirmed or current Teeth changes before activation.",_d(entry) if entry is not None else None),
  {"direction":direction,"confirmed":confirmed,"fractal_level":_d(level),"trigger_price":_d(trigger),"current_teeth":_d(teeth),"tick":_d(tick),"prompt_template_id":t})

def _course(rng,split,i,t=0):
 direction="LONG" if i%2==0 else "SHORT";hist=Decimal(rng.randint(-900,900))/100;gap=Decimal(rng.randint(20,250))/100
 signal=hist-gap if direction=="LONG" else hist+gap
 answer="LONG" if hist>signal else "SHORT" if hist<signal else "NO_DIRECTION"
 p=[f"On the closed bar, the 5/34 histogram is {_d(hist)} and its 5-period signal line is {_d(signal)}. What momentum direction does the course rule permit?",
 f"Compare histogram {_d(hist)} with its 5-period signal line {_d(signal)}. Return LONG or SHORT based on relative position, not sign versus zero.",
 f"Completed 5/34/5 MACD values: histogram={_d(hist)}, signal={_d(signal)}. Which side is supported by this momentum test?",
 f"Do not confuse zero-line position with signal-line position. Histogram={_d(hist)}; signal={_d(signal)}. Determine course momentum direction."]
 calc=f"Histogram - signal = {_d(hist)} - {_d(signal)} = {_d(hist-signal)}."
 return _row(split,"course_momentum",i,p[t],_answer(answer,RULES["course_momentum"]["rule_id"],calc,
  "Histogram above its signal line indicates LONG momentum; below indicates SHORT momentum.",
  "A negative histogram can still be above its signal line; sign versus zero is a different comparison.",
  "None; both closed-bar values are provided.","Do not use a forming bar or infer from missing histogram/signal values."),
  {"histogram":_d(hist),"signal_line":_d(signal),"prompt_template_id":t})

GENERATORS={"ao_difference":_ao,"zone":_zone,"fractal":_fractal,"wm1_shape":_wm1_shape,"wm1_stop":_wm1_stop,"wm2":_wm2,"wm3":_wm3,"course_momentum":_course}

def validate_curriculum(rows):
 seen_ids=set();seen_prompts=set()
 for row in rows:
  if row["case_id"] in seen_ids:raise ValueError("DUPLICATE_CORRECTIVE_CASE_ID")
  seen_ids.add(row["case_id"]);ph=hashlib.sha256(row["prompt"].encode()).hexdigest()
  if ph in seen_prompts:raise ValueError("DUPLICATE_CORRECTIVE_PROMPT")
  seen_prompts.add(ph);target=json.loads(row["answer"]);x=row["inputs"];family=row["family"]
  if target["rule_id"]!=row["rule_id"]:raise ValueError("CORRECTIVE_RULE_ID_MISMATCH")
  if family=="ao_difference":expected=_d(Decimal(x["sma5"])-Decimal(x["sma34"]))
  elif family=="zone":
   da=Decimal(x["current_ao"])-Decimal(x["previous_ao"]);dc=Decimal(x["current_ac"])-Decimal(x["previous_ac"])
   expected="GREEN" if da>0 and dc>0 else "RED" if da<0 and dc<0 else "GRAY"
  elif family=="fractal":
   h=[Decimal(v) for v in x["highs"]];l=[Decimal(v) for v in x["lows"]]
   up=h[2]>max(h[:2]+h[3:]);down=l[2]<min(l[:2]+l[3:])
   expected="NONE" if not x["right_bars_closed"] or up==down else "UP" if up else "DOWN"
  elif family=="wm1_shape":
   low=Decimal(x["low"]);high=Decimal(x["high"]);close=Decimal(x["close"]);mid=(low+high)/2;prev=Decimal(x["previous_extreme"])
   valid=(low<prev and close>mid) if x["direction"]=="LONG" else (high>prev and close<mid)
   expected="YES" if valid else "NO"
  elif family=="wm1_stop":expected=_d(Decimal(x["low"])-Decimal(x["tick"]) if x["direction"]=="LONG" else Decimal(x["high"])+Decimal(x["tick"]))
  elif family=="wm2":
   v=[Decimal(z) for z in x["ao_values"]];valid=all(v[j]<v[j+1] for j in range(3)) if x["direction"]=="LONG" else all(v[j]>v[j+1] for j in range(3))
   expected=x["direction"] if valid else "NO_SIGNAL"
   stop=_d(Decimal(x["third_bar_high"])+Decimal(x["tick"]) if x["direction"]=="LONG" else Decimal(x["third_bar_low"])-Decimal(x["tick"])) if valid else None
   if target["entry_stop"]!=stop:raise ValueError("WM2_ENTRY_STOP_ETALON_MISMATCH")
  elif family=="wm3":
   price=Decimal(x["trigger_price"]);level=Decimal(x["fractal_level"]);teeth=Decimal(x["current_teeth"])
   beyond=(price>level and price>teeth) if x["direction"]=="LONG" else (price<level and price<teeth)
   valid=x["confirmed"] and beyond;expected="ACTIVE" if valid else "NOT_ACTIVE"
   if target["entry_stop"]!=(_d(price) if valid else None):raise ValueError("WM3_ENTRY_LEVEL_ETALON_MISMATCH")
  elif family=="course_momentum":
   h=Decimal(x["histogram"]);signal=Decimal(x["signal_line"]);expected="LONG" if h>signal else "SHORT" if h<signal else "NO_DIRECTION"
  else:raise ValueError("UNKNOWN_CORRECTIVE_FAMILY")
  if target["answer"]!=expected:raise ValueError("CORRECTIVE_ANSWER_ETALON_MISMATCH:"+row["case_id"])
  if not all(isinstance(target[k],str) and target[k].strip() for k in ("calculation","reason","counterexample","uncertainty","cancel_if")):
   raise ValueError("CORRECTIVE_EXPLANATION_FIELD_MISSING")
 return {"rows":len(rows),"unique_ids":len(seen_ids),"unique_prompts":len(seen_prompts),"etalon_recomputed":True}

def _template_label(row):
 answer=json.loads(row["answer"])["answer"]
 if row["family"]=="ao_difference":return "POS" if Decimal(answer)>0 else "NEG" if Decimal(answer)<0 else "ZERO"
 if row["family"]=="wm1_stop":return row["inputs"]["direction"]
 return str(answer)

def _template_baseline(rows):
 groups={}
 for row in rows:groups.setdefault(row["inputs"]["prompt_template_id"],[]).append(_template_label(row))
 majority=max(Counter(_template_label(r) for r in rows).values())/len(rows)
 shortcut=sum(max(Counter(v).values()) for v in groups.values())/len(rows)
 return {"global_majority_accuracy":round(majority,4),"template_only_accuracy":round(shortcut,4),
  "labels_by_template":{str(k):dict(Counter(v)) for k,v in sorted(groups.items())}}

def _balanced_template_plan(labels,seed,require_low_baseline):
 count=len(labels);base=(list(range(4))*((count+3)//4))[:count];rng=random.Random(seed)
 majority=max(Counter(labels).values())/count
 for _ in range(5000):
  plan=base[:];rng.shuffle(plan);groups={}
  for tid,label in zip(plan,labels):groups.setdefault(tid,[]).append(label)
  score=sum(max(Counter(v).values()) for v in groups.values())/count
  diverse=all(len(set(v))>=2 for v in groups.values() if len(v)>1)
  if not require_low_baseline or (score<=majority+0.05 and diverse):return plan
 raise ValueError("PROMPT_TEMPLATE_SHORTCUT_ASSIGNMENT_FAILED")

def corrective_curriculum():
 loaded=load_rule_cards();ids=set(loaded.keys()) if isinstance(loaded,dict) else {x["id"] for x in loaded}
 missing=sorted({x["rule_id"] for x in RULES.values()}-ids)
 if missing:raise ValueError("CORRECTIVE_RULE_CARD_MISSING:"+",".join(missing))
 rows=[];template_audit={}
 for si,split in enumerate(SPLIT_ORDER):
  for fi,(family,gen) in enumerate(GENERATORS.items()):
   seed=SEED+si*100003+fi*1009;count=SPLIT_COUNTS[split]
   preview_rng=random.Random(seed)
   preview=[gen(preview_rng,split,i,0) for i in range(count)]
   labels=[_template_label(x) for x in preview]
   tids=_balanced_template_plan(labels,seed+7919,split=="train")
   rng=random.Random(seed);family_rows=[gen(rng,split,i,tids[i]) for i in range(count)]
   rows.extend(family_rows)
   if split=="train":template_audit[family]=_template_baseline(family_rows)
 validate_curriculum(rows)
 split_counts=dict(Counter(x["split"] for x in rows))
 family_counts={s:dict(Counter(x["family"] for x in rows if x["split"]==s)) for s in SPLIT_ORDER}
 canonical=json.dumps(rows,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
 manifest={"schema":"WILLIAMS_AI_ADAPTIVE_CORRECTIVE_CURRICULUM_V1","generator_version":GENERATOR_VERSION,
  "seed":SEED,"dataset_sha256":hashlib.sha256(canonical).hexdigest(),"split_rows":split_counts,
  "split_family_rows":family_counts,"rule_cards":sorted({x["rule_id"] for x in rows}),
  "source_editions":sorted({x["edition"] for x in rows}),"candidate_labels_used":0,"historical_chart_rows":0,
  "locked_holdout_in_training_export":False,"final_exam_included":False,"training_template_shortcut_audit":template_audit,
  "notes":["Original paraphrases; no copied book passages.","Splits are disjoint by case ID and prompt hash.",
   "The prior 32-case objective comparison exam is not imported or reproduced.",
   "These isolate rules; they are not full historical-chart or campaign qualification."]}
 return rows,manifest

def training_rows():
 rows,manifest=corrective_curriculum()
 return [x for x in rows if x["split"] in ("train","validation")],manifest

if __name__=="__main__":
 rows,manifest=corrective_curriculum()
 print(json.dumps({"manifest":manifest,"row_count":len(rows)},ensure_ascii=False,indent=2))
