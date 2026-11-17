"""v7.1 guided JSON with pinned LMFE incremental-decoding bugfix and diagnostics."""
from __future__ import annotations
from hashlib import sha256
from .independent_exam_v6 import FAMILIES
from .local_model import _strict_chart_json

SCHEMA="WILLIAMS_V71_GUIDED_JSON_TOKEN_DIAGNOSTICS_NO_GOLD_V1"
ENFORCER_COMMIT="d4d9b7cc20937bcd3a643be7a8dafb7b1ca20dfb"
MAX_GENERATED_TOKENS=384
MAX_INPUT_TOKENS=3200
MAX_REASON_CHARS=220

def require(ok,why):
    if not ok: raise ValueError("V71_DECODER_"+why)

def choices(family):
    require(family in FAMILIES,"UNKNOWN_FAMILY")
    return list(FAMILIES[family])+["ABSTAIN"]

def schema(family):
    return {"type":"object","properties":{
        "decision":{"type":"string","enum":choices(family)},
        "reason":{"type":"string","minLength":1,"maxLength":MAX_REASON_CHARS}},
        "required":["decision","reason"],"additionalProperties":False}

def decode(raw,family,*,eos_seen=True,at_limit=False):
    fail={"decision":None,"model_reason":None,"finish_reason":"error",
        "failure_code":None,"valid_model_decision":False,"trade_action":None,"book_gold":None}
    if at_limit:return {**fail,"finish_reason":"length","failure_code":"TRUNCATED"}
    if not eos_seen:return {**fail,"failure_code":"MISSING_EOS"}
    try:obj=_strict_chart_json(raw)
    except (ValueError,TypeError):obj=None
    if not isinstance(obj,dict):why="INVALID_JSON"
    elif set(obj)!={"decision","reason"}:why="INVALID_FIELDS"
    elif obj.get("decision") not in choices(family):why="INVALID_CATEGORY"
    elif type(obj.get("reason")) is not str or not 1<=len(obj["reason"])<=MAX_REASON_CHARS:why="INVALID_REASON"
    else:return {**fail,"decision":obj["decision"],"model_reason":obj["reason"],
        "finish_reason":"stop","failure_code":None,"valid_model_decision":True}
    return {**fail,"failure_code":why}

def enforcer_with_trace(tokenizer,family):
    try:
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    except ImportError as exc:raise RuntimeError("V71_PINNED_ENFORCER_MISSING") from exc
    fn=build_transformers_prefix_allowed_tokens_fn(tokenizer,JsonSchemaParser(schema(family)))
    trace=[]
    def tracked(batch_id,input_ids):
        allowed=fn(batch_id,input_ids)
        if len(trace)<48:
            trace.append({"step":len(trace),"batch_id":int(batch_id),
                "allowed_count":len(allowed),"allowed_sample_ids":list(allowed[:12])})
        return allowed
    return tracked,trace

def generate(model,tokenizer,messages,family,*,torch,max_new_tokens=MAX_GENERATED_TOKENS):
    require(type(max_new_tokens) is int and 64<=max_new_tokens<=MAX_GENERATED_TOKENS,"BAD_MAX_TOKENS")
    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
        enable_thinking=False,return_tensors="pt")
    n_input=int(ids.shape[-1]);require(n_input<=MAX_INPUT_TOKENS,"PROMPT_TOO_LONG")
    ids=ids.to("cuda:0");prefix,trace=enforcer_with_trace(tokenizer,family)
    try:
        with torch.inference_mode():
            out=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
                prefix_allowed_tokens_fn=prefix,do_sample=False,max_new_tokens=max_new_tokens,
                use_cache=True,pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
    except Exception as exc:
        if "out of memory" in str(exc).lower():raise
        return {"decision":None,"model_reason":None,"finish_reason":"error",
            "failure_code":"INFERENCE_EXCEPTION","valid_model_decision":False,
            "trade_action":None,"book_gold":None,"tokens_in":n_input,"tokens_out":0,
            "enforcer_trace":trace}
    gen=out[0][n_input:];ids_out=[int(t) for t in gen.tolist()]
    eos_id=tokenizer.eos_token_id
    eos_seen=(ids_out[-1]==eos_id) if isinstance(eos_id,int) and ids_out else (
        ids_out[-1] in eos_id if isinstance(eos_id,(list,tuple)) and ids_out else False)
    raw=tokenizer.decode(gen,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
    raw_all=tokenizer.decode(gen,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    result=decode(raw,family,eos_seen=eos_seen,at_limit=len(ids_out)>=max_new_tokens and not eos_seen)
    debug={"raw_without_special_tokens":raw[:1200],"raw_with_special_tokens":raw_all[:1600],
        "token_ids_first_48":ids_out[:48],"token_pieces_first_48":tokenizer.convert_ids_to_tokens(ids_out[:48]),
        "eos_token_id":eos_id,"eos_seen":bool(eos_seen),"prefix_callback_calls":len(trace),
        "allowed_counts_first_12":[x["allowed_count"] for x in trace[:12]],
        "first_allowed_ids":trace[0]["allowed_sample_ids"] if trace else []}
    return {**result,"tokens_in":n_input,"tokens_out":len(ids_out),
        "raw_output_sha256":sha256(raw.encode()).hexdigest(),"diagnostic":debug,
        "schema":SCHEMA,"enforcer_commit":ENFORCER_COMMIT,"weights_changed":False,
        "trade_action":None,"book_gold":None}
