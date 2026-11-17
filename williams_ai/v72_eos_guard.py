"""v7.2 research-only EOS guard: deny early Qwen EOS until strict JSON is complete.

This is a decoding repair, NOT model knowledge improvement or training.
Do not use with broker orders; all observed cases are historical and gold-free.
"""
from __future__ import annotations
from hashlib import sha256
from .v71_guided_json_diagnostics import (
    MAX_GENERATED_TOKENS,MAX_INPUT_TOKENS,ENFORCER_COMMIT,
    choices,decode,schema,
)

SCHEMA="WILLIAMS_V72_EOS_GATED_JSON_NO_GOLD_V1"

def require(ok,why):
    if not ok:
        raise ValueError("V72_EOS_GUARD_"+why)

def filter_eos_until_json_complete(allowed,eos_token_id,generated_text,family):
    """Pure testable barrier against the 10 early-EOS failures in v7.1.

    LMFE may report an allowed EOS while still inside a JSON string.  A
    syntactically *and* schema-invalid prefix cannot terminate successfully.
    """
    require(type(allowed) in (list,tuple) and
            all(type(i) is int and i >= 0 for i in allowed),"BAD_ALLOWED")
    eoses=(eos_token_id,) if type(eos_token_id) is int else eos_token_id
    require(type(eoses) in (list,tuple) and eoses and
            all(type(i) is int and i>=0 for i in eoses),"BAD_EOS_IDS")
    require(type(generated_text) is str,"BAD_TEXT")
    eos_set=set(eoses)
    if not any(t in eos_set for t in allowed):
        return list(allowed),False
    validity=decode(generated_text.strip(),family,eos_seen=True)
    if validity["valid_model_decision"]:
        return list(allowed),False
    safe=[t for t in allowed if t not in eos_set]
    require(bool(safe),"ONLY_PREMATURE_EOS_ALLOWED")
    return safe,True

def guarded_enforcer(tokenizer,family,prompt_len):
    require(type(prompt_len) is int and prompt_len>0,"BAD_PROMPT_LEN")
    try:
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    except ImportError as exc:
        raise RuntimeError("V72_PINNED_ENFORCER_MISSING") from exc
    fn=build_transformers_prefix_allowed_tokens_fn(
        tokenizer,JsonSchemaParser(schema(family)))
    metrics={"callbacks":0,"eos_blocked":0,"max_allowed_count":0}
    def tracked(batch_id,input_ids):
        allowed=fn(batch_id,input_ids)
        # Decode only the generated suffix, never the system prompt or
        # full history.  Does not use or expose future candles/gold.
        suffix=input_ids[prompt_len:]
        text=tokenizer.decode(suffix,skip_special_tokens=True,
                              clean_up_tokenization_spaces=False)
        checked,blocked=filter_eos_until_json_complete(
            list(allowed),tokenizer.eos_token_id,text,family)
        metrics["callbacks"]+=1
        metrics["eos_blocked"]+=int(blocked)
        metrics["max_allowed_count"]=max(metrics["max_allowed_count"],len(checked))
        return checked
    return tracked,metrics

def generate(model,tokenizer,messages,family,*,torch,max_new_tokens=MAX_GENERATED_TOKENS):
    require(type(max_new_tokens) is int and
            64<=max_new_tokens<=MAX_GENERATED_TOKENS,"BAD_MAX_TOKENS")
    require(type(messages) is list and len(messages)==2 and
            all(type(x) is dict and set(x)=={"role","content"} for x in messages),
            "BAD_MESSAGES")
    ids=tokenizer.apply_chat_template(messages,tokenize=True,
        add_generation_prompt=True,enable_thinking=False,return_tensors="pt")
    input_count=int(ids.shape[-1])
    require(input_count<=MAX_INPUT_TOKENS,"PROMPT_TOO_LONG")
    ids=ids.to("cuda:0")
    prefix,stats=guarded_enforcer(tokenizer,family,input_count)
    try:
        with torch.inference_mode():
            output=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
                prefix_allowed_tokens_fn=prefix,do_sample=False,
                max_new_tokens=max_new_tokens,use_cache=True,
                pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
    except Exception as exc:
        if "out of memory" in str(exc).lower():
            raise
        return {"schema":SCHEMA,"decision":None,"model_reason":None,
            "finish_reason":"error","failure_code":"INFERENCE_EXCEPTION",
            "valid_model_decision":False,"trade_action":None,"book_gold":None,
            "tokens_in":input_count,"tokens_out":0,
            "eos_guard_activated":True,"metrics":stats,"weights_changed":False}
    tokens=output[0][input_count:]
    out_ids=[int(t) for t in tokens.tolist()]
    ids_eos=tokenizer.eos_token_id
    eoses=([ids_eos] if type(ids_eos) is int else list(ids_eos))
    eos_seen=bool(out_ids and out_ids[-1] in eoses)
    raw=tokenizer.decode(tokens,skip_special_tokens=True,
                         clean_up_tokenization_spaces=False).strip()
    result=decode(raw,family,eos_seen=eos_seen,
                  at_limit=len(out_ids)>=max_new_tokens and not eos_seen)
    return {**result,"schema":SCHEMA,"tokens_in":input_count,
        "tokens_out":len(out_ids),"eos_seen":eos_seen,
        "raw_output_sha256":sha256(raw.encode("utf-8")).hexdigest(),
        "eos_guard_activated":True,"metrics":stats,
        "enforcer_commit":ENFORCER_COMMIT,
        "weights_changed":False,"trade_action":None,"book_gold":None}
