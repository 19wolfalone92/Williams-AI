"""v7 experimental constrained HF JSON decoder, no training, no trading.

A DIFFERENT decoding protocol from frozen v6. Format compliance alone does
not prove Bill Williams reasoning correctness or trade profitability.
"""
from __future__ import annotations
from hashlib import sha256
from .independent_exam_v6 import FAMILIES
from .local_model import _strict_chart_json

SCHEMA="WILLIAMS_V7_GUIDED_JSON_RESEARCH_NO_BOOK_GOLD_V1"
ENFORCER_VERSION="0.11.3"
MAX_GENERATED_TOKENS=384
MAX_INPUT_TOKENS=3200
MAX_REASON_CHARS=220

def require(ok,why):
    if not ok:
        raise ValueError("V7_DECODER_"+why)

def choices(family):
    require(family in FAMILIES,"UNKNOWN_FAMILY")
    return list(FAMILIES[family])+["ABSTAIN"]

def schema(family):
    return {
        "type":"object",
        "properties":{
            "decision":{"type":"string","enum":choices(family)},
            "reason":{"type":"string","minLength":1,"maxLength":MAX_REASON_CHARS}
        },
        "required":["decision","reason"],"additionalProperties":False
    }

def decode(raw, family, *, eos_seen=True, at_limit=False):
    """Never turn transport/model output errors into a genuine ABSTAIN."""
    require(family in FAMILIES,"UNKNOWN_FAMILY")
    bad={
        "decision":None,"model_reason":None,"finish_reason":"error",
        "failure_code":None,"valid_model_decision":False,
        "trade_action":None,"book_gold":None,
    }
    if at_limit:
        return {**bad,"finish_reason":"length","failure_code":"TRUNCATED"}
    if not eos_seen:
        return {**bad,"failure_code":"MISSING_EOS"}
    try:
        result=_strict_chart_json(raw)
    except (ValueError, TypeError):
        result=None
    if not isinstance(result,dict):
        reason="INVALID_JSON"
    elif set(result)!={"decision","reason"}:
        reason="INVALID_FIELDS"
    elif result.get("decision") not in choices(family):
        reason="INVALID_CATEGORY"
    elif type(result.get("reason")) is not str or not 1<=len(result["reason"])<=MAX_REASON_CHARS:
        reason="INVALID_REASON"
    else:
        return {**bad,"decision":result["decision"],"model_reason":result["reason"],
                "finish_reason":"stop","failure_code":None,"valid_model_decision":True}
    return {**bad,"failure_code":reason}

def prefix_fn(tokenizer,family):
    try:
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn)
    except ImportError as exc:
        raise RuntimeError("V7_LM_FORMAT_ENFORCER_NOT_INSTALLED") from exc
    return build_transformers_prefix_allowed_tokens_fn(
        tokenizer,JsonSchemaParser(schema(family)))

def infer_one_hf(model,tokenizer,messages,family,*,torch,max_new_tokens=MAX_GENERATED_TOKENS):
    """One real generation, deterministic; no repairs, retries, or orders."""
    require(type(max_new_tokens) is int and
            64<=max_new_tokens<=MAX_GENERATED_TOKENS,"INVALID_GENERATION_LIMIT")
    require(type(messages) is list and len(messages)==2 and
            all(type(x) is dict and set(x)=={"role","content"} for x in messages),
            "INVALID_MESSAGES")
    ids=tokenizer.apply_chat_template(messages,tokenize=True,
        add_generation_prompt=True,enable_thinking=False,return_tensors="pt")
    size=ids.shape[-1]
    require(size<=MAX_INPUT_TOKENS,"PROMPT_TOO_LONG_FOR_GPU")
    ids=ids.to("cuda:0")
    filter_fn=prefix_fn(tokenizer,family)
    try:
        with torch.inference_mode():
            response=model.generate(input_ids=ids,
                attention_mask=torch.ones_like(ids),
                prefix_allowed_tokens_fn=filter_fn,do_sample=False,
                max_new_tokens=max_new_tokens,use_cache=True,
                pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
    except Exception as exc:
        if "out of memory" in str(exc).lower():
            raise
        return {"decision":None,"model_reason":None,
                "finish_reason":"error","failure_code":"INFERENCE_EXCEPTION",
                "valid_model_decision":False,"trade_action":None,
                "book_gold":None,"tokens_in":size,"tokens_out":0}
    output=response[0][size:]
    n=len(output)
    eos=bool(n and output[-1].item()==tokenizer.eos_token_id)
    raw=tokenizer.decode(output,skip_special_tokens=True).strip()
    obj=decode(raw,family,eos_seen=eos,at_limit=n>=max_new_tokens and not eos)
    return {**obj,"tokens_in":size,"tokens_out":n,
        "output_sha256":sha256(raw.encode("utf-8")).hexdigest(),
        "schema":SCHEMA,"json_generation_constrained":True,
        "weights_changed":False}
