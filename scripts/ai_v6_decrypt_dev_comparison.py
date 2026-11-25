"""Decrypt sealed V5/V6/BASE development answers in runner memory and encrypt the map."""
from __future__ import annotations
import argparse, base64, hashlib, io, json, os
from collections import Counter
from pathlib import Path
from zipfile import BadZipFile, ZipFile
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from williams_ai.v6_objective_practice import curriculum

# Must match scripts.ai_submit_v5_qlora_kaggle.DOMAIN exactly.
DOMAIN=b"Williams AI V5 four-book QLoRA private artifact v1"
MAGIC=b"WILLIAMSV5A1"
ARTIFACT_RUN_ID=45497821820
ARTIFACT_SHA256="7dd3aad231dee6d5b09927ee7bb324d16f7adcbafdfd76f9df1aa079dfd60228"
PUBLIC_KEY_FINGERPRINT_SHA256="5918e3bf1d07e06f28513ea0ee230b5be0b899e1d6595878449dd05a33934ac5"
ENVELOPE_DOMAIN=b"WILLIAMS_AI_DIAGNOSTIC_MAP_ENVELOPE_V1"
MEMBER_SHA256={
 "aggregate.json":"b4aec9080bd8bdca3aaef754283c20c88f7fbc70a62f2b090fedb473d49e906c",
 "BASE.json.sealed":"854ade5eee25b8090a61fdba5f374a190844943ae1f2239e06033349ee9f62ee",
 "V5.json.sealed":"9e95846d5167ff525ae1ddfab39f603617361ef803681a0bf71403fa42041ac4",
 "A_V6.json.sealed":"c9495e34504c100572b5295e3832a27048c22f01beb6ed64b32fbd6a18a2164f",
}
MODEL_FILES={"BASE":"BASE.json.sealed","V5":"V5.json.sealed","V6":"A_V6.json.sealed"}
EXPECTED_MEMBERS={"aggregate.json",*MODEL_FILES.values()}

def _unique(pairs):
    out={}
    for k,v in pairs:
        if k in out: raise ValueError("DUPLICATE_JSON_KEY")
        out[k]=v
    return out

def _answer(raw):
    try: value=json.loads(raw,object_pairs_hook=_unique)
    except (TypeError,ValueError): return None,False
    if not isinstance(value,dict) or set(value)!={"answer"}: return None,False
    answer=value["answer"]
    if answer is not None and type(answer) not in (str,int,float,bool): return None,False
    return answer,True

def grade(raw,answer,truncated=False):
    if truncated: return "TECHNICAL_TRUNCATION"
    try:
        candidate=json.loads(raw,object_pairs_hook=_unique)
        expected=json.loads(answer)
        if not isinstance(candidate,dict) or set(candidate)!={"answer"}: return "INVALID_FORMAT"
        a,b=candidate["answer"],expected["answer"]
        if type(a) is not type(b): return "INVALID_FORMAT"
        return "CORRECT" if a==b else "INCORRECT"
    except (TypeError,ValueError): return "INVALID_FORMAT"

def decrypt_sealed(sealed,secret):
    if not isinstance(secret,str) or len(secret)<20: raise ValueError("DECRYPT_SECRET_MISSING_OR_INVALID")
    if not isinstance(sealed,bytes) or not sealed.startswith(MAGIC) or len(sealed)<=len(MAGIC)+28:
        raise ValueError("SEALED_RESPONSE_HEADER_INVALID")
    n=len(MAGIC); salt=sealed[n:n+16]; nonce=sealed[n+16:n+28]
    key=HKDF(algorithm=hashes.SHA256(),length=32,salt=salt,info=DOMAIN).derive(secret.encode())
    try: return AESGCM(key).decrypt(nonce,sealed[n+28:],DOMAIN)
    except Exception: raise ValueError("SEALED_RESPONSE_AUTHENTICATION_FAILED") from None

def _counts(records,field):
    return dict(sorted(Counter(r[field] for r in records).items()))

def _group(records,field):
    return {g:_counts([r for r in records if r[field]==g],"grade") for g in sorted({r[field] for r in records})}

def build_error_map(archive_bytes,secret,expected_archive_sha256=None,expected_member_sha256=MEMBER_SHA256):
    archive_sha=hashlib.sha256(archive_bytes).hexdigest()
    if expected_archive_sha256 and archive_sha!=expected_archive_sha256:
        raise ValueError("COMPARISON_ARTIFACT_SHA256_MISMATCH")
    try:
        with ZipFile(io.BytesIO(archive_bytes)) as z:
            names=set(z.namelist())
            if names!=EXPECTED_MEMBERS: raise ValueError("COMPARISON_ARTIFACT_MEMBER_SET_MISMATCH")
            for name in names:
                p=Path(name); info=z.getinfo(name)
                if p.is_absolute() or ".." in p.parts: raise ValueError("UNSAFE_COMPARISON_ARTIFACT_PATH")
                if ((info.external_attr>>16)&0o170000)==0o120000: raise ValueError("SYMLINK_IN_COMPARISON_ARTIFACT")
                if expected_member_sha256 is not None:
                    if hashlib.sha256(z.read(name)).hexdigest()!=expected_member_sha256.get(name):
                        raise ValueError("COMPARISON_ARTIFACT_MEMBER_SHA256_MISMATCH")
            report=json.loads(z.read("aggregate.json"),object_pairs_hook=_unique)
            rows,manifest=curriculum()
            if (report.get("schema")!="WILLIAMS_V6_RECOVERED_GGUF_DEV_COMPARISON_V1" or
                report.get("dataset")!=manifest or
                manifest.get("dataset_sha256")!="2915c75df164b6531471bad56bae389762a04d428a3cfdf67867e5cd80c26c4b" or
                manifest.get("split_rows",{}).get("development")!=32 or
                manifest.get("candidate_labels_used")!=0 or manifest.get("historical_chart_rows")!=0 or
                manifest.get("independent_final_exam") is not False or report.get("new_optimizer_steps")!=0 or
                report.get("independent_final_exam") is not False or report.get("execution_allowed") is not False or
                report.get("qualified") is not False):
                raise ValueError("COMPARISON_ARTIFACT_PROVENANCE_CONTRACT_FAILED")
            dev=[r for r in rows if r["split"]=="development"]
            if len(dev)!=32: raise ValueError("EXPECTED_32_DEVELOPMENT_ROWS")
            expected={r["case_id"]:r for r in dev}; decoded={}
            for model,filename in MODEL_FILES.items():
                records=json.loads(decrypt_sealed(z.read(filename),secret),object_pairs_hook=_unique)
                if not isinstance(records,list) or len(records)!=32: raise ValueError("MODEL_RESPONSE_COUNT_MISMATCH")
                if [r.get("id") for r in records]!=[r["case_id"] for r in dev]:
                    raise ValueError("MODEL_RESPONSE_CASE_ORDER_MISMATCH")
                normalized=[]
                for record in records:
                    row=expected.get(record.get("id"))
                    if row is None or record.get("family")!=row["family"] or record.get("edition")!=row["edition"] or not isinstance(record.get("raw"),str):
                        raise ValueError("MODEL_RESPONSE_CASE_IDENTITY_MISMATCH")
                    recalc=grade(record["raw"],row["answer"],False)
                    stored=record.get("grade")
                    if stored=="TECHNICAL_TRUNCATION": final_grade=stored
                    elif recalc!=stored: raise ValueError("MODEL_RESPONSE_GRADE_RECOMPUTE_MISMATCH")
                    else: final_grade=recalc
                    predicted,valid=_answer(record["raw"])
                    normalized.append({"id":row["case_id"],"family":row["family"],"edition":row["edition"],
                        "expected_answer":json.loads(row["answer"],object_pairs_hook=_unique)["answer"],
                        "predicted_answer":predicted if valid else None,"grade":final_grade,
                        "raw_response_sha256":hashlib.sha256(record["raw"].encode()).hexdigest()})
                decoded[model]=normalized
                recorded=report.get("results",{}).get("A_V6" if model=="V6" else model,{})
                counts=_counts(normalized,"grade"); families=_group(normalized,"family"); editions=_group(normalized,"edition")
                if counts!=recorded.get("counts") or families!=recorded.get("by_family") or editions!=recorded.get("by_edition"):
                    raise ValueError("MODEL_RESPONSE_AGGREGATE_MISMATCH")
            by_id={m:{r["id"]:r for r in rs} for m,rs in decoded.items()}
            cases=[]
            for row in dev:
                cid=row["case_id"]
                models={m:{"predicted_answer":by_id[m][cid]["predicted_answer"],"grade":by_id[m][cid]["grade"],
                           "raw_response_sha256":by_id[m][cid]["raw_response_sha256"]} for m in MODEL_FILES}
                any_error=any(v["grade"]!="CORRECT" for v in models.values())
                invalid=any(v["grade"]=="INVALID_FORMAT" for v in models.values())
                cases.append({"id":cid,"family":row["family"],"edition":row["edition"],
                    "expected_answer":json.loads(row["answer"],object_pairs_hook=_unique)["answer"],
                    "models":models,"cause_assessment":"FORMAT_GENERATION_ERROR_OBSERVED" if invalid else
                    "NOT_INFERABLE_FROM_ANSWER_ONLY_OUTPUTS" if any_error else "NO_ERROR_ON_THIS_CASE"})
            summaries={}
            for model,records in decoded.items():
                counts=_counts(records,"grade")
                summaries[model]={"total":len(records),"counts":counts,"by_family":_group(records,"family"),
                    "by_edition":_group(records,"edition"),"accuracy":round(counts.get("CORRECT",0)/len(records),6)}
            def pair(candidate,baseline):
                improved=[]; regressed=[]; unchanged=[]
                for cid in sorted(by_id["BASE"]):
                    c=by_id[candidate][cid]["grade"]=="CORRECT"; b=by_id[baseline][cid]["grade"]=="CORRECT"
                    (improved if c and not b else regressed if b and not c else unchanged).append(cid)
                return {"improved_cases":improved,"regressed_cases":regressed,"unchanged_correctness_cases":unchanged}
            return {"schema":"WILLIAMS_V6_DECRYPTED_DEVELOPMENT_ERROR_MAP_V1","artifact_run_id":ARTIFACT_RUN_ID,
                "artifact_sha256":archive_sha,"dataset_sha256":manifest["dataset_sha256"],"development_cases":32,
                "models":summaries,"pairwise":{"V5_vs_BASE":pair("V5","BASE"),"V6_vs_BASE":pair("V6","BASE")},
                "cases":cases,"qualification":False,"interpretation_limits":[
                    "32 objective development cases, not an independent final exam.",
                    "Four cases per family; no historical chart episodes.",
                    "Answer-only responses do not expose reasoning; root causes need controlled follow-up probes.",
                    "A correct answer here does not establish book mastery or trading competence."] }
    except BadZipFile: raise ValueError("COMPARISON_ARTIFACT_ZIP_INVALID") from None

def build_error_map_from_directory(directory,secret):
    if directory.is_symlink() or not directory.is_dir(): raise ValueError("COMPARISON_ARTIFACT_DIRECTORY_REQUIRED")
    payloads={}
    for name in EXPECTED_MEMBERS:
        path=directory/name
        if path.is_symlink() or not path.is_file(): raise ValueError("COMPARISON_ARTIFACT_MEMBER_MISSING_OR_LINKED")
        data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=MEMBER_SHA256[name]: raise ValueError("COMPARISON_ARTIFACT_MEMBER_SHA256_MISMATCH")
        payloads[name]=data
    out=io.BytesIO()
    with ZipFile(out,"w") as z:
        for name in sorted(payloads): z.writestr(name,payloads[name])
    return build_error_map(out.getvalue(),secret,expected_member_sha256=MEMBER_SHA256)

def encrypt_error_map(report,public_key_pem,expected_fingerprint=PUBLIC_KEY_FINGERPRINT_SHA256):
    key=serialization.load_pem_public_key(public_key_pem)
    if not isinstance(key,rsa.RSAPublicKey) or key.key_size<3072: raise ValueError("RSA3072_PUBLIC_KEY_REQUIRED")
    der=key.public_bytes(serialization.Encoding.DER,serialization.PublicFormat.SubjectPublicKeyInfo)
    fingerprint=hashlib.sha256(der).hexdigest()
    if expected_fingerprint is not None and fingerprint!=expected_fingerprint:
        raise ValueError("DIAGNOSTIC_PUBLIC_KEY_FINGERPRINT_MISMATCH")
    plain=json.dumps(report,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
    data_key=os.urandom(32); nonce=os.urandom(12)
    ciphertext=AESGCM(data_key).encrypt(nonce,plain,ENVELOPE_DOMAIN)
    wrapped=key.encrypt(data_key,padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),algorithm=hashes.SHA256(),label=ENVELOPE_DOMAIN))
    envelope={"schema":"WILLIAMS_AI_ENCRYPTED_DIAGNOSTIC_ENVELOPE_V1","algorithm":"RSA-OAEP-SHA256+AES-256-GCM",
        "public_key_fingerprint_sha256":fingerprint,"nonce_b64":base64.b64encode(nonce).decode(),
        "wrapped_key_b64":base64.b64encode(wrapped).decode(),"ciphertext_b64":base64.b64encode(ciphertext).decode(),
        "plaintext_sha256":hashlib.sha256(plain).hexdigest()}
    return json.dumps(envelope,sort_keys=True,separators=(",",":")).encode()

def main():
    p=argparse.ArgumentParser()
    src=p.add_mutually_exclusive_group(required=True)
    src.add_argument("--archive",type=Path); src.add_argument("--directory",type=Path)
    p.add_argument("--secret-env",default="KAGGLE_API_TOKEN"); p.add_argument("--expected-sha256",default=ARTIFACT_SHA256)
    p.add_argument("--public-key",required=True,type=Path); p.add_argument("--encrypted-output",required=True,type=Path)
    a=p.parse_args(); secret=os.environ.pop(a.secret_env,"")
    try:
        report=build_error_map_from_directory(a.directory,secret) if a.directory else build_error_map(a.archive.read_bytes(),secret,a.expected_sha256)
        if a.public_key.is_symlink() or not a.public_key.is_file(): raise ValueError("DIAGNOSTIC_PUBLIC_KEY_FILE_REQUIRED")
        encrypted=encrypt_error_map(report,a.public_key.read_bytes())
        a.encrypted_output.parent.mkdir(parents=True,exist_ok=True)
        if a.encrypted_output.exists() or a.encrypted_output.is_symlink(): raise ValueError("NEW_ENCRYPTED_OUTPUT_PATH_REQUIRED")
        a.encrypted_output.write_bytes(encrypted)
        summary={"cases":32,"models":{m:v["counts"] for m,v in report["models"].items()},
            "encrypted_sha256":hashlib.sha256(encrypted).hexdigest(),
            "public_key_fingerprint_sha256":PUBLIC_KEY_FINGERPRINT_SHA256,"plaintext_emitted":False}
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print("WILLIAMS_V6_DECRYPT_DIAGNOSTIC_FAILED",str(exc),flush=True); return 2
    print("WILLIAMS_V6_DIAGNOSTIC_ENCRYPTED "+json.dumps(summary,sort_keys=True),flush=True)
    return 0

if __name__=="__main__": raise SystemExit(main())
