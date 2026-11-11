"""Private, opt-in visual understanding of the scanned Profitunity Course.

Unlike text-only GGUF inference, the optional Qwen3-VL backend is given the
ACTUAL page pixels. Outputs are provisional visual interpretations; no model
can certify its own diagrams or generate approved trading-rule labels.

No PDF/page/large OCR text is bundled with the source repository.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

from .book_index import _private_database, _known_book
from .course_visual_private import course_visual_audit
from .reasoning_research_v2 import digest


def prepare_visual_jobs(database, *, pages=None):
    """Bind each vision job to exact original-PDF and page-image SHA256."""
    path=_private_database(database)
    if not path.is_file():
        raise FileNotFoundError("PRIVATE_BOOK_INDEX_NOT_INSTALLED")
    # Global integrity check, including image-only page 56.
    audit=course_visual_audit(path)
    if pages is None:
        pages=list(range(1,123))
    if (not isinstance(pages,list) or not pages or len(pages)>122
            or any(type(p) is not int or not 1<=p<=122 for p in pages)
            or len(set(pages))!=len(pages)):
        raise ValueError("BOUNDED_UNIQUE_COURSE_PAGE_NUMBERS_REQUIRED")
    jobs=[]
    with sqlite3.connect(f"file:{path}?mode=ro",uri=True) as db:
        for page in sorted(pages):
            src=db.execute("SELECT section,page_image_path,page_image_sha256,"
                           "source_sha256,ocr_text,ocr_confidence FROM course_pages "
                           "WHERE pdf_page=?",(page,)).fetchone()
            if not src or src[3]!=audit["source_sha256"]:
                raise ValueError("UNPINNED_COURSE_ORIGINAL")
            section,rel,image_sha,original_sha,ocr,conf=src
            # Pages can contain multiple figures. Use the *full-page* image
            # to avoid falsely isolating a caption as belonging to a crop.
            descriptions=[{"figure_id":fid,"unverified_description":description}
                for fid,description in db.execute(
                "SELECT figure_id,description FROM course_visuals "
                "WHERE pdf_page=? ORDER BY figure_id",(page,))]
            image=path.parent/rel
            if (image.is_symlink() or not image.is_file() or
                    hashlib.sha256(image.read_bytes()).hexdigest()!=image_sha):
                raise ValueError("COURSE_IMAGE_CHANGED_BEFORE_REVIEW")
            jobs.append({
                "schema":"COURSE_PRIVATE_VISUAL_JOB_V3",
                "edition":"COURSE","pdf_page":page,
                "section":section,"source_pdf_sha256":original_sha,
                "page_image_sha256":image_sha,
                "page_image_relative_path":rel,
                "ocr_excerpt_unverified":(ocr or "")[:3500],
                "ocr_confidence_unverified":conf,
                "existing_unverified_diagrams":descriptions,
                "required_checks":["visible_wave_degrees","arrow_directions",
                    "numeric_labels","figure_text_alignment","contradictions",
                    "unknown_or_unreadable_segments"],
                "original_page_image_given_to_vision_backend":False,
                "human_expert_approval":False,
                "usable_as_qlora_gold":False,
            })
    return jobs


def _review_candidate(job,response):
    expected={"page_summary","figure_observations","visible_labels",
              "uncertainties","ocr_conflicts","grounding_notes"}
    if not isinstance(response,dict) or set(response)!=expected:
        raise ValueError("EXACT_VISION_PROPOSAL_FIELDS_REQUIRED")
    for key in expected:
        value=response[key]
        if key=="page_summary":
            if not isinstance(value,str) or not 8<=len(value)<=1200:
                raise ValueError("VISUAL_SUMMARY_REQUIRED")
        else:
            if (not isinstance(value,list) or len(value)>30
                    or any(not isinstance(x,str) or not 1<=len(x)<=480
                           for x in value)):
                raise ValueError("BOUNDED_EVIDENCE_LISTS_REQUIRED")
    return {
        "schema":"COURSE_PRIVATE_VISUAL_CANDIDATE_V3",
        "edition":"COURSE","pdf_page":job["pdf_page"],
        "source_pdf_sha256":job["source_pdf_sha256"],
        "page_image_sha256":job["page_image_sha256"],
        "visual_model_draft":response,
        "source_visual_ocr_conflicts_found":bool(response["ocr_conflicts"]),
        "review_status":"VISION_MODEL_PROPOSAL_NOT_EXPERT_VERIFIED",
        "verified_by_independent_humans":False,
        "usable_as_qlora_gold":False,
        "execution_allowed":False,
    }


def review_visual_job(database,job,vision_backend):
    """A vision backend gets real local image pixels, not a text-only path."""
    if not callable(vision_backend):
        raise ValueError("EXPLICIT_ACTUAL_IMAGE_BACKEND_REQUIRED")
    fresh=prepare_visual_jobs(database,pages=[job.get("pdf_page")])
    if (not isinstance(job,dict) or
            job.get("schema")!="COURSE_PRIVATE_VISUAL_JOB_V3" or
            job.get("source_pdf_sha256")!=fresh[0]["source_pdf_sha256"] or
            job.get("page_image_sha256")!=fresh[0]["page_image_sha256"] or
            job.get("page_image_relative_path")!=fresh[0]["page_image_relative_path"]):
        raise ValueError("STALE_OR_FORGED_VISUAL_REVIEW_JOB")
    image=_private_database(database).parent/job["page_image_relative_path"]
    result=vision_backend(image,job)
    return _review_candidate(job,result)


class LocalQwenVLBackend:
    """Opt-in image-to-text Qwen3-VL; no auto-download, cloud or training.

    Install a supported transformers version and own the local model weights.
    This class is deliberately lazy; CPU research CI never loads a 4B model.
    """
    def __init__(self,model_directory):
        from .book_index import ROOT
        self.path=Path(model_directory).expanduser().resolve()
        if (not self.path.is_dir() or self.path.is_symlink()
                or self.path==ROOT or ROOT in self.path.parents):
            raise ValueError("PRIVATE_LOCAL_HF_VL_MODEL_REQUIRED")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("REAL_GPU_REQUIRED_FOR_4B_VISION_EXPERIMENT")
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.processor=AutoProcessor.from_pretrained(
            str(self.path),local_files_only=True,
            min_pixels=224*224,max_pixels=1024*1024)
        self.model=AutoModelForImageTextToText.from_pretrained(
            str(self.path),local_files_only=True,device_map="auto",
            torch_dtype="auto",trust_remote_code=False)
        self.model.eval()

    def __call__(self,image_path,job):
        from PIL import Image
        import torch
        with Image.open(image_path) as src:
            image=src.convert("RGB")
            # Downscale before creating vision tokens to control T4 VRAM.
            image.thumbnail((1400,1400))
            task=("Read the ACTUAL scanned-page image and return a JSON object "
                  "with EXACTLY six fields: page_summary (string), "
                  "figure_observations (list of strings), visible_labels "
                  "(list), uncertainties (list), ocr_conflicts (list), "
                  "grounding_notes (list). Distinguish nested Elliott degrees, "
                  "letter vs numeral wave labels, price arrows and text layout. "
                  "Describe visible evidence, DO NOT infer profitable trading "
                  "signals or claim independent expert validation. "
                  "Uncertain labels must be explicitly uncertain. "
                  "OCR context is unverified: "+job["ocr_excerpt_unverified"][:1800])
            messages=[{"role":"user","content":[
                {"type":"image","image":image},
                {"type":"text","text":task}]}]
            tensors=self.processor.apply_chat_template(
                messages,tokenize=True,add_generation_prompt=True,
                return_dict=True,return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                ids=self.model.generate(**tensors,max_new_tokens=700,
                                        do_sample=False)
            answer=self.processor.batch_decode(
                ids[:,tensors.input_ids.shape[1]:],
                skip_special_tokens=True)[0].strip()
        # Fail closed on wrappers, free text, or accidental hallucinated schema.
        if answer.startswith("```json"):
            answer=answer.removeprefix("```json").removesuffix("```").strip()
        return json.loads(answer)


def save_private_visual_proposals(database,vision_backend,out_file,*,pages=None):
    target=Path(out_file).expanduser().resolve()
    from .book_index import ROOT
    if (target==ROOT or ROOT in target.parents or target.is_symlink()
            or target.exists() or not target.parent.is_dir()):
        raise ValueError("NEW_PRIVATE_OUTPUT_OUTSIDE_GIT_REQUIRED")
    jobs=prepare_visual_jobs(database,pages=pages)
    results=[review_visual_job(database,job,vision_backend) for job in jobs]
    import os
    fd=os.open(target,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as stream:
            for result in results:
                stream.write(json.dumps(result,ensure_ascii=False,sort_keys=True)+"\n")
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return {"pages_processed":len(results),
            "source_sha256":jobs[0]["source_pdf_sha256"],
            "independent_expert_approval_count":0,
            "status":"VISION_RESEARCH_CANDIDATES_ONLY",
            "execution_allowed":False}
