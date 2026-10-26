"""Offline tests for private original-PDF text corpus: no source books in CI."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from williams_ai.book_rules import registry
from williams_ai.private_book_source import (
    FORMAT, load_private_source_corpus, tokenize_source_pages,
)
from williams_ai.private_book_qlora import train_private_book_qlora

EDITIONS=("TC1", "NTD", "TC2", "COURSE")
SPLITS=("train", "validation", "test")


def corpus(root: Path) -> Path:
    books={x["id"]: x for x in registry()["books"]}
    source=root/"private-four-originals"
    source.mkdir(mode=0o700)
    manifest={
        "format": FORMAT, "training_use":"TEXT_ONLY_CAUSAL_LM_NO_MARKET_HUMAN_LABELS",
        "page_boundary_embargo": 2,"status":"SOURCE_ONLY_NOT_EXPERT_APPROVED",
        "all_figures_understood": False,
        "market_training_or_profitability_verified": False,
        "book_registry":{ed: {"sha256":books[ed]["sha256"],"pages":books[ed]["pages"]}
                         for ed in EDITIONS},
        "splits":{},
    }
    for split,pages in (("train",(1,2)),("validation",(6,7)),("test",(11,12))):
        payload=[]
        for ed in EDITIONS:
            for page in pages:
                payload.append({
                    "edition":ed,"pdf_page":page,
                    "text":"Only artificial placeholders, never real source text. " * 4,
                    "extraction_status":("UNVERIFIED_OCR" if ed=="COURSE"
                                         else "PDF_TEXT_NOT_EXPERT_REVIEWED"),
                    "source_sha256":books[ed]["sha256"]})
        raw=("\n".join(json.dumps(row,ensure_ascii=False) for row in payload)+"\n").encode()
        (source/(split+".jsonl")).write_bytes(raw)
        manifest["splits"][split]={
            "rows":len(payload),
            "books":dict.fromkeys(EDITIONS,2),
            "sha256":hashlib.sha256(raw).hexdigest()}
    (source/"manifest.json").write_text(json.dumps(manifest))
    return source


class FakeTokenizer:
    eos_token_id=2
    def __call__(self,text,add_special_tokens=False):
        return {"input_ids": [(ord(x)%200)+3 for x in text]}


class PrivateFourBookSourceGateTests(unittest.TestCase):
    def test_four_editions_private_page_disjoint_and_test_text_not_leaked(self):
        with TemporaryDirectory() as td:
            directory=corpus(Path(td))
            result=load_private_source_corpus(directory)
            self.assertEqual(len(result["train"]),8)
            self.assertEqual(len(result["validation"]),8)
            self.assertNotIn("test",result)
            self.assertFalse(result["audit"]["test_texts_loaded_into_training_memory"])
            self.assertTrue(result["audit"]["source_only_not_trading_ready"])

    def test_source_tamper_is_rejected(self):
        with TemporaryDirectory() as td:
            directory=corpus(Path(td))
            original=(directory/"validation.jsonl").read_bytes()
            (directory/"validation.jsonl").write_bytes(original+b" ")
            with self.assertRaisesRegex(ValueError,"HASH_MISMATCH"):
                load_private_source_corpus(directory)

    def test_cross_split_neighbor_leakage_even_with_updated_checksum(self):
        with TemporaryDirectory() as td:
            directory=corpus(Path(td))
            path=directory/"validation.jsonl"
            content=path.read_text()
            # PDF page 6 -> 4 is within two pages of train page 2
            content=content.replace('"pdf_page": 6','"pdf_page": 4')
            path.write_text(content)
            manifest=json.loads((directory/"manifest.json").read_text())
            manifest["splits"]["validation"]["sha256"]=hashlib.sha256(
                content.encode()).hexdigest()
            (directory/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,"LEAKAGE"):
                load_private_source_corpus(directory)

    def test_training_tokenization_masks_book_prompt_and_bounds(self):
        text="ABC"*170
        row={"edition":"TC1","pdf_page":5,"text":text}
        packed=tokenize_source_pages([row],FakeTokenizer(),max_tokens=256)
        self.assertGreaterEqual(len(packed),2)
        for record in packed:
            self.assertTrue(any(x==-100 for x in record["labels"]))
            self.assertTrue(any(x!=-100 for x in record["labels"]))
            self.assertEqual(len(record["labels"]),len(record["input_ids"]))
            self.assertLessEqual(len(record["input_ids"]),256)

    def test_gpu_training_requires_explicit_permission_before_loading_weights(self):
        with self.assertRaisesRegex(ValueError,"AUTHORIZATION_REQUIRED"):
            train_private_book_qlora(private_source_dir=Path("/private"),
                hf_base=Path("/qwen"),new_output=Path("/out"),
                authorize_private_gpu=False)

if __name__=="__main__":
    unittest.main()
