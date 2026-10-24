"""Failing source extraction in one edition cannot block another edition."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from williams_ai.book_index import index_pages
from williams_ai.book_rules import registry
from williams_ai.study import study_batch,study_report

class WrongReader:
    def study_passage(self,text,*,edition,page,rule_cards):
        return json.dumps({"summary":"Сфабрикованная теория",
                           "claims":[{"statement":"Нет в первоисточнике",
                                      "support":"invented supporting quote not in book"}],
                           "questions":[]},ensure_ascii=False)
class RightReader:
    def study_passage(self,text,*,edition,page,rule_cards):
        return json.dumps({"summary":"Изучается фрактальная структура двух свечей.",
                           "claims":[{"statement":"Есть исходное утверждение",
                                      "support":text[:70]}],
                           "questions":["Как доказать причинность?"],},ensure_ascii=False)

class NoCrossBookRetryContamination(unittest.TestCase):
    def test_identical_page_chunk_failed_three_times_only_for_one_book(self):
        with TemporaryDirectory() as d:
            db=Path(d)/"sources.db"
            memory=Path(d)/"study.db"
            sentence="Управление риском и сложные вложенные волны подтверждаются свечами. "*2
            for ed in ("TC1","NTD"):
                b=next(x for x in registry()["books"] if x["id"]==ed)
                pages=[""]*b["pages"]
                pages[10]=sentence
                index_pages(db,ed,b["sha256"],pages,
                            declared_pages=b["pages"])
            for _ in range(3):
                result=study_batch(db,memory,"TC1",WrongReader(),
                                   model_id="same-model",budget=1)
                self.assertEqual(result["failed_chunks"],1)
            fourth=study_batch(db,memory,"TC1",WrongReader(),
                               model_id="same-model",budget=1)
            self.assertEqual(fourth["failed_chunks"],0)
            # Same exact hash/text, another edition: must be an independent task!
            second=study_batch(db,memory,"NTD",RightReader(),
                               model_id="same-model",budget=1)
            self.assertEqual(second["studied_chunks"],1)
            report=study_report(db,memory,model_id="same-model")
            cases={b["edition"]:b for b in report["books"]}
            self.assertEqual(cases["NTD"]["studied_chunks"],1)
            self.assertEqual(cases["TC1"]["studied_chunks"],0)
            self.assertEqual(report["chunks_needing_retry_or_review"],1)

if __name__=="__main__":
    unittest.main()
