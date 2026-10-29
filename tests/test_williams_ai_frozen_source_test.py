import base64
import copy
from hashlib import sha256
import json
import unittest

from williams_ai.book_rules import registry
from williams_ai.deep_book_research import EDITIONS,checked_payload
from williams_ai.frozen_source_test import FORMAT,manifest_digest,checked_test_payload,recompute_source_scores,grade_source_test


def fixture():
    books={b['id']:b for b in registry()['books']};counts=dict(zip(EDITIONS,(19,22,33,22)));rows=[]
    for e,n in counts.items():
        for p in range(10,10+n):
            rows.append({'edition':e,'pdf_page':p,'text':f'Synthetic nonlicensed source-test fixture only, {e}, page {p}.',
                'source_sha256':books[e]['sha256'],'extraction_status':'UNVERIFIED_OCR' if e=='COURSE' else 'PDF_TEXT_NOT_EXPERT_REVIEWED'})
    raw=b'\n'.join(json.dumps(r).encode() for r in rows)+b'\n'
    m={'book_registry':{e:{'sha256':b['sha256'],'pages':b['pages']} for e,b in books.items()},'page_boundary_embargo':2,
        'status':'SOURCE_ONLY_NOT_EXPERT_APPROVED','all_figures_understood':False,'market_training_or_profitability_verified':False,
        'splits':{'train':{'books':dict.fromkeys(EDITIONS,1)},'validation':{'books':dict.fromkeys(EDITIONS,1)},
            'test':{'books':counts,'rows':96,'sha256':sha256(raw).hexdigest()}}}
    payload={'format':FORMAT,'manifest':m,'test_file_base64':base64.b64encode(raw).decode(),
        'training_page_ids':{e:[1] for e in EDITIONS},'validation_page_ids':{e:[5] for e in EDITIONS},
        'training_allowed':False,'candidate_selection_allowed':False,'execution_allowed':False}
    return payload,manifest_digest(m)


class FrozenSourceTestTests(unittest.TestCase):
    def test_test_only_schema_cannot_enter_training_and_has_exact_96_original_identities(self):
        p,d=fixture();rows,a=checked_test_payload(p,expected_manifest_sha256=d)
        self.assertEqual(len(rows),96);self.assertTrue(a['test_text_was_excluded_from_training']);self.assertFalse(a['training_performed_here'])
        with self.assertRaises(ValueError):checked_payload(p)
        with self.assertRaises(ValueError):checked_test_payload(p,expected_manifest_sha256='0'*64)

    def test_training_permission_source_hash_or_two_page_overlap_is_rejected(self):
        p,d=fixture()
        for field in ('training_allowed','candidate_selection_allowed','execution_allowed'):
            bad=copy.deepcopy(p);bad[field]=True
            with self.assertRaises(ValueError):checked_test_payload(bad,expected_manifest_sha256=d)
        bad=copy.deepcopy(p);bad['training_page_ids']['TC1']=[10]
        with self.assertRaisesRegex(ValueError,'EMBARGO'):checked_test_payload(bad,expected_manifest_sha256=d)
        bad=copy.deepcopy(p);bad['test_file_base64']=base64.b64encode(base64.b64decode(p['test_file_base64'])+b' ').decode()
        with self.assertRaisesRegex(ValueError,'HASH'):checked_test_payload(bad,expected_manifest_sha256=d)

    def test_every_model_and_window_is_required_and_aggregates_recompute(self):
        counts=dict.fromkeys(EDITIONS,2);rows=[]
        for stage,value in (('base',3.),('previous40',2.),('focused69',1.9),('candidate',1.7)):
            for e in EDITIONS:
                for i in range(2):rows.append((stage,e,i,value))
        scores=recompute_source_scores(rows,counts)
        self.assertAlmostEqual(scores['candidate']['ALL'],1.7)
        self.assertTrue(grade_source_test(scores)['source_test_gate_passed'])
        for bad in (rows[:-1],rows+[rows[0]],[*rows[:-1],('candidate','COURSE',1,float('nan'))]):
            with self.assertRaises(ValueError):recompute_source_scores(bad,counts)

    def test_source_gate_still_rejects_any_book_regression(self):
        scores={s:dict.fromkeys((*EDITIONS,'ALL'),2.) for s in ('base','previous40','focused69','candidate')}
        scores['candidate']={**scores['candidate'],'ALL':1.7,'NTD':2.041}
        result=grade_source_test(scores)
        self.assertFalse(result['source_test_gate_passed']);self.assertFalse(result['trading_qualification']);self.assertFalse(result['execution_allowed'])


if __name__=='__main__':unittest.main()
