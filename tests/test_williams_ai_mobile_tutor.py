import unittest
from williams_ai.mobile_tutor import practice_cases,public_source,prompt,parse_answer


class MobileTutorTests(unittest.TestCase):
    def test_four_books_and_keys_never_enter_shared_prompt(self):
        cases=practice_cases()
        self.assertEqual(len(cases),8)
        self.assertEqual({c['edition'] for c in cases},{'TC1','NTD','TC2','COURSE'})
        all_cases=practice_cases(None)
        self.assertEqual(len(all_cases),46)
        self.assertEqual(len({c['id'] for c in all_cases}),46)
        for c in all_cases:
            altered={**c,'correct':'LABEL_LEAK_CANARY','feedback':'FEEDBACK_LEAK_CANARY'}
            self.assertEqual(prompt(c,public_source(c)),prompt(altered,public_source(c)))
            self.assertTrue(prompt(c,None).endswith('/no_think'))

    def test_other_book_or_page_cannot_be_a_source(self):
        c=practice_cases()[0];source=public_source(c)
        for field,value in (('book','TC2'),('page',10000)):
            with self.assertRaises(ValueError):prompt(c,{**source,field:value})

    def test_final_answer_is_typed_and_duplicate_or_unfinished_output_rejected(self):
        self.assertEqual(parse_answer('<think>private</think>{"choice":"B","reason":"source"}')['choice'],'B')
        for raw in ('{"choice":"A","choice":"B","reason":"x"}',
                    '<think>unfinished','{"choice":"A","reason":"x"} trailing',
                    '{"choice":"D","reason":"x"}','{"choice":"A","reason":false}'):
            with self.subTest(raw=raw),self.assertRaises(ValueError):parse_answer(raw)
        with self.assertRaises(ValueError):parse_answer('{"choice":"A","reason":"x"}','length')


if __name__=='__main__': unittest.main()
