import json
import unittest
from scripts.ai_export_portable_restorative import portable_config
from williams_ai.deep_book_research import MODEL,REVISION


class PortableRestorativeTests(unittest.TestCase):
    def test_only_base_pointer_changes_and_training_parameters_remain_exact(self):
        original={'peft_type':'LORA','r':16,'lora_alpha':32,'task_type':'CAUSAL_LM','bias':'none',
            'target_modules':['q_proj','v_proj'],'base_model_name_or_path':'/root/.cache/nonportable','revision':None,
            'inference_mode':True,'lora_dropout':0.05}
        raw,delta=portable_config(json.dumps(original).encode());actual=json.loads(raw)
        self.assertEqual(actual.pop('base_model_name_or_path'),MODEL);self.assertEqual(actual.pop('revision'),REVISION)
        self.assertEqual(actual,{k:v for k,v in original.items() if k not in ('base_model_name_or_path','revision')})
        self.assertEqual(set(delta),{'base_model_name_or_path','revision'})
        for bad in ({**original,'r':8},{**original,'peft_type':'PREFIX_TUNING'}):
            with self.assertRaises(ValueError):portable_config(json.dumps(bad).encode())


if __name__=='__main__':unittest.main()
