"""Integrity/rollback regressions. All bytes and receipts below are synthetic."""
from copy import deepcopy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from williams_ai.lora_integrity import local_hf_identity, validation_admission
from williams_ai.lora_training import LoraPlan
from williams_ai.model_bundle import (CONVERTER_COMMIT, _convertible_config,
    export_runtime_lora, file_sha256, _exportable_training_receipt)
from williams_ai.private_lora import checked_private_training_data, train_private_lora
from williams_ai.local_model import LocalWilliamsModel
from test_williams_ai_private_lora import candidates
from test_williams_ai_unified_runtime import fake_bundle


def hf_fixture(root):
    root.mkdir()
    (root/'config.json').write_text(json.dumps({'model_type':'qwen3'}))
    (root/'tokenizer_config.json').write_text('{}')
    (root/'tokenizer.json').write_text('{}')
    (root/'model.safetensors').write_bytes(b'NOT REAL WEIGHTS; FILE IDENTITY FIXTURE')
    return root


def export_fixture(root):
    base=hf_fixture(root/'hf')
    trained=root/'trained';trained.mkdir()
    (trained/'adapter_model.safetensors').write_bytes(b'NOT REAL ADAPTER WEIGHTS')
    config={'peft_type':'LORA','task_type':'CAUSAL_LM','r':8,'lora_alpha':16,'bias':'none'}
    (trained/'adapter_config.json').write_text(json.dumps(config))
    before={ed:2.0 for ed in ('ALL','TC1','NTD','TC2','COURSE')}
    after={ed:1.5 for ed in before}
    identity=local_hf_identity(base)
    receipt={'mode':'LOCAL_RESEARCH_LORA_NOT_TRADING_READY','training_receipt_version':2,
        'optimizer_steps':10,'selected_checkpoint_step':5,
        'source_dataset_sha256':'c'*64,'local_base_config':identity['config'],
        'local_base_identity':identity,'source_only_research':True,
        'adapter_weights_sha256':file_sha256(trained/'adapter_model.safetensors'),
        'adapter_config_sha256':file_sha256(trained/'adapter_config.json'),
        'baseline_validation_loss':2.0,'trained_validation_loss':1.5,
        'baseline_by_book':before,'adapted_by_book':after,
        'validation_gate':validation_admission(before,after),
        'best_validation_checkpoint_reloaded':True,'adapter_weights_changed':True,
        'base_parameters_frozen':True,'test_labels_not_supplied_to_trainer':True,
        'execution_allowed':False,'trade_action':None}
    (trained/'research_manifest.json').write_text(json.dumps(receipt))
    gguf=root/'base.gguf';gguf.write_bytes(b'GGUF NOT REAL MODEL')
    checkout=root/'converter';checkout.mkdir()
    (checkout/'convert_lora_to_gguf.py').write_text('# injected test converter')
    return base,trained,gguf,checkout,receipt


class LoraIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_same_configuration_cannot_hide_changed_base_weights_or_tokenizer(self):
        base=hf_fixture(self.root/'hf')
        first=local_hf_identity(base)
        (base/'model.safetensors').write_bytes(b'DIFFERENT WEIGHTS SAME CONFIG')
        second=local_hf_identity(base)
        self.assertEqual(first['config'],second['config'])
        self.assertNotEqual(first['sha256'],second['sha256'])
        (base/'tokenizer.json').write_text('{"different":"token IDs"}')
        self.assertNotEqual(second['sha256'],local_hf_identity(base)['sha256'])

    def test_every_shard_is_bound_and_missing_extra_or_traversal_shards_rejected(self):
        base=hf_fixture(self.root/'hf');(base/'model.safetensors').unlink()
        index=base/'model.safetensors.index.json'
        shard=base/'model-00001-of-00001.safetensors';shard.write_bytes(b'SHARD')
        index.write_text(json.dumps({'weight_map':{'layer.weight':shard.name}}))
        first=local_hf_identity(base)
        shard.write_bytes(b'CHANGED SHARD')
        self.assertNotEqual(first['sha256'],local_hf_identity(base)['sha256'])
        extra=base/'model-00002-of-00002.safetensors';extra.write_bytes(b'EXTRA')
        with self.assertRaisesRegex(ValueError,'UNINDEXED'):local_hf_identity(base)
        extra.unlink();shard.unlink()
        with self.assertRaisesRegex(ValueError,'UNINDEXED'):local_hf_identity(base)
        index.write_text(json.dumps({'weight_map':{'layer.weight':'../outside.safetensors'}}))
        with self.assertRaisesRegex(ValueError,'SHARD_MAP'):local_hf_identity(base)

    def test_pickle_only_quantized_or_symlinked_model_cannot_enter_fit(self):
        base=hf_fixture(self.root/'hf')
        (base/'model.safetensors').rename(base/'pytorch_model.bin')
        with self.assertRaisesRegex(ValueError,'SAFETENSORS_LAYOUT'):local_hf_identity(base)
        (base/'pytorch_model.bin').rename(base/'model.safetensors')
        (base/'config.json').write_text(json.dumps({'model_type':'qwen3','quantization_config':{}}))
        with self.assertRaisesRegex(ValueError,'UNQUANTIZED'):local_hf_identity(base)
        (base/'config.json').write_text('{"model_type":"qwen3"}')
        (base/'model.safetensors').rename(base/'weights')
        (base/'model.safetensors').symlink_to(base/'weights')
        with self.assertRaisesRegex(ValueError,'REGULAR'):local_hf_identity(base)

    def test_one_book_regression_cannot_hide_inside_better_overall_loss(self):
        before={ed:2.0 for ed in ('ALL','TC1','NTD','TC2','COURSE')}
        after={ed:1.5 for ed in before};after['COURSE']=2.1
        gate=validation_admission(before,after)
        self.assertFalse(gate['accepted']);self.assertEqual(gate['regressed_books'],['COURSE'])
        after['COURSE']=2.0
        self.assertTrue(validation_admission(before,after)['accepted'])
        after={**before,'ALL':1.999}
        self.assertFalse(validation_admission(before,after)['accepted'])

    def test_nan_missing_editions_or_boolean_losses_cannot_be_accepted(self):
        good={ed:2.0 for ed in ('ALL','TC1','NTD','TC2','COURSE')}
        for bad in ({**good,'ALL':float('nan')},{**good,'TC1':True},
                    {k:v for k,v in good.items() if k!='NTD'}):
            with self.subTest(bad=bad),self.assertRaises(ValueError):validation_admission(good,bad)

    def test_invalid_plans_and_missing_k_projection_are_not_silent_defaults(self):
        self.assertIn('k_proj',LoraPlan().targets)
        self.assertIn('down_proj',LoraPlan(target_profile='attention-mlp').targets)
        for kwargs in ({'rank':True},{'rank':256},{'learning_rate':float('nan')},
                       {'gradient_accumulation':0},{'target_profile':'unknown'},{'seed':-1}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):LoraPlan(**kwargs)

    def test_wrong_json_types_and_duplicate_fields_are_rejected_before_optional_runtime(self):
        path=self.root/'data.jsonl'
        for key,value in (('rule_id',[]),('edition',{}),('split',[]),('case_id',None)):
            rows=candidates();rows[0][key]=value
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'FIELD_TYPES'):
                checked_private_training_data(path)
        first=json.dumps(candidates()[0])[:-1]+',"split":"test"}'
        path.write_text(first)
        with self.assertRaisesRegex(ValueError,'INVALID_JSONL'):checked_private_training_data(path)

    def test_unsupported_scaling_dora_and_saved_layers_cannot_be_misconverted(self):
        good={'peft_type':'LORA','task_type':'CAUSAL_LM','r':16,'lora_alpha':32,'bias':'none'}
        _convertible_config(good)
        for key,value in (('use_rslora',True),('use_dora',True),('alpha_pattern',{'x':8}),
                          ('modules_to_save',['lm_head']),('bias','all'),('lora_alpha',float('nan'))):
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'NOT_SUPPORTED'):
                _convertible_config({**good,key:value})

    def test_export_refuses_legacy_receipt_or_false_per_book_gate(self):
        _,_,_,_,receipt=export_fixture(self.root)
        _exportable_training_receipt(receipt)
        old=deepcopy(receipt);old.pop('local_base_identity')
        with self.assertRaisesRegex(ValueError,'V2_REQUIRED'):_exportable_training_receipt(old)
        bad=deepcopy(receipt);bad['adapted_by_book']['NTD']=2.5
        with self.assertRaisesRegex(ValueError,'GATE_NOT_PASSED'):_exportable_training_receipt(bad)

    def test_changed_hf_weights_stop_export_before_converter_is_executed(self):
        base,trained,gguf,checkout,_=export_fixture(self.root)
        (base/'model.safetensors').write_bytes(b'CHANGED BASE SAME CONFIG')
        with patch('williams_ai.model_bundle.subprocess.run') as run:
            with self.assertRaisesRegex(ValueError,'BASE_WEIGHTS_OR_TOKENIZER_MISMATCH'):
                export_runtime_lora(trained_adapter=trained,hf_base=base,base_gguf=gguf,
                    llama_cpp=checkout,output=self.root/'output')
            run.assert_not_called()

    def test_failed_or_mutated_conversion_never_publishes_a_runtime_manifest(self):
        base,trained,gguf,checkout,_=export_fixture(self.root)
        output=self.root/'output'
        for failure in ('exit','receipt','weights'):
            original=(trained/'research_manifest.json').read_bytes()
            original_weights=(base/'model.safetensors').read_bytes()
            def run(args,**kwargs):
                if args[0]=='git':
                    return subprocess.CompletedProcess(args,0,CONVERTER_COMMIT+'\n' if 'rev-parse' in args else '')
                Path(args[args.index('--outfile')+1]).write_bytes(b'GGUF SYNTHETIC OUTPUT')
                if failure=='receipt':(trained/'research_manifest.json').write_bytes(original+b' ')
                if failure=='weights':(base/'model.safetensors').write_bytes(b'CHANGED DURING CONVERT')
                return subprocess.CompletedProcess(args,1 if failure=='exit' else 0)
            with self.subTest(failure=failure),patch('williams_ai.model_bundle.subprocess.run',side_effect=run):
                with self.assertRaises((ValueError,RuntimeError)):
                    export_runtime_lora(trained_adapter=trained,hf_base=base,base_gguf=gguf,
                        llama_cpp=checkout,output=output)
            self.assertFalse(output.exists())
            self.assertFalse(list(self.root.glob('.williams-convert-*')))
            (trained/'research_manifest.json').write_bytes(original)
            (base/'model.safetensors').write_bytes(original_weights)

    def test_complete_export_preserves_frozen_receipt_and_private_permissions(self):
        base,trained,gguf,checkout,_=export_fixture(self.root)
        output=self.root/'output'
        def run(args,**kwargs):
            if args[0]=='git':
                return subprocess.CompletedProcess(args,0,CONVERTER_COMMIT+'\n' if 'rev-parse' in args else '')
            self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'],'1')
            Path(args[args.index('--outfile')+1]).write_bytes(b'GGUF SYNTHETIC OUTPUT')
            return subprocess.CompletedProcess(args,0)
        with patch('williams_ai.model_bundle.subprocess.run',side_effect=run):
            result=export_runtime_lora(trained_adapter=trained,hf_base=base,base_gguf=gguf,
                llama_cpp=checkout,output=output)
        self.assertFalse(result['adapter_loaded'])
        self.assertEqual((output/'research_manifest.json').read_bytes(),(trained/'research_manifest.json').read_bytes())
        self.assertTrue(all(p.stat().st_mode & 0o077 == 0 for p in output.iterdir()))

    def test_weight_replacement_inside_native_loader_is_rejected_and_closed(self):
        base,adapter,manifest=fake_bundle(self.root)
        calls=[]
        class ChangedRuntime:
            def __init__(self,**kwargs):adapter.write_bytes(b'GGUF CHANGED DURING LOAD')
            def close(self):calls.append('closed')
        with self.assertRaisesRegex(ValueError,'ADAPTER_BYTES_CHANGED'):
            LocalWilliamsModel(base,lora_adapter=adapter,lora_manifest=manifest,loader=ChangedRuntime)
        self.assertEqual(calls,['closed'])

    @contextmanager
    def fit_fixture(self,gate_accepted=True,mutation=None):
        base=hf_fixture(self.root/'hf')
        data=self.root/'reviewed-fixture.jsonl'
        data.write_text('\n'.join(json.dumps(row) for row in candidates()))
        target=self.root/'output'
        before={ed:2.0 for ed in ('ALL','TC1','NTD','TC2','COURSE')}
        after={ed:(1.5 if gate_accepted else 2.0) for ed in before}
        report={'optimizer_steps':10,'selected_checkpoint_step':5,
            'baseline_by_book':before,'adapted_by_book':after,
            'validation_gate':validation_admission(before,after),
            'best_validation_checkpoint_reloaded':True,'adapter_weights_changed':True,
            'base_parameters_frozen':True}
        class Tokenizer:
            eos_token='<end>';pad_token='<pad>';pad_token_id=0
            def apply_chat_template(self,messages,**kwargs):
                return [1,2,3] if kwargs['add_generation_prompt'] else [1,2,3,4,5]
            def save_pretrained(self,path): (Path(path)/'tokenizer.json').write_text('{}')
        class Model:
            def save_pretrained(self,path,**kwargs):
                self_check=kwargs['safe_serialization']
                assert self_check is True
                (Path(path)/'adapter_model.safetensors').write_bytes(b'SYNTHETIC')
                (Path(path)/'adapter_config.json').write_text(json.dumps({
                    'peft_type':'LORA','task_type':'CAUSAL_LM','r':16,'lora_alpha':32}))
        def load_model(*args,**kwargs):
            self.assertTrue(kwargs['use_safetensors']);self.assertTrue(kwargs['local_files_only'])
            self.assertNotIn('device_map',kwargs)
            return Model()
        def fit(model,train,validation,**kwargs):
            self.assertEqual(len(train),64)
            self.assertEqual(set(validation),{'TC1','NTD','TC2','COURSE'})
            self.assertEqual(sum(len(x) for x in validation.values()),18)
            if mutation=='base':(base/'model.safetensors').write_bytes(b'MUTATED DURING FIT')
            if mutation=='target':
                target.mkdir();(target/'user-owned').write_text('preserve')
            return model,report
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,
            is_bf16_supported=lambda:True),bfloat16='bf16',float16='fp16')
        transformers=SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer()),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model))
        with patch.dict('sys.modules',{'torch':torch,'transformers':transformers}),\
             patch('williams_ai.private_lora.fit_lora',side_effect=fit),\
             patch('importlib.metadata.version',return_value='synthetic-fixture'):
            yield lambda:train_private_lora(data=data,model_dir=base,output_dir=target,
                source_only_research=True),target

    def test_fit_wrapper_publishes_only_v2_receipt_with_actual_base_identity(self):
        with self.fit_fixture() as (run,target):
            result=run()
        self.assertTrue(result['adapter_saved'])
        receipt=json.loads((target/'research_manifest.json').read_text())
        _exportable_training_receipt(receipt)
        self.assertEqual(receipt['local_base_identity'],local_hf_identity(self.root/'hf'))
        self.assertTrue(all(path.stat().st_mode & 0o077 == 0 for path in target.iterdir()))
        self.assertFalse(list(self.root.glob('.williams-lora-*')))

    def test_rejected_fit_does_not_leave_adapter_or_block_a_clean_retry(self):
        with self.fit_fixture(gate_accepted=False) as (run,target):
            self.assertFalse(run()['adapter_saved'])
            self.assertFalse(run()['adapter_saved'])
        self.assertFalse(target.exists())
        self.assertFalse(list(self.root.glob('.williams-lora-*')))

    def test_changed_base_during_fit_does_not_publish_adapter(self):
        with self.fit_fixture(mutation='base') as (run,target):
            with self.assertRaisesRegex(ValueError,'BASE_CHANGED_DURING_TRAINING'):run()
        self.assertFalse(target.exists())

    def test_output_appearing_during_fit_is_never_replaced_or_deleted(self):
        with self.fit_fixture(mutation='target') as (run,target):
            with self.assertRaises(FileExistsError):run()
        self.assertEqual((target/'user-owned').read_text(),'preserve')


if __name__=='__main__':unittest.main()
