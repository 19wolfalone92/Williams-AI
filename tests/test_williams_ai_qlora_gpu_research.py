"""Pure-CPU QLoRA contract/checkpoint tests; no GPU or private data."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from williams_ai.qlora_gpu_research import (
    contract, seal_checkpoints, verify_resume, latest_checkpoint, run_gpu,
)


def spec():
    return contract(dataset_sha="a"*64,base_sha="b"*64,
                    revision="d"*40,max_steps=100,max_tokens=1024,
                    rank=8,rate=5e-5,seed=42,runtime={"torch":"test-fixture-only"})


class Integrity(unittest.TestCase):
    def make(self,root):
        dst=Path(root)
        cp=dst/"checkpoints"/"checkpoint-10"
        cp.mkdir(parents=True)
        for name in ("optimizer.pt","scheduler.pt","rng_state.pth","scaler.pt",
                     "training_args.bin","adapter_model.safetensors"):
            (cp/name).write_bytes(("TEST FIXTURE "+name).encode())
        (cp/"adapter_config.json").write_text(json.dumps({"peft_type":"LORA",
            "r":8,"lora_alpha":16,"lora_dropout":0.05,
            "target_modules":spec()["target_modules"]}))
        (cp/"trainer_state.json").write_text(json.dumps({"global_step":10,"max_steps":100}))
        receipt=dst/"contract.json"
        receipt.write_text(json.dumps({"contract":spec(),
            "original_baseline":{ed:2.0 for ed in ("ALL","TC1","NTD","TC2","COURSE")}}))
        return cp,receipt

    def test_contract_fixed_identifiers_and_bounded_hyperparameters(self):
        self.assertFalse(spec()["execution_allowed"])
        for kwargs in ({"max_steps":13},{"rank":True},{"seed":-1},{"rate":float("nan")},
                       {"dataset_sha":"not-a-hash"},{"revision":"90354179"},
                       {"runtime":{}},{"source_only_research":"yes"}):
            p=dict(dataset_sha="a"*64,base_sha="b"*64,revision="d"*40,
                   max_steps=100,max_tokens=1024,rank=8,rate=5e-5,seed=42,
                   runtime={"torch":"test-fixture-only"})
            p.update(kwargs)
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                contract(**p)

    def test_full_checkpoint_roundtrip_and_tamper_detection(self):
        with TemporaryDirectory() as root:
            cp,receipt=self.make(root)
            m=seal_checkpoints(Path(root))
            self.assertIn("optimizer.pt",m["checkpoint-10"])
            self.assertEqual(verify_resume(cp,receipt,spec()),10)
            (cp/"optimizer.pt").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError,"CHANGED_CHECKPOINT"):
                verify_resume(cp,receipt,spec())

    def test_cross_dataset_or_wrong_step_rejected(self):
        with TemporaryDirectory() as root:
            cp,receipt=self.make(root)
            seal_checkpoints(Path(root))
            changed={**spec(),"dataset_sha256":"c"*64}
            with self.assertRaisesRegex(ValueError,"INCOMPATIBLE"):
                verify_resume(cp,receipt,changed)
            (cp/"trainer_state.json").write_text('{"global_step":20}')
            with self.assertRaisesRegex(ValueError,"CHANGED_CHECKPOINT"):
                verify_resume(cp,receipt,spec())

    def test_incomplete_checkpoint_never_sealed(self):
        with TemporaryDirectory() as root:
            cp,receipt=self.make(root)
            (cp/"optimizer.pt").unlink()
            with self.assertRaisesRegex(ValueError,"NO_COMPLETE"):
                seal_checkpoints(Path(root))

    def test_symlinks_and_nested_files_rejected(self):
        with TemporaryDirectory() as root:
            cp,receipt=self.make(root)
            (cp/"nested").mkdir()
            with self.assertRaisesRegex(ValueError,"NESTED_OR_LINKED"):
                seal_checkpoints(Path(root))

    def test_model_weights_config_and_fp16_scaler_are_required(self):
        # The old implementation sealed these optimizer-only directories.
        for missing in ("adapter_model.safetensors","adapter_config.json",
                        "scaler.pt","training_args.bin"):
            with self.subTest(missing=missing),TemporaryDirectory() as root:
                cp,receipt=self.make(root)
                (cp/missing).unlink()
                with self.assertRaisesRegex(ValueError,"NO_COMPLETE"):
                    seal_checkpoints(Path(root))

    def test_original_baseline_is_bound_to_checkpoint_seal(self):
        with TemporaryDirectory() as root:
            cp,receipt=self.make(root)
            seal_checkpoints(Path(root))
            data=json.loads(receipt.read_text())
            data["original_baseline"]["ALL"]=200.0
            receipt.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"CHANGED_BASELINE"):
                verify_resume(cp,receipt,spec())

    def test_changed_runtime_or_supervision_cannot_resume(self):
        with TemporaryDirectory() as root:
            cp,receipt=self.make(root)
            seal_checkpoints(Path(root))
            for changed in ({**spec(),"runtime":{"torch":"another-version"}},
                            {**spec(),"source_only_research":True}):
                with self.subTest(changed=changed),self.assertRaisesRegex(ValueError,"INCOMPATIBLE"):
                    verify_resume(cp,receipt,changed)

    def test_stale_temp_from_killed_writer_does_not_disable_sealing(self):
        with TemporaryDirectory() as root:
            cp,receipt=self.make(root)
            (Path(root)/"checkpoint_sha256.json.new").write_text("interrupted old writer")
            seal_checkpoints(Path(root))
            self.assertEqual(verify_resume(cp,receipt,spec()),10)

    def test_latest_sealed_checkpoint_ignores_unsealed_partial_save(self):
        with TemporaryDirectory() as root:
            cp,receipt=self.make(root)
            seal_checkpoints(Path(root))
            (cp.parent/"checkpoint-20").mkdir()
            selected,selected_receipt=latest_checkpoint(Path(root))
            self.assertEqual((selected,selected_receipt),(cp,receipt))
            (cp/"optimizer.pt").write_bytes(b"damaged")
            with self.assertRaisesRegex(ValueError,"CHANGED_CHECKPOINT"):
                latest_checkpoint(Path(root))

    def test_rehashed_optimizer_only_directory_still_cannot_resume(self):
        with TemporaryDirectory() as root:
            cp,receipt=self.make(root)
            seal_checkpoints(Path(root))
            (cp/"adapter_model.safetensors").unlink()
            manifest=Path(root)/"checkpoint_sha256.json"
            data=json.loads(manifest.read_text())
            del data["checkpoints"][cp.name]["adapter_model.safetensors"]
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"FULL_OPTIMIZER_STATE"):
                verify_resume(cp,receipt,spec())

    def test_duplicate_json_keys_and_infinite_baseline_are_rejected(self):
        for invalid in ('{"contract":{},"contract":{}}',
                        '{"contract":{},"original_baseline":{"ALL":NaN}}'):
            with self.subTest(invalid=invalid),TemporaryDirectory() as root:
                cp,receipt=self.make(root)
                receipt.write_text(invalid)
                with self.assertRaises(ValueError):seal_checkpoints(Path(root))

    def test_adapter_configuration_and_training_schedule_must_match(self):
        for filename,field,value in (("adapter_config.json","r",16),
                                    ("trainer_state.json","max_steps",200)):
            with self.subTest(field=field),TemporaryDirectory() as root:
                cp,receipt=self.make(root)
                data=json.loads((cp/filename).read_text())
                data[field]=value
                (cp/filename).write_text(json.dumps(data))
                with self.assertRaises(ValueError):seal_checkpoints(Path(root))

    def test_resume_cannot_use_checkpoint_outside_receipt_tree(self):
        with TemporaryDirectory() as root,TemporaryDirectory() as other:
            cp,receipt=self.make(root)
            seal_checkpoints(Path(root))
            other_cp,_=self.make(other)
            with self.assertRaisesRegex(ValueError,"BELONG_TO_RECEIPT"):
                verify_resume(other_cp,receipt,spec())

    def test_cloud_and_pickle_authorization_fail_before_reading_any_data(self):
        arguments=dict(data=Path("absent.jsonl"),model_dir=Path("absent"),
                       output=Path("out"),source_revision="d"*40)
        with self.assertRaisesRegex(ValueError,"CLOUD_PERMISSION"):
            run_gpu(**arguments)
        with self.assertRaisesRegex(ValueError,"UNTRUSTED_OPTIMIZER"):
            run_gpu(**arguments,cloud_data_authorized=True,resume_directory=Path("absent"))


if __name__=="__main__":unittest.main()
