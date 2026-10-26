"""Pure-CPU QLoRA contract/checkpoint tests; no GPU or private data."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from williams_ai.qlora_gpu_research import contract, seal_checkpoints, verify_resume


def spec():
    return contract(dataset_sha="a"*64,base_sha="b"*64,
                    revision="90354179",max_steps=100,max_tokens=1024,
                    rank=8,rate=5e-5,seed=42)


class Integrity(unittest.TestCase):
    def make(self,root):
        dst=Path(root)
        cp=dst/"checkpoints"/"checkpoint-10"
        cp.mkdir(parents=True)
        for name in ("optimizer.pt","scheduler.pt","rng_state.pth"):
            (cp/name).write_bytes(("TEST FIXTURE "+name).encode())
        (cp/"trainer_state.json").write_text(json.dumps({"global_step":10}))
        receipt=dst/"contract.json"
        receipt.write_text(json.dumps({"contract":spec()}))
        return cp,receipt

    def test_contract_fixed_identifiers_and_bounded_hyperparameters(self):
        self.assertFalse(spec()["execution_allowed"])
        for kwargs in ({"max_steps":13},{"rank":True},{"seed":-1},{"rate":float("nan")},
                       {"dataset_sha":"not-a-hash"}):
            p=dict(dataset_sha="a"*64,base_sha="b"*64,revision="sha",
                   max_steps=100,max_tokens=1024,rank=8,rate=5e-5,seed=42)
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


if __name__=="__main__":unittest.main()
