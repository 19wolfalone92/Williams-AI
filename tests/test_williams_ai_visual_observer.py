"""Boundaries of the optional private multimodal reading pathway."""
import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from williams_ai.local_vision import LocalChartVision,local_image_data


class FakeVisionLlama:
    def __init__(self,**kwargs):
        self.loader_kwargs=kwargs
        self.calls=[]

    def create_chat_completion(self,**kwargs):
        self.calls.append(kwargs)
        return {"choices":[{"message":{"content":json.dumps({
            "observations":["На диаграмме видны A,B,C"],
            "hypotheses":["Вероятна коррекция, масштаб неизвестен"],
            "missing_evidence":["Нет соответствующих OHLC"],
            "verified_wave_degree":True,
            "trade_action":"BUY",
        },ensure_ascii=False)}}]}


class PrivateLocalVisionTests(unittest.TestCase):
    def test_real_image_bytes_go_only_to_mock_multimodal_image_part(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            model=root/"private-model.gguf"
            mmproj=root/"private-mmproj.gguf"
            image=root/"user-chart.png"
            model.write_bytes(b"fake model")
            mmproj.write_bytes(b"fake mmproj")
            image.write_bytes(bytes.fromhex("89504e470d0a1a0a")+b"x"*90)
            analyzer=LocalChartVision(model,mmproj,loader=FakeVisionLlama)
            report=analyzer.observe(image,"Различи видимые линии и будущие проекции")
            self.assertEqual(report["mode"],"PRIVATE_LOCAL_VISION_RESEARCH_ONLY")
            self.assertEqual(report["observations_unverified"][0],"На диаграмме видны A,B,C")
            self.assertEqual(report["verified_wave_degree"],False)
            self.assertIsNone(report["trade_action"])
            self.assertFalse(report["execution_allowed"])
            self.assertNotIn("BUY",str(report))
            call=analyzer._llm.calls[0]
            self.assertNotIn("tools",call)
            self.assertEqual(call["temperature"],0.0)
            self.assertEqual(call["response_format"],{"type":"json_object"})
            user=call["messages"][1]["content"]
            self.assertEqual(user[1]["type"],"image_url")
            uri=user[1]["image_url"]["url"]
            self.assertTrue(uri.startswith("data:image/png;base64,"))
            self.assertEqual(base64.b64decode(uri.split(",",1)[1]),image.read_bytes())
            self.assertIn("mmproj_path",analyzer._llm.loader_kwargs)

    def test_reject_wrong_mime_paths_and_limits(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            image=root/"image.jpg"
            image.write_bytes(b"Not really JPEG"*4)
            with self.assertRaisesRegex(ValueError,"Invalid JPEG"):
                local_image_data(image)
            tiny=root/"small.png"
            tiny.write_bytes(b"x"*10)
            with self.assertRaisesRegex(ValueError,"Image must"):
                local_image_data(tiny)
            fake=root/"x.pdf"
            fake.write_bytes(b"x"*60)
            with self.assertRaisesRegex(ValueError,"PNG/JPEG"):
                local_image_data(fake)

    def test_no_model_or_mmproj_fails_closed(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            model=root/"model.gguf"
            model.write_bytes(b"GGUF")
            with self.assertRaises(FileNotFoundError):
                LocalChartVision(model,root/"missing-mmproj.gguf",loader=FakeVisionLlama)


if __name__=="__main__":
    unittest.main()
