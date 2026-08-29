import unittest
from app import app

class PlasmoAITestCase(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_index(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"PlasmoAI", res.data)
        self.assertIn(b"Malaria", res.data)

    def test_diagnose(self):
        res = self.client.get("/diagnose")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Microscopy Cell Analyzer", res.data)

    def test_form_compat(self):
        res = self.client.get("/form")
        self.assertEqual(res.status_code, 200)

    def test_result(self):
        res = self.client.get("/result")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Diagnostic Pathology Report", res.data)

    def test_batch(self):
        res = self.client.get("/batch")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Batch Smear &amp; Parasitemia Studio", res.data)

    def test_model_explorer(self):
        res = self.client.get("/model")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Convolutional Neural Network", res.data)
        self.assertIn(b"95.23%", res.data)

    def test_research(self):
        res = self.client.get("/research")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Research Background", res.data)

    def test_api_sample_images(self):
        res = self.client.get("/api/sample-images")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("parasitized", data["data"])
        self.assertIn("uninfected", data["data"])

    def test_api_model_info(self):
        res = self.client.get("/api/model-info")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["test_accuracy"], 95.23)

    def test_api_predict_sample(self):
        payload = {"image": "C101P62ThinF_IMG_20150918_151149_cell_73.png"}
        res = self.client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["prediction"], "Parasitized")
        self.assertGreaterEqual(data["data"]["confidence"], 90.0)

    def test_api_predict_batch(self):
        payload = {
            "images": [
                "C101P62ThinF_IMG_20150918_151149_cell_73.png",
                "C109P70ThinF_IMG_20150930_103811_cell_31.png"
            ]
        }
        res = self.client.post("/api/predict-batch", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["total_cells"], 2)
        self.assertEqual(data["data"]["parasitemia_index_percent"], 50.0)

    def test_404_error(self):
        res = self.client.get("/non_existent_page")
        self.assertEqual(res.status_code, 404)

if __name__ == "__main__":
    unittest.main()
