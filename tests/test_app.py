import json

def test_index(app, client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"PlasmoAI" in res.data
    assert b"Malaria" in res.data

def test_diagnose(app, client):
    res = client.get("/diagnose")
    assert res.status_code == 200
    assert b"Microscopy Cell Analyzer" in res.data

def test_form_redirect_compat(app, client):
    res = client.get("/form")
    assert res.status_code == 200

def test_result(app, client):
    res = client.get("/result")
    assert res.status_code == 200
    assert b"Diagnostic Pathology Report" in res.data

def test_batch(app, client):
    res = client.get("/batch")
    assert res.status_code == 200
    assert b"Batch Smear & Parasitemia Studio" in res.data

def test_model_explorer(app, client):
    res = client.get("/model")
    assert res.status_code == 200
    assert b"Convolutional Neural Network" in res.data
    assert b"95.23%" in res.data

def test_research(app, client):
    res = client.get("/research")
    assert res.status_code == 200
    assert b"Research Background" in res.data

def test_api_sample_images(app, client):
    res = client.get("/api/sample-images")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "parasitized" in data["data"]
    assert "uninfected" in data["data"]

def test_api_model_info(app, client):
    res = client.get("/api/model-info")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["data"]["test_accuracy"] == 95.23

def test_api_predict_sample(app, client):
    payload = {"image": "C101P62ThinF_IMG_20150918_151149_cell_73.png"}
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["data"]["prediction"] == "Parasitized"
    assert data["data"]["confidence"] >= 90.0

def test_api_predict_batch(app, client):
    payload = {
        "images": [
            "C101P62ThinF_IMG_20150918_151149_cell_73.png",
            "C109P70ThinF_IMG_20150930_103811_cell_31.png"
        ]
    }
    res = client.post("/api/predict-batch", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["data"]["total_cells"] == 2
    assert data["data"]["parasitemia_index_percent"] == 50.0

def test_404_error(app, client):
    res = client.get("/non_existent_page")
    assert res.status_code == 404
