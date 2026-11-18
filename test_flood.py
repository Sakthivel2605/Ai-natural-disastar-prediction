from flood_predictor import train


def test_model_learns_the_simulated_pattern():
    model, metrics = train()
    assert metrics["accuracy"] > 0.8
    wet = model.predict_proba([[200, 5.0, 90]])[0][1]
    dry = model.predict_proba([[5, 2.0, 20]])[0][1]
    assert wet > dry
