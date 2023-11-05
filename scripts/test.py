from sklearn.metrics import f1_score
def test(model,images,labels):
    predictions = model.predict(images)
    predictions = (predictions > 0.5).astype(int)  # Binarisez les prédictions

    # Calculer le F1-score
    f1 = f1_score(labels, predictions)

    return f1