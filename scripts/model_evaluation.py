from sklearn.metrics import classification_report, accuracy_score

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))
