from sklearn.feature_selection import SelectKBest, f_classif

# Feature selection
selector = SelectKBest(score_func=f_classif, k='all')  # Adjust 'k' as needed
X_selected = selector.fit_transform(X_preprocessed, y)
