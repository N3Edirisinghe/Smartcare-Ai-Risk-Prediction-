from sklearn.preprocessing import StandardScaler


def create_scaler():
    """
    Create StandardScaler for numerical features.
    """

    scaler = StandardScaler()

    return scaler


def scale_features(
    scaler,
    X_train,
    X_test
):
    """
    Fit scaler using training data and transform
    both training and testing data.
    """

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    return X_train_scaled, X_test_scaled