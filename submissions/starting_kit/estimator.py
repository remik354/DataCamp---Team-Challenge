from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

def get_estimator():
    num_processor = "passthrough"
    num_columns = ["G1_math", "G2_math", 'G3_math']

    preprocessor = make_column_transformer(
        (num_processor, num_columns),
        remainder="drop",
    )
    return make_pipeline(preprocessor, RandomForestRegressor())
