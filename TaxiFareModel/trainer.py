# imports
from sklearn.pipeline import Pipeline
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
import joblib

class Trainer():
    def __init__(self, X, y, estimator):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.estimator = estimator
        self.val_score = None
        self.cv_score = None
        self.X = X
        self.y = y
        self.MLFLOW_URI = "https://mlflow.lewagon.ai/"
        self.experiment_name = "[DE] [Berlin] [N1tram] taxi_fare_model_v2"

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())])

        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude",
                                     'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])],
                                         remainder="drop")

        self.pipeline=Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', self.estimator)])


    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(X=self.X, y=self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        self.val_score = compute_rmse(y_pred, y_test)
        self.cv_score = cross_validate(self.pipeline, X_test, y_test,
                                       cv=5)['test_score'].mean()


    def save_model(self):
        """ Save the trained model into a model.joblib file """
        self.evaluate(X_test,y_test)
        joblib.dump(self.pipeline, 'pipeline.joblib')
        trainer.mlflow_log_param('models',self.estimator)
        trainer.mlflow_log_metric('val_score', self.val_score)
        trainer.mlflow_log_metric('cv_score', self.cv_score)

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from TaxiFareModel.data import get_data, clean_data
    from sklearn.linear_model import LinearRegression, SGDRegressor
    estimators = [LinearRegression(),SGDRegressor()]
    for estimator in estimators:
        N = 10_000
        df = get_data(nrows=N)
        df = clean_data(df)
        y = df["fare_amount"]
        X = df.drop("fare_amount", axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        trainer = Trainer(X_train, y_train, estimator)
        trainer.run()
        trainer.save_model()
