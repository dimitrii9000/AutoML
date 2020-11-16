import logging
from collections.abc import Iterable
from multiprocessing import Pool

import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, log_loss

# models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


class AutoML:
    def __init__(self, X: Iterable = "", y: Iterable = "", file: str = "", verbose: bool = False):
        """
        :param X: Фичи, на которых предполагается обучать модель
        :param y: Лейблы
        :param file: Путь к csv файлу, откуда можно загрузить все данные
        :param verbose: Нужен ли расширенный output?
        """

        # Устанавливаем уровень логирования
        if verbose:
            logging.basicConfig(level=logging.DEBUG,
            format="%(asctime)s : %(filename)s[LINE:%(lineno)d] : %(levelname)s : %(message)s")
        else:
            logging.basicConfig(level=logging.INFO,
            format="%(asctime)s : %(filename)s[LINE:%(lineno)d] : %(levelname)s : %(message)s")

        # Если данные для обучения в файле, то достаем их
        if file:
            if not file.endswith(".csv"):
                raise Exception("Необходимо, чтобы файл с данными имел расширение '.csv'!")

            dataframe = pandas.read_csv(file)
            try:
                y = dataframe.pop("label").to_numpy()
            except KeyError:
                raise KeyError("There is no 'label' column in dataset!")

            X = dataframe.to_numpy()

        # Разделяем на test и train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        logging.debug("Finished processing data")

        # Определяем модели, которые будем обучать
        _model_names = ["Random Forest",
                        "Gradient Boosting",
                        "GaussianNB",
                        "AdaBoostClassifier",
                        "DecisionTreeClassifier",
                        "KNeighborsClassifier"]

        _models = [
            RandomForestClassifier(),
            GradientBoostingClassifier(),
            GaussianNB(),
            AdaBoostClassifier(),
            DecisionTreeClassifier(),
            KNeighborsClassifier(),
            ]
        self.models_with_names = zip(_model_names, _models)

        self.best_metrics = {}
        self.best_model = None
        self.best_model_name = ""

    def _train(self, zip_model: zip):
        """
        Обучить одну модель
        :param zip_model: zip, состоящий из (имя модели, сама модель)
        :return: zip(Модель, Метрики модели)
        """
        name, clf = zip_model
        try:

            model = clf
            model.fit(self.X_train, self.y_train)

            metrics = self.get_metrics(model)
            return name, model, metrics

        except Exception as e:
            logging.exception(e)
            return None

    def train(self):
        """
        Обучить модели на данных и выбрать среди них лучшую
        :return: None
        """
        logging.debug("Beginning training models")
        with Pool(6) as p:
            models_and_metrics = p.map(self._train, self.models_with_names)
        logging.debug("Finished training models")

        best_auc = 0
        for mm in models_and_metrics:
            if mm:
                name, model, metrics = mm
                if metrics.get("roc_auc") > best_auc:
                    best_auc = metrics.get("roc_auc")
                    self.best_metrics = metrics
                    self.best_model_name = name
                    self.best_model = model

        if not self.best_model_name:
            raise Exception("Ни одна модель не была успешно обучена! Скорее всего в данных есть ошибки!")

        logging.debug("---------------------------")
        logging.debug(f"Лучшая модель: {self.best_model_name}")
        logging.debug(f"Её ROC AUC: {best_auc}")
        logging.debug("---------------------------")

    def get_metrics(self, model):
        """
        Получить метрики модели на тестовых данных
        :param model: Модель, метрики которой нужно проверить
        :return: Метрики модели
        """

        y_pred = model.predict(self.X_test)
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)

        metrics = {
            'true_positive': int(tp), 'false_positive': int(fp),
            'true_negative': int(tn), 'false_negative': int(fn),
            'roc_auc': roc_auc_score(self.y_test, y_pred),
            'accuracy': accuracy_score(self.y_test, y_pred),
            'log_loss': log_loss(self.y_test, y_pred),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        return metrics
