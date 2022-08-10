from collections import defaultdict, Counter
from math import isclose
from typing import List
from pickle import dump, load


class IntentSuggester:
    """
    Class for intent recognition using partial phrases. Uses n-gram language model to learn statistics from training data

    Attributes
    ----------
    n
    smoothing
    recursive
    counter_ : Counter
        n-gram frequencies for different intents
    child_ : IntentSuggester
        child model with n equal n-1 (where n is parent model's parameter)
    """
    def __init__(self, n=3, smoothing=0.01, recursive=0.01):
        """
        Parameters
        ----------
        n : int
            n-gram size. Last n words of input sequences will affect intent probabilities the most
        smoothing : float
            prevents zero-division and applies label smoothing
        recursive : float
            weight for child models predictions
        """
        self.n = n
        self.smoothing = smoothing
        self.recursive = recursive

        self.counter_ = defaultdict(Counter)

        if self.n > 1:  # the last model uses only 1 context word and doesn't have the child model
            self.child_ = IntentSuggester(self.n - 1, self.smoothing, self.recursive)
        else:
            self.child_ = None

    def fit(self, items, labels):
        """Uses provided data to build n-grams frequency counts mapped to the corresponding intents

        Parameters
        ----------
        items : List
            training phrases
        labels : List
        """
        self.intents_ = list(dict.fromkeys(labels))

        for item, label in zip(items, labels):
            if isinstance(item, str):
                item = item.split()
            self.fit_step(item, label)

    def fit_step(self, item, label):
        """Uses one training example for training

        Parameters
        ----------
        item : List of str
            training example
        label : str or int
        """
        if len(item) >= self.n:
            for idx, _ in enumerate(item[self.n - 1:]):  # slide over the item using n-word-sized window
                context = item[idx: (idx + self.n)]
                context = " ".join(context)  # use the current n-gram as a key in the counter
                self.counter_[context][label] += 1  # increase the counter for the corresponding intent

        if self.child_:
            self.child_.fit_step(item, label)

    def get_counts(self, context):
        """Calculates n-gram frequencies for the input phrase

        Parameters
        ----------
        context : str

        Returns
        -------
        Counter
        """
        assert len(context) > 0, "Empty string"

        context = context.split()
        context = context[-self.n:]  # use only the last n words

        context = " ".join(context)

        context_counts = self.counter_.get(context, None)  # get intent counts for our n-gram
        intent_counts = Counter({intent: self.smoothing for intent in self.intents_})  # initialise empty counter with smoothing to avoid zero division

        if context_counts:  # if the (parent) model has seen this n-gram during training
            for intent in self.intents_:  # update counter values for every intent the model has seen this n-gram in
                intent_counts[intent] += context_counts.get(intent, 0)

        if self.child_:  # add to the counter weighted child models' counts
            self.child_.intents_ = self.intents_
            child_counts = self.child_.get_counts(context)
            child_counts = {key: (value * self.recursive) for key, value in child_counts.items()}
            intent_counts += child_counts

        return intent_counts

    def predict_proba(self, context):
        """Predicts probabilities for the input phrase to be attributed to each intent

        Parameters
        ----------
        context : str

        Returns
        -------
        dict
        """
        intent_counts = self.get_counts(context)
        sum_ = sum(intent_counts.values())
        return {key: round(value/sum_, 5) for key, value in intent_counts.items()}

    def predict(self, context, k=3):
        """Returns k most probable intents for the input phrase. If the phrase does not correspond to any intent -
        empty dictionary is returned

        Parameters
        ----------
        context : str
        k : int

        Returns
        -------
        dict
        """
        intent_prob = self.predict_proba(context)
        intent_prob = sorted(intent_prob.items(), key=lambda x: x[1], reverse=True)[:k]

        if not all([isclose(intent_prob[0][1], item[1]) for item in intent_prob]):
            return dict(intent_prob)
        else:
            return {}

    def to_pickle(self, path):
        """Uses pickle to save the model

        Parameters
        ----------
        path : str
            saving path
        """
        with open(path, "wb") as file:
            dump(self, file)

    @staticmethod
    def from_pickle(path):
        """Loads pretrained model saved with pickle

        Parameters
        ----------
        path : str
            path to the model saved with pickle
        Returns
        -------
        IntentSuggester
        """
        with open(path, "rb") as file:
            suggester = load(file)

        assert isinstance(suggester, IntentSuggester), f"Unknown type. Expected IntentSuggester," \
                                                       f" but got {type(suggester)}"
        return suggester
