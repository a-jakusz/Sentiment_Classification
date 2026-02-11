#Movie Review Sentiment Classifier using SGD

A binary sentiment classifier that predicts whether a movie review is positive (1) or negative (-1). The system implements a custom feature extractor, SGD-based weight training, and a prediction module. Given training directories of positive and negative reviews, it learns to classify unseen test reviews and outputs the accuracy percentage.

Features:
- Custom feature extraction from raw text reviews
- SGD algorithm for weight vector optimization
- Binary prediction (-1 or 1) for test reviews
- Command-line interface for training data directories
- Performance evaluation on test set
