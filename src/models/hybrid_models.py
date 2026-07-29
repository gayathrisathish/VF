"""Hybrid deep-learning volatility models.

These wrappers reuse the validated deep-learning architectures and training
pipeline while exposing distinct model names for the GARCH-augmented sequence
inputs produced upstream.
"""

from src.models.deep_learning import (
	_build_gru_model,
	_build_lstm_model,
	_build_transformer_model,
	_train_and_evaluate,
)


def train_garch_lstm(
	X_train,
	y_train,
	X_validation,
	y_validation,
	X_test,
	y_test,
	index,
	horizon,
):
	"""Train the GARCH-LSTM hybrid model on GARCH-augmented sequences."""
	del index
	return _train_and_evaluate(
		_build_lstm_model,
		"GARCHLSTM",
		X_train,
		y_train,
		X_validation,
		y_validation,
		X_test,
		y_test,
		horizon,
	)


def train_garch_gru(
	X_train,
	y_train,
	X_validation,
	y_validation,
	X_test,
	y_test,
	index,
	horizon,
):
	"""Train the GARCH-GRU hybrid model on GARCH-augmented sequences."""
	del index
	return _train_and_evaluate(
		_build_gru_model,
		"GARCHGRU",
		X_train,
		y_train,
		X_validation,
		y_validation,
		X_test,
		y_test,
		horizon,
	)


def train_garch_transformer(
	X_train,
	y_train,
	X_validation,
	y_validation,
	X_test,
	y_test,
	index,
	horizon,
):
	"""Train the GARCH-Transformer hybrid model on GARCH-augmented sequences."""
	del index
	return _train_and_evaluate(
		_build_transformer_model,
		"GARCHTransformer",
		X_train,
		y_train,
		X_validation,
		y_validation,
		X_test,
		y_test,
		horizon,
	)
