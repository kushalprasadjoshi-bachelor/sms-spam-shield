import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional

from .base_trainer import BaseTrainer
from backend.app.core.config import settings
from backend.app.core.logger import logger
from ml.training.versioning import ModelVersionManager


class LSTMTrainer(BaseTrainer):
    """LSTM model with attention mechanism for SMS classification"""

    def __init__(self, use_pretrained_embeddings: bool = False, embedding_dim: int = 100):
        # Call BaseTrainer __init__ with model_name
        super().__init__("lstm")
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.embedding_dim = embedding_dim
        self.max_words = 10000
        self.max_len = 100
        self.tokenizer = None
        self.label_encoder = None
        self.version_manager = ModelVersionManager(self.model_name)

        self.model_params = {
            "lstm_units": 128,
            "dropout": 0.5,
            "recurrent_dropout": 0.2,
            "dense_units": 64,
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 20,
            "embedding_dim": self.embedding_dim,
            "use_attention": True,
            "use_pretrained_embeddings": self.use_pretrained_embeddings
        }

    # Override abstract methods from BaseTrainer
    def create_vectorizer(self):
        """No vectorizer for LSTM – we use tokenizer."""
        # Simply set vectorizer to None to avoid errors in base methods
        self.vectorizer = None
        logger.info("LSTM uses tokenizer, not vectorizer.")

    def create_model(self, num_classes: int):
        """Create the LSTM model architecture with unique layer names."""
        inputs = Input(shape=(self.max_len,))
    
        embedding = Embedding(
            input_dim=self.max_words,
            output_dim=self.embedding_dim,
            name="embedding"
        )(inputs)
    
        lstm_out = LSTM(
            units=self.model_params["lstm_units"],
            dropout=self.model_params["dropout"],
            recurrent_dropout=self.model_params["recurrent_dropout"],
            return_sequences=True,
            name="lstm"
        )(embedding)
    
        # Attention mechanism – give this Dense a unique name
        attention = Dense(1, activation='tanh', name='attention_score')(lstm_out)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax', name='attention_weights')(attention)
        attention = tf.keras.layers.RepeatVector(self.model_params["lstm_units"])(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)
        attended = tf.keras.layers.Multiply()([lstm_out, attention])
        attended = GlobalAveragePooling1D()(attended)
    
        # This Dense can keep the name "dense" (unique now)
        dense = Dense(self.model_params["dense_units"], activation='relu', name="dense")(attended)
        dropout = Dropout(self.model_params["dropout"])(dense)
        outputs = Dense(num_classes, activation='softmax', name="output")(dropout)
    
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.model_params["learning_rate"])
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        logger.info(f"LSTM model created with {model.count_params()} parameters")

    def extract_features(self, X_train, X_test):
        """Override to handle tokenizer instead of vectorizer."""
        # This method is called in base train_pipeline; we skip actual feature extraction here
        # because it's handled in train() using tokenizer.
        pass

    def train(self):
        """Train the LSTM model (overrides base train method)."""
        # This method is called after create_model in our custom pipeline
        # We'll implement the full training inside train_pipeline instead.
        pass  # Not used directly; we override train_pipeline

    def train_pipeline(self) -> Dict[str, Any]:
        """Complete training pipeline for LSTM (overrides base)."""
        try:
            df = self.load_data()
            X_raw, y_raw = self.preprocess_data(df)

            # Encode labels
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y_raw)
            num_classes = len(self.label_encoder.classes_)

            # Create and fit tokenizer
            self.tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(X_raw)

            # Convert texts to sequences and pad
            sequences = self.tokenizer.texts_to_sequences(X_raw)
            X_seq = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_seq, y_encoded,
                test_size=settings.TEST_SIZE,
                random_state=settings.RANDOM_STATE,
                stratify=y_encoded
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=0.1,
                random_state=settings.RANDOM_STATE,
                stratify=y_train
            )

            # Create model
            self.create_model(num_classes)

            # Load pre-trained embeddings if requested
            if self.use_pretrained_embeddings:
                emb_path = getattr(settings, 'PRETRAINED_EMBEDDINGS_PATH', None)
                if emb_path and Path(emb_path).exists():
                    self._load_pretrained_embeddings(emb_path)

            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
                ModelCheckpoint(
                    filepath=str(self.version_manager.model_dir / 'best_model.h5'),
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max'
                )
            ]

            # Train
            history = self.model.fit(
                X_train, y_train,
                batch_size=self.model_params["batch_size"],
                epochs=self.model_params["epochs"],
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )

            # Load best weights
            self.model.load_weights(self.version_manager.model_dir / 'best_model.h5')

            # Evaluate
            metrics = self.evaluate_model(X_test, y_test, history)

            # Save model and artifacts
            self.save_model(metrics, self.model_params, history)

            return {
                "success": True,
                "metrics": metrics,
                "model": self.model_name,
                "history": {k: [float(v) for v in vals] for k, vals in history.history.items()}
            }

        except Exception as e:
            logger.error(f"LSTM training failed: {str(e)}")
            return {"success": False, "error": str(e), "model": self.model_name}

    def evaluate_model(self, X_test, y_test, history) -> Dict[str, float]:
        """Evaluate LSTM model."""
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred_prob = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)

        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            "accuracy": float(accuracy),
            "loss": float(loss),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "val_accuracy": float(max(history.history['val_accuracy'])),
            "val_loss": float(min(history.history['val_loss'])),
            "confusion_matrix": cm.tolist()
        }
        logger.info(f"LSTM Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return metrics

    def _load_pretrained_embeddings(self, embedding_path: str):
        """Load pre-trained embeddings (e.g., GloVe) into the embedding layer."""
        logger.info(f"Loading pre-trained embeddings from {embedding_path}")
        embeddings_index = {}
        with open(embedding_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        embedding_matrix = np.zeros((self.max_words, self.embedding_dim))
        for word, i in self.tokenizer.word_index.items():
            if i < self.max_words:
                vec = embeddings_index.get(word)
                if vec is not None:
                    embedding_matrix[i] = vec

        self.model.get_layer('embedding').set_weights([embedding_matrix])
        logger.info("Pre-trained embeddings loaded into model")

    def save_model(self, metrics: Dict[str, float], params: Dict[str, Any], history):
        """Save model, tokenizer, label encoder, and metadata."""
        model_path = self.version_manager.model_dir / "model.h5"
        self.model.save(model_path)

        with open(self.version_manager.model_dir / "tokenizer.pkl", 'wb') as f:
            pickle.dump(self.tokenizer, f)
        with open(self.version_manager.model_dir / "label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)

        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(self.version_manager.model_dir / "history.json", 'w') as f:
            json.dump(history_dict, f, indent=2)

        # Use version manager to save version
        version = self.version_manager.save_version(
            model=self.model,
            vectorizer=self.tokenizer,  # we store tokenizer as vectorizer for compatibility
            metrics=metrics,
            params=params
        )
        if len(self.version_manager.list_versions()) == 1:
            self.version_manager.set_production(version)

    def load_production_model(self):
        """Load the production version of the model."""
        try:
            prod = self.version_manager.get_production_version()
            if not prod:
                return False
            version_dir = self.version_manager.versions_dir / prod["version"]
            self.model = load_model(version_dir / "model.h5")
            with open(version_dir / "tokenizer.pkl", 'rb') as f:
                self.tokenizer = pickle.load(f)
            with open(version_dir / "label_encoder.pkl", 'rb') as f:
                self.label_encoder = pickle.load(f)
            logger.info(f"Loaded production LSTM version {prod['version']}")
            return True
        except Exception as e:
            logger.error(f"Failed to load production LSTM: {e}")
            return False