import logging
import os
import sys
import warnings
import numpy as np
import pandas as pd


import mindspore as ms
from mindspore import nn, ops, Tensor, save_checkpoint
from mindspore.train import Model, LossMonitor, TimeMonitor
from mindspore.dataset import NumpySlicesDataset
from mindspore.nn import Adam, MSELoss


from sklearn.preprocessing import RobustScaler

from config import CONFIG


def _auto_configure_hardware():
    """
    Configure MindSpore pour Ascend si disponible, sinon CPU fallback.
    Retourne: (device_target, use_dataset_sink)
    """
    # Supprimer temporairement les warnings Ascend parasites sur CPU
    original_glog = os.environ.get('GLOG_v', None)
    os.environ['GLOG_v'] = '3'
    
    try:
        # Tentative Ascend (prioritaire pour compétition)
        ms.set_device("Ascend")
        ms.set_context(mode=ms.GRAPH_MODE, max_call_depth=10000)
        if ms.get_context("device_target") == "Ascend":
            print(" [HARDWARE] Ascend NPU detected - using GRAPH_MODE + dataset_sink")
            return "Ascend", True
    except (RuntimeError, ValueError):
        pass  # Silencieux - fallback CPU
    
    # Fallback CPU (garanti fonctionnel)
    ms.set_device("CPU")
    ms.set_context(mode=ms.PYNATIVE_MODE)
    print(" [HARDWARE] CPU detected - using PYNATIVE_MODE (no dataset_sink)")
    
    # Restaurer GLOG_v original si existant
    if original_glog is not None:
        os.environ['GLOG_v'] = original_glog
    else:
        os.environ.pop('GLOG_v', None)
    
    return "CPU", False


DEVICE_TARGET, USE_DATASET_SINK = _auto_configure_hardware()

# Création dossiers de sortie
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

# Configuration logging standard
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"], mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
warnings.filterwarnings('ignore')


class TabularMLP(nn.Cell):
    """
    MLP tabulaire robuste pour données urbaines.
    Architecture inchangée - fonctionne sur CPU ET Ascend.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.SequentialCell([
            nn.Dense(input_dim, 256),
            nn.LayerNorm((256,)),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Dense(256, 128),
            nn.LayerNorm((128,)),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Dense(128, 64),
            nn.ReLU(),
            nn.Dense(64, 1)  # Régression
        ])
    
    def construct(self, x):
        return self.network(x)


class ExplainableMLCore:
   
    def __init__(self, graph=None):
        self.graph = graph  # Non utilisé par MLP (conservé pour compatibilité API)
        self.model = None
        self.scaler = RobustScaler()
        self.feature_importance = None
        self.device_target = DEVICE_TARGET  # Pour logs explicites
        
    def train(self, X_df, y_series, graph=None):
        """
        Entraîne un MLP MindSpore avec configuration hardware adaptative.
        Architecture MLP inchangée - seulement optimisation runtime adaptative.
        """
        logging.info("=" * 80)
        logging.info("PHASE 4: MACHINE LEARNING WITH MINDSPORE (AUTO-HARDWARE MLP)")
        logging.info("=" * 80)
        logging.info(f"  ⚙️  Hardware: {self.device_target} | Mode: {'GRAPH' if USE_DATASET_SINK else 'PYNATIVE'} | Sink: {USE_DATASET_SINK}")
        
        # Préprocessing des features (inchangé)
        X = X_df.fillna(0).astype(np.float32)
        y = y_series.fillna(0).astype(np.float32)
        X_scaled = self.scaler.fit_transform(X).astype(np.float32)
        y_scaled = y.values.reshape(-1, 1).astype(np.float32)
        
        # Création dataset MindSpore (inchangé)
        dataset = NumpySlicesDataset(
            (X_scaled, y_scaled),
            column_names=["data", "label"],
            shuffle=True
        ).batch(64)
        
        # Initialisation du modèle (inchangé)
        self.model = TabularMLP(X.shape[1])
        param_count = sum(p.size for p in self.model.get_parameters())
        logging.info(f"  Initialized TabularMLP | Input dim: {X.shape[1]} | Params: {param_count:,}")
        
        # Optimisation (inchangé)
        loss_fn = MSELoss()
        optimizer = Adam(self.model.trainable_params(), learning_rate=0.001)
        model_wrapper = Model(network=self.model, loss_fn=loss_fn, optimizer=optimizer, metrics={"mae": nn.MAE()})
        
        # 🔧 ENTRAÎNEMENT AVEC CONFIGURATION ADAPTATIVE (SEULE MODIFICATION MINIMALE)
        logging.info(f"  Training MLP (100 epochs) on {self.device_target}...")
        model_wrapper.train(
            epoch=100,
            train_dataset=dataset,
            callbacks=[LossMonitor(per_print_times=20), TimeMonitor(data_size=64)],
            dataset_sink_mode=USE_DATASET_SINK 
        )
        
        # Évaluation (inchangé)
        X_test_tensor = Tensor(X_scaled, ms.float32)
        y_pred = self.model(X_test_tensor).asnumpy().flatten()
        train_r2 = self._compute_r2(y.values, y_pred)
        train_mae = np.mean(np.abs(y.values - y_pred))
        
        logging.info(f"   Training R²: {train_r2:.4f}")
        logging.info(f"   Training MAE: {train_mae:.4f}")
        logging.info(f"   Model size: {param_count * 4 / 1024:.2f} KB")
        
        # Explicabilité (inchangé)
        self._compute_feature_importance_mlp(X_test_tensor, X_df.columns)
        
        # 🔧 SAUVEGARDE AVEC SUFFIXE HARDWARE (pour traçabilité soumission)
        suffix = "ascend" if self.device_target == "Ascend" else "cpu"
        ckpt_path = os.path.join(CONFIG["OUTPUT_DIR"], f"mindspore_mlp_model_{suffix}.ckpt")
        save_checkpoint(self.model, ckpt_path)
        logging.info(f"  Model saved to: {ckpt_path} ({self.device_target} execution)")
        
        return self.model
    
    def predict(self, X_df):
        """Prédit les scores avec le MLP entraîné (inchangé)."""
        if self.model is None:
            raise RuntimeError("Modèle non entraîné. Appeler train() d'abord.")
        
        X = X_df.fillna(0).astype(np.float32)
        X_scaled = self.scaler.transform(X).astype(np.float32)
        X_tensor = Tensor(X_scaled, ms.float32)
        preds = self.model(X_tensor).asnumpy().flatten()
        return pd.Series(preds, index=X_df.index)
    
    def _compute_r2(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    
    def _compute_feature_importance_mlp(self, X_tensor, feature_names):
        """Explicabilité CPU-safe via gradients moyens (inchangé)."""
        logging.info("  Computing feature importance via gradient approximation...")
        grad_fn = ops.value_and_grad(self.model, grad_position=0)
        _, grads = grad_fn(X_tensor)
        mean_abs_grad = np.mean(np.abs(grads.asnumpy()), axis=0)
        
        if np.sum(mean_abs_grad) > 0:
            mean_abs_grad = mean_abs_grad / np.sum(mean_abs_grad)
        
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_grad
        }).sort_values('importance', ascending=False)
        
        logging.info("\n  Top 3 Most Important Features (Gradient Approximation):")
        for idx, row in self.feature_importance.head(3).iterrows():
            logging.info(f"    {row['feature']:20s} | {row['importance']:.4f}")
        
        imp_path = os.path.join(CONFIG["OUTPUT_DIR"], "feature_importance_mlp.csv")
        self.feature_importance.to_csv(imp_path, index=False)
        logging.info(f"  Feature importance saved to: {imp_path}")
