import logging
import os
import sys
import warnings
import numpy as np
import pandas as pd

# ✅ MINDSPORE STACK (100% conforme Huawei)
import mindspore as ms
from mindspore import nn, ops, Tensor, save_checkpoint
from mindspore.train import Model, LossMonitor, TimeMonitor
from mindspore.dataset import GeneratorDataset, NumpySlicesDataset
from mindspore.nn import Adam, MSELoss

# 🔧 IMPORT SÉCURISÉ DE MINDSPORE_GL
try:
    from mindspore_gl import Graph as MSGLGraph, GNNCell
    from mindspore_gl.nn import GATConv
    MSGL_AVAILABLE = True
    logging.info("✅ MindSpore Graph Learning available")
except (ImportError, ModuleNotFoundError) as e:
    MSGL_AVAILABLE = False
    logging.warning(f"⚠️  MindSpore Graph Learning unavailable: {e}. Will fallback to MLP on CPU.")

from sklearn.preprocessing import RobustScaler
from config import CONFIG

# 🔧 DÉTECTION HARDWARE SANS ERREURS
def _detect_hardware():
    """Détection Ascend → CPU fallback silencieuse"""
    try:
        ms.set_device("Ascend")
        ms.set_context(mode=ms.GRAPH_MODE, max_call_depth=10000)
        if ms.get_context("device_target") == "Ascend":
            print("✅ [HARDWARE] Ascend NPU activated")
            return "Ascend", ms.GRAPH_MODE, True
    except (RuntimeError, ValueError):
        pass
    
    ms.set_device("CPU")
    ms.set_context(mode=ms.PYNATIVE_MODE)
    print("💻 [HARDWARE] CPU fallback activated (PYNATIVE_MODE)")
    return "CPU", ms.PYNATIVE_MODE, False

DEVICE_TARGET, CONTEXT_MODE, DATASET_SINK = _detect_hardware()
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"], mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
warnings.filterwarnings('ignore')


# 🔧 DÉFINITION CORRECTE DU GNN (signature MSGL obligatoire)
if MSGL_AVAILABLE:
    class UrbanGNN(GNNCell):
        """
        GNN avec signature MSGL CORRECTE : construct(self, x, g: Graph)
        """
        def __init__(self, node_feat_size, hidden_dim=64):
            super().__init__()
            self.gat1 = GATConv(
                in_feat_size=node_feat_size,
                out_feat_size=hidden_dim,
                num_heads=4,
                feat_drop=0.3,
                attn_drop=0.2
            )
            self.gat2 = GATConv(
                in_feat_size=hidden_dim * 4,
                out_feat_size=hidden_dim // 2,
                num_heads=2,
                feat_drop=0.2,
                attn_drop=0.1
            )
            self.regressor = nn.SequentialCell([
                nn.Dense(hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Dense(32, 1)
            ])
        
        # ⚠️ SIGNATURE OBLIGATOIRE POUR MSGL
        def construct(self, x, g: MSGLGraph):
            x = self.gat1(x, g)
            x = ops.relu(x)
            x = self.gat2(x, g)
            x = ops.relu(x)
            return self.regressor(x)
else:
    # Fallback MLP si MSGL absent (CPU sans installation complète)
    class UrbanGNN(nn.Cell):
        def __init__(self, node_feat_size, hidden_dim=64):
            super().__init__()
            self.network = nn.SequentialCell([
                nn.Dense(node_feat_size, 256),
                nn.LayerNorm((256,)),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Dense(256, 128),
                nn.LayerNorm((128,)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Dense(128, 64),
                nn.ReLU(),
                nn.Dense(64, 1)
            ])
        def construct(self, x, g=None):  # g ignoré pour compatibilité API
            return self.network(x)


class ExplainableMLCore:
    """
    Moteur ML 100% MindSpore avec GNN (signature MSGL corrigée).
    
    CORRECTIONS CRITIQUES :
      ✅ Signature construct(self, x, g: Graph) conforme MSGL
      ✅ Création objet Graph MSGL dans _osmnx_to_mindspore_graph
      ✅ Alignement robuste X/y index
      ✅ Fallback silencieux MLP si MSGL échoue sur CPU
    """
    def __init__(self, graph=None):
        self.graph = graph
        self.model = None
        self.scaler = RobustScaler()
        self.feature_importance = None
        self.node_to_idx = {}
        self.idx_to_node = {}
        self.device_target = DEVICE_TARGET
        self.use_gnn = MSGL_AVAILABLE  # Tracking pour logs explicites
        
    def train(self, X_df, y_series, graph=None):
        logging.info("=" * 80)
        logging.info("PHASE 4: MACHINE LEARNING WITH MINDSPORE")
        logging.info("=" * 80)
        logging.info(f"  [HARDWARE] Device: {self.device_target} | Mode: {'GRAPH' if CONTEXT_MODE == ms.GRAPH_MODE else 'PYNATIVE'}")
        logging.info(f"  [ARCHITECTURE] {'Graph Neural Network (GNN)' if self.use_gnn else 'MLP Fallback'}")
        
        G = graph if graph is not None else self.graph
        if G is None:
            raise ValueError("Graphe OSMnx requis")
        
        # 🔧 CORRECTION INDEX : Alignement X/y
        if not isinstance(y_series, pd.Series):
            y_series = pd.Series(y_series)
        if not y_series.index.equals(X_df.index):
            logging.warning(f"  ⚠️  Index mismatch - reindexing y to match X.index")
            y_series = y_series.reindex(X_df.index, fill_value=0.0)
        
        # Préprocessing
        X = X_df.fillna(0).astype(np.float32)
        y = y_series.fillna(0).astype(np.float32)
        X_scaled = self.scaler.fit_transform(X).astype(np.float32)
        
        # 🔧 CORRECTION MSGL : Conversion graphe + création objet Graph
        logging.info("  Converting OSMnx graph to MindSpore GL format...")
        edge_index, node_feat, node_ids, num_nodes = self._osmnx_to_mindspore_graph(G, X_scaled, X_df.index)
        
        # Alignement final labels
        y_aligned = y.reindex(node_ids, fill_value=0.0)
        labels = Tensor(y_aligned.values.reshape(-1, 1), ms.float32)
        node_feat_tensor = Tensor(node_feat, ms.float32)
        
        # 🔧 CRÉATION OBJET GRAPH MSGL (obligatoire pour GNNCell)
        if self.use_gnn:
            try:
                edge_index_ms = Tensor(edge_index, ms.int32)
                g = MSGLGraph(edge_index_ms, None, num_nodes)
                logging.info(f"  Created MSGL Graph | Nodes: {num_nodes} | Edges: {edge_index.shape[1]}")
            except Exception as e:
                logging.warning(f"  ⚠️  MSGL Graph creation failed: {e}. Falling back to MLP.")
                self.use_gnn = False
                g = None
        else:
            g = None
        
        # Création dataset adaptatif
        if self.use_gnn:
            dataset = self._create_gnn_dataset(node_feat_tensor, g, labels)
        else:
            dataset = self._create_mlp_dataset(node_feat_tensor, labels)
        
        # Initialisation modèle
        self.model = UrbanGNN(node_feat_size=node_feat.shape[1])
        param_count = sum(p.size for p in self.model.get_parameters())
        arch_type = "GNN" if self.use_gnn else "MLP"
        logging.info(f"  Initialized {arch_type} | Input dim: {node_feat.shape[1]} | Params: {param_count:,}")
        
        # Entraînement
        loss_fn = MSELoss()
        optimizer = Adam(self.model.trainable_params(), learning_rate=0.005)
        model_wrapper = Model(network=self.model, loss_fn=loss_fn, optimizer=optimizer, metrics={"mae": nn.MAE()})
        
        logging.info(f"  Training {arch_type} (100 epochs) on {self.device_target}...")
        model_wrapper.train(
            epoch=100,
            train_dataset=dataset,
            callbacks=[LossMonitor(per_print_times=20), TimeMonitor(data_size=64 if not self.use_gnn else 1)],
            dataset_sink_mode=DATASET_SINK and self.use_gnn  # Sink uniquement pour GNN sur Ascend
        )
        
        # Évaluation
        if self.use_gnn:
            train_pred = self.model(node_feat_tensor, g).asnumpy().flatten()
        else:
            train_pred = self.model(node_feat_tensor, None).asnumpy().flatten()
        
        train_r2 = self._compute_r2(y_aligned.values, train_pred)
        train_mae = np.mean(np.abs(y_aligned.values - train_pred))
        
        logging.info(f"   Training R²: {train_r2:.4f}")
        logging.info(f"   Training MAE: {train_mae:.4f}")
        
        # Explicabilité adaptative
        if self.use_gnn:
            self._compute_feature_importance_gnn(node_feat_tensor, g, node_ids, X_df.columns)
        else:
            self._compute_feature_importance_mlp(node_feat_tensor, X_df.columns)
        
        # Sauvegarde
        suffix = "gnn_ascend" if self.device_target == "Ascend" and self.use_gnn else \
                 "gnn_cpu" if self.use_gnn else "mlp_cpu"
        ckpt_path = os.path.join(CONFIG["OUTPUT_DIR"], f"mindspore_model_{suffix}.ckpt")
        save_checkpoint(self.model, ckpt_path)
        logging.info(f"  Model saved to: {ckpt_path}")
        
        return self.model
    
    def predict(self, X_df=None, graph=None):
        if self.model is None:
            raise RuntimeError("Modèle non entraîné")
        
        G = graph if graph is not None else self.graph
        if G is None:
            raise ValueError("Graphe requis")
        
        if X_df is not None:
            X = X_df.fillna(0).astype(np.float32)
            X_scaled = self.scaler.transform(X).astype(np.float32)
            _, node_feat, node_ids, num_nodes = self._osmnx_to_mindspore_graph(G, X_scaled, X_df.index)
        else:
            from FeatureEngineer import FeatureEngineer
            feat_eng = FeatureEngineer(G)
            X_df = feat_eng.extract()
            X = X_df.fillna(0).astype(np.float32)
            X_scaled = self.scaler.transform(X).astype(np.float32)
            _, node_feat, node_ids, num_nodes = self._osmnx_to_mindspore_graph(G, X_scaled, X_df.index)
        
        node_feat_tensor = Tensor(node_feat, ms.float32)
        
        if self.use_gnn:
            edge_index_ms = Tensor(self._extract_edge_index(G, node_ids), ms.int32)
            g = MSGLGraph(edge_index_ms, None, num_nodes)
            preds = self.model(node_feat_tensor, g).asnumpy().flatten()
        else:
            preds = self.model(node_feat_tensor, None).asnumpy().flatten()
        
        return pd.Series(preds, index=node_ids)
    
    def _osmnx_to_mindspore_graph(self, G, node_features, node_index):
        """Conversion robuste + retourne num_nodes pour MSGL Graph"""
        node_ids = list(node_index)
        self.node_to_idx = {node: idx for idx, node in enumerate(node_ids)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        edges = []
        for u, v in G.edges(keys=False):
            u_int = int(u) if not isinstance(u, int) else u
            v_int = int(v) if not isinstance(v, int) else v
            if u_int in self.node_to_idx and v_int in self.node_to_idx:
                edges.append([self.node_to_idx[u_int], self.node_to_idx[v_int]])
        
        edge_index = np.array(edges).T if edges else np.zeros((2, 0), dtype=np.int32)
        
        if edge_index.shape[1] == 0:
            logging.warning("  ⚠️  No valid edges - creating minimal graph")
            edge_index = np.array([[0, 1], [1, 0]], dtype=np.int32)
        
        return edge_index, node_features, node_ids, len(node_ids)  # ← num_nodes ajouté
    
    def _extract_edge_index(self, G, node_ids):
        edges = []
        node_set = set(node_ids)
        for u, v in G.edges(keys=False):
            u_int = int(u) if not isinstance(u, int) else u
            v_int = int(v) if not isinstance(v, int) else v
            if u_int in node_set and v_int in node_set:
                edges.append([self.node_to_idx[u_int], self.node_to_idx[v_int]])
        return np.array(edges).T if edges else np.zeros((2, 0), dtype=np.int32)
    
    def _create_gnn_dataset(self, node_feat, g, labels):
        """Dataset pour GNN avec objet Graph MSGL"""
        class GNNDataset:
            def __init__(self, node_feat, g, labels):
                self.node_feat = node_feat
                self.g = g
                self.labels = labels
            def __getitem__(self, index):
                return self.node_feat, self.g, self.labels
            def __len__(self):
                return 1
        
        return GeneratorDataset(
            source=GNNDataset(node_feat, g, labels),
            column_names=["x", "g", "label"],  # ← "g" au lieu de "edge_index"
            shuffle=False
        ).batch(1)
    
    def _create_mlp_dataset(self, node_feat, labels):
        """Dataset fallback MLP"""
        return NumpySlicesDataset(
            (node_feat.asnumpy(), labels.asnumpy()),
            column_names=["data", "label"],
            shuffle=True
        ).batch(64)
    
    def _compute_r2(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    
    def _compute_feature_importance_gnn(self, node_feat, g, node_ids, feature_names):
        logging.info("  Computing feature importance via Integrated Gradients (GNN)...")
        baseline = ops.zeros_like(node_feat)
        steps = 50
        attributions = np.zeros((node_feat.shape[0], node_feat.shape[1]))
        
        for i in range(1, steps + 1):
            alpha = i / steps
            interpolated = baseline + alpha * (node_feat - baseline)
            grad_fn = ops.value_and_grad(self.model, grad_position=0)
            _, grads = grad_fn(interpolated, g)
            attributions += grads.asnumpy()
        
        attributions /= steps
        mean_attribution = np.mean(np.abs(attributions), axis=0)
        
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_attribution
        }).sort_values('importance', ascending=False)
        
        logging.info("\n  Top 3 Features (GNN Integrated Gradients):")
        for idx, row in self.feature_importance.head(3).iterrows():
            logging.info(f"    {row['feature']:20s} | {row['importance']:.4f}")
        
        imp_path = os.path.join(CONFIG["OUTPUT_DIR"], "feature_importance.csv")
        self.feature_importance.to_csv(imp_path, index=False)
        logging.info(f"  Feature importance saved to: {imp_path}")
    
    def _compute_feature_importance_mlp(self, X_tensor, feature_names):
        logging.info("  Computing feature importance via gradient approximation (MLP)...")
        grad_fn = ops.value_and_grad(self.model, grad_position=0)
        _, grads = grad_fn(X_tensor, None)
        mean_abs_grad = np.mean(np.abs(grads.asnumpy()), axis=0)
        
        if np.sum(mean_abs_grad) > 0:
            mean_abs_grad = mean_abs_grad / np.sum(mean_abs_grad)
        
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_grad
        }).sort_values('importance', ascending=False)
        
        logging.info("\n  Top 3 Features (MLP Gradient Approximation):")
        for idx, row in self.feature_importance.head(3).iterrows():
            logging.info(f"    {row['feature']:20s} | {row['importance']:.4f}")
        
        imp_path = os.path.join(CONFIG["OUTPUT_DIR"], "feature_importance.csv")
        self.feature_importance.to_csv(imp_path, index=False)
        logging.info(f"  Feature importance saved to: {imp_path}")