"""
Modelo de Datos Simplificado para Pares de Genes
Ya tienes gene_symbol en ambos datasets - proceso directo
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenePairDataProcessor:
    """
    Procesador de datos simplificado para pares de genes
    Asume que ya tienes gene_symbol en ambos datasets
    """
    
    def __init__(self, cache_dir: str = "./gene_data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Datos procesados
        self.features_df = None
        self.targets_df = None
        self.gene_pairs = None
        self.scaler = StandardScaler()
        
        # Tensores
        self.tensors = {}
        self.data_loaders = {}
        
        # Metadata
        self.feature_names = []
        self.target_names = []
        self.stats = {}
        
        logger.info(f"🧬 Procesador de datos inicializado")
        logger.info(f"📁 Cache directory: {cache_dir}")
    
    def load_and_process(self, 
                        interactions_file: str, 
                        individual_scores_file: str,
                        target_strategies: List[str] = None,
                        interactions_separator: str = '\t',
                        individual_separator: str = ',') -> 'GenePairDataProcessor':
        """
        Cargar y procesar datos en un solo paso
        
        Args:
            interactions_file: TXT/CSV con Gene1, Gene2, AA_Score, etc.
            individual_scores_file: CSV con gene_symbol, development_scores, etc.
            target_strategies: ['average', 'maximum', 'synergy', 'multiplicative']
            interactions_separator: Separador para interactions_file ('\t' para TXT)
            individual_separator: Separador para individual_scores_file (',' para CSV)
        """
        
        if target_strategies is None:
            target_strategies = ['average', 'synergy']
        
        logger.info("🔄 Cargando datasets...")
        logger.info(f"   📄 Interactions: {interactions_file} (separador: '{interactions_separator}')")
        logger.info(f"   📄 Individual scores: {individual_scores_file} (separador: '{individual_separator}')")
        
        # Cargar datasets con separadores específicos
        try:
            interactions_df = pd.read_csv(interactions_file, sep=interactions_separator)
            logger.info(f"   ✅ Interactions cargado: {interactions_df.shape}")
            logger.info(f"   📋 Columnas interactions: {list(interactions_df.columns)}")
        except Exception as e:
            logger.error(f"   ❌ Error cargando interactions: {e}")
            raise
        
        try:
            individual_scores_df = pd.read_csv(individual_scores_file, sep=individual_separator)
            logger.info(f"   ✅ Individual scores cargado: {individual_scores_df.shape}")
            logger.info(f"   📋 Columnas individual: {list(individual_scores_df.columns)}")
        except Exception as e:
            logger.error(f"   ❌ Error cargando individual scores: {e}")
            raise
        
        logger.info(f"   📊 Interacciones: {len(interactions_df)} filas")
        logger.info(f"   📊 Genes individuales: {len(individual_scores_df)} filas")
        logger.info(f"   📊 Genes únicos en individual: {individual_scores_df['gene_symbol'].nunique()}")
        
        # Verificar genes en común
        genes_in_interactions = set(interactions_df['Gene1'].tolist() + interactions_df['Gene2'].tolist())
        genes_in_individual = set(individual_scores_df['gene_symbol'].tolist())
        common_genes = genes_in_interactions & genes_in_individual
        
        logger.info(f"   🔗 Genes en común: {len(common_genes)}")
        
        if len(common_genes) < 10:
            logger.warning("⚠️  Muy pocos genes en común. Verifica los nombres de genes.")
        
        # Procesar datos
        self._create_features_and_targets(interactions_df, individual_scores_df, target_strategies)
        
        # Estadísticas
        self.stats = {
            'n_interactions_original': len(interactions_df),
            'n_individual_genes': len(individual_scores_df),
            'n_common_genes': len(common_genes),
            'n_valid_pairs': len(self.gene_pairs),
            'n_features': len(self.feature_names),
            'n_targets': len(self.target_names),
            'coverage_rate': len(self.gene_pairs) / len(interactions_df) * 100
        }
        
        logger.info(f"✅ Procesamiento completo:")
        logger.info(f"   🎯 Pares válidos: {self.stats['n_valid_pairs']} ({self.stats['coverage_rate']:.1f}%)")
        logger.info(f"   📈 Features: {self.stats['n_features']}")
        logger.info(f"   🎯 Targets: {self.stats['n_targets']}")
        
    def _save_processed_csvs(self, X_scaled_df: pd.DataFrame, csv_prefix: str,
                           train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray):
        """
        Guardar datos procesados en CSVs
        """
        
        logger.info("💾 Guardando CSVs procesados...")
        
        # Crear DataFrame completo con toda la información
        complete_df = pd.DataFrame()
        
        # Agregar información de genes
        complete_df['Gene1'] = [pair[0] for pair in self.gene_pairs]
        complete_df['Gene2'] = [pair[1] for pair in self.gene_pairs]
        complete_df['gene_pair_id'] = [f"{pair[0]}_{pair[1]}" for pair in self.gene_pairs]
        
        # Agregar features escalados
        for col in X_scaled_df.columns:
            complete_df[f"feature_{col}"] = X_scaled_df[col].values
        
        # Agregar targets
        for col in self.targets_df.columns:
            complete_df[f"target_{col}"] = self.targets_df[col].values
        
        # Agregar información del split
        split_labels = [''] * len(complete_df)
        for i in train_idx:
            split_labels[i] = 'train'
        for i in val_idx:
            split_labels[i] = 'val'
        for i in test_idx:
            split_labels[i] = 'test'
        complete_df['data_split'] = split_labels
        
        # Agregar índices
        complete_df['original_index'] = range(len(complete_df))
        
        # Guardar CSV completo
        complete_file = self.cache_dir / f"{csv_prefix}_complete.csv"
        complete_df.to_csv(complete_file, index=False)
        logger.info(f"   📄 Datos completos: {complete_file}")
        
        # Guardar CSVs por split
        for split_name, indices in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
            split_df = complete_df.iloc[indices].copy()
            split_file = self.cache_dir / f"{csv_prefix}_{split_name}.csv"
            split_df.to_csv(split_file, index=False)
            logger.info(f"   📄 {split_name.capitalize()}: {split_file} ({len(split_df)} filas)")
        
        # Guardar CSV solo con features (para modelos externos)
        features_df = complete_df.copy()
        feature_cols = [col for col in features_df.columns if col.startswith('feature_')]
        target_cols = [col for col in features_df.columns if col.startswith('target_')]
        
        ml_ready_df = features_df[['Gene1', 'Gene2', 'gene_pair_id', 'data_split'] + feature_cols + target_cols]
        ml_file = self.cache_dir / f"{csv_prefix}_ml_ready.csv"
        ml_ready_df.to_csv(ml_file, index=False)
        logger.info(f"   📄 ML Ready: {ml_file}")
        
        # Guardar CSV con features originales (sin escalar)
        original_complete_df = pd.DataFrame()
        original_complete_df['Gene1'] = [pair[0] for pair in self.gene_pairs]
        original_complete_df['Gene2'] = [pair[1] for pair in self.gene_pairs]
        original_complete_df['gene_pair_id'] = [f"{pair[0]}_{pair[1]}" for pair in self.gene_pairs]
        
        # Features originales
        for col in self.features_df.columns:
            original_complete_df[f"feature_{col}"] = self.features_df[col].values
        
        # Targets
        for col in self.targets_df.columns:
            original_complete_df[f"target_{col}"] = self.targets_df[col].values
        
        original_complete_df['data_split'] = split_labels
        
        original_file = self.cache_dir / f"{csv_prefix}_original_features.csv"
        original_complete_df.to_csv(original_file, index=False)
        logger.info(f"   📄 Features originales: {original_file}")
        
        # Crear metadata CSV
        metadata_df = pd.DataFrame({
            'metric': [
                'total_pairs', 'train_pairs', 'val_pairs', 'test_pairs',
                'n_features', 'n_targets', 'coverage_rate',
                'train_ratio', 'val_ratio', 'test_ratio'
            ],
            'value': [
                len(complete_df), len(train_idx), len(val_idx), len(test_idx),
                len(self.feature_names), len(self.target_names), self.stats.get('coverage_rate', 0),
                len(train_idx)/len(complete_df), len(val_idx)/len(complete_df), len(test_idx)/len(complete_df)
            ]
        })
        
        metadata_file = self.cache_dir / f"{csv_prefix}_metadata.csv"
        metadata_df.to_csv(metadata_file, index=False)
        logger.info(f"   📄 Metadata: {metadata_file}")
        
        logger.info(f"✅ Todos los CSVs guardados en: {self.cache_dir}")
    
    def export_analysis_csvs(self, prefix: str = 'gene_analysis') -> Dict[str, str]:
        """
        Exportar CSVs adicionales para análisis exploratorio
        """
        
        if self.features_df is None:
            raise ValueError("Primero ejecuta load_and_process()")
        
        logger.info("📊 Creando CSVs para análisis exploratorio...")
        
        exported_files = {}
        
        # 1. Resumen estadístico de features
        features_stats = self.features_df.describe()
        features_stats_file = self.cache_dir / f"{prefix}_features_statistics.csv"
        features_stats.to_csv(features_stats_file)
        exported_files['features_stats'] = str(features_stats_file)
        
        # 2. Resumen estadístico de targets
        targets_stats = self.targets_df.describe()
        targets_stats_file = self.cache_dir / f"{prefix}_targets_statistics.csv"
        targets_stats.to_csv(targets_stats_file)
        exported_files['targets_stats'] = str(targets_stats_file)
        
        # 3. Matriz de correlación de features
        features_corr = self.features_df.corr()
        features_corr_file = self.cache_dir / f"{prefix}_features_correlation.csv"
        features_corr.to_csv(features_corr_file)
        exported_files['features_corr'] = str(features_corr_file)
        
        # 4. Matriz de correlación de targets
        targets_corr = self.targets_df.corr()
        targets_corr_file = self.cache_dir / f"{prefix}_targets_correlation.csv"
        targets_corr.to_csv(targets_corr_file)
        exported_files['targets_corr'] = str(targets_corr_file)
        
        # 5. Top gene pairs por cada target
        top_pairs_data = []
        for target_col in self.target_names:
            target_values = self.targets_df[target_col]
            top_indices = target_values.nlargest(20).index
            
            for idx in top_indices:
                gene_pair = self.gene_pairs[idx]
                top_pairs_data.append({
                    'target_type': target_col,
                    'gene1': gene_pair[0],
                    'gene2': gene_pair[1],
                    'gene_pair_id': f"{gene_pair[0]}_{gene_pair[1]}",
                    'target_value': target_values.iloc[idx],
                    'rank': len(top_pairs_data) % 20 + 1
                })
        
        top_pairs_df = pd.DataFrame(top_pairs_data)
        top_pairs_file = self.cache_dir / f"{prefix}_top_gene_pairs.csv"
        top_pairs_df.to_csv(top_pairs_file, index=False)
        exported_files['top_pairs'] = str(top_pairs_file)
        
        # 6. Gene frequency analysis
        gene_freq_data = []
        all_genes = []
        
        for pair in self.gene_pairs:
            all_genes.extend(pair)
        
        gene_counts = pd.Series(all_genes).value_counts()
        
        for gene, count in gene_counts.items():
            # Calcular estadísticas promedio para este gen
            gene_targets = []
            for i, pair in enumerate(self.gene_pairs):
                if gene in pair:
                    gene_targets.append(self.targets_df.iloc[i]['avg_combined_score'] 
                                      if 'avg_combined_score' in self.targets_df.columns 
                                      else self.targets_df.iloc[i].iloc[0])
            
            gene_freq_data.append({
                'gene': gene,
                'interaction_count': count,
                'avg_target_score': np.mean(gene_targets) if gene_targets else 0,
                'max_target_score': np.max(gene_targets) if gene_targets else 0,
                'std_target_score': np.std(gene_targets) if gene_targets else 0
            })
        
        gene_freq_df = pd.DataFrame(gene_freq_data)
        gene_freq_file = self.cache_dir / f"{prefix}_gene_frequency_analysis.csv"
        gene_freq_df.to_csv(gene_freq_file, index=False)
        exported_files['gene_frequency'] = str(gene_freq_file)
        
        logger.info(f"✅ CSVs de análisis exportados:")
        for name, filepath in exported_files.items():
            logger.info(f"   📄 {name}: {filepath}")
        
        return exported_files
    
    def _create_features_and_targets(self, interactions_df: pd.DataFrame, 
                                   individual_scores_df: pd.DataFrame,
                                   target_strategies: List[str]):
        """
        Crear features y targets combinando ambos datasets
        """
        
        logger.info("🏗️  Creando features y targets...")
        
        features_list = []
        targets_list = []
        gene_pairs_list = []
        
        # Crear lookup dict para scores individuales (más eficiente)
        individual_lookup = {}
        for _, row in individual_scores_df.iterrows():
            gene = row['gene_symbol']
            individual_lookup[gene] = {
                'immunology': row.get('development_score_log_norm_immunology', 0.0),
                'general': row.get('development_score_log_norm', 0.0),
                'companies_immunology': row.get('companies_num_launched_immunology', 0),
                'companies_general': row.get('companies_num_launched', 0)
            }
        
        # Procesar cada interacción
        for _, row in interactions_df.iterrows():
            gene1 = row['Gene1']
            gene2 = row['Gene2']
            
            # Verificar que ambos genes tengan scores individuales
            if gene1 not in individual_lookup or gene2 not in individual_lookup:
                continue
            
            # === CREAR FEATURES ===
            features = self._extract_interaction_features(row)
            
            # Añadir features derivados de genes individuales
            g1_scores = individual_lookup[gene1]
            g2_scores = individual_lookup[gene2]
            
            individual_features = {
                'gene1_immunology_score': g1_scores['immunology'],
                'gene2_immunology_score': g2_scores['immunology'],
                'gene1_general_score': g1_scores['general'],
                'gene2_general_score': g2_scores['general'],
                
                # Features derivados
                'genes_immunology_diff': abs(g1_scores['immunology'] - g2_scores['immunology']),
                'genes_general_diff': abs(g1_scores['general'] - g2_scores['general']),
                'genes_avg_immunology': (g1_scores['immunology'] + g2_scores['immunology']) / 2,
                'genes_avg_general': (g1_scores['general'] + g2_scores['general']) / 2,
                'genes_min_immunology': min(g1_scores['immunology'], g2_scores['immunology']),
                'genes_max_immunology': max(g1_scores['immunology'], g2_scores['immunology']),
                
                # Indicadores
                'both_high_immunology': int(g1_scores['immunology'] > 0.7 and g2_scores['immunology'] > 0.7),
                'both_low_immunology': int(g1_scores['immunology'] < 0.3 and g2_scores['immunology'] < 0.3),
                'complementary_specialization': abs(g1_scores['immunology'] - g1_scores['general']) + 
                                               abs(g2_scores['immunology'] - g2_scores['general'])
            }
            
            # Combinar features
            all_features = {**features, **individual_features}
            
            # === CREAR TARGETS ===
            targets = self._create_targets_for_strategies(g1_scores, g2_scores, target_strategies)
            
            # Guardar si tenemos datos válidos
            if any(v > 0 for v in targets.values()):
                features_list.append(all_features)
                targets_list.append(targets)
                gene_pairs_list.append((gene1, gene2))
        
        # Convertir a DataFrames
        self.features_df = pd.DataFrame(features_list)
        self.targets_df = pd.DataFrame(targets_list)
        self.gene_pairs = gene_pairs_list
        
        # Limpiar datos
        self.features_df = self.features_df.fillna(0)
        self.targets_df = self.targets_df.fillna(0)
        
        # Guardar nombres
        self.feature_names = list(self.features_df.columns)
        self.target_names = list(self.targets_df.columns)
        
        logger.info(f"   ✅ Features creados: {self.feature_names}")
        logger.info(f"   ✅ Targets creados: {self.target_names}")
    
    def _extract_interaction_features(self, row: pd.Series) -> Dict[str, float]:
        """
        Extraer features de interacción de una fila
        """
        
        features = {
            # Features principales
            'AA_Score': row.get('AA_Score', 0.0),
            'Genetics': row.get('Genetics', 0.0),
            'Knowledge_Graph_Connectivity': row.get('Knowledge_Graph_Connectivity', 0.0),
            'Credentialing': row.get('Credentialing', 0.0),
            'Single_Cell_Expression': row.get('Single_Cell_Expression', 0.0),
            'Single_Cell_Specificity': row.get('Single_Cell_Specificity', 0.0),
            'SC_KO_positive': row.get('SC_KO_positive', 0.0),
            'SC_KO_Negative': row.get('SC_KO_Negative', 0.0),
            'Ligand_Receptor_Significance': row.get('Ligand_Receptor_Significance', 0.0),
            
            # Features derivados de interacciones
            'Genetics_x_Credentialing': row.get('Genetics', 0.0) * row.get('Credentialing', 0.0),
            'AA_Score_x_Genetics': row.get('AA_Score', 0.0) * row.get('Genetics', 0.0),
            'KO_Net_Effect': row.get('SC_KO_positive', 0.0) - row.get('SC_KO_Negative', 0.0),
            'Interaction_Quality': ((row.get('AA_Score', 0.0) * 
                                   row.get('Genetics', 0.0) * 
                                   row.get('Credentialing', 0.0)) + 1e-6) ** (1/3),
            'Validation_Score': (row.get('SC_KO_positive', 0.0) - row.get('SC_KO_Negative', 0.0) + 1) / 2
        }
        
        return features
    
    def _create_targets_for_strategies(self, g1_scores: Dict, g2_scores: Dict, 
                                     strategies: List[str]) -> Dict[str, float]:
        """
        Crear targets según diferentes estrategias
        """
        
        targets = {}
        
        for strategy in strategies:
            if strategy == 'average':
                targets.update({
                    'avg_immunology_score': (g1_scores['immunology'] + g2_scores['immunology']) / 2,
                    'avg_general_score': (g1_scores['general'] + g2_scores['general']) / 2,
                    'avg_combined_score': (g1_scores['immunology'] + g1_scores['general'] + 
                                         g2_scores['immunology'] + g2_scores['general']) / 4
                })
            
            elif strategy == 'maximum':
                targets.update({
                    'max_immunology_score': max(g1_scores['immunology'], g2_scores['immunology']),
                    'max_general_score': max(g1_scores['general'], g2_scores['general'])
                })
            
            elif strategy == 'synergy':
                avg_immuno = (g1_scores['immunology'] + g2_scores['immunology']) / 2
                max_immuno = max(g1_scores['immunology'], g2_scores['immunology'])
                avg_general = (g1_scores['general'] + g2_scores['general']) / 2
                max_general = max(g1_scores['general'], g2_scores['general'])
                
                targets.update({
                    'synergy_immunology': max(0, avg_immuno - max_immuno),  # Solo sinergia positiva
                    'synergy_general': max(0, avg_general - max_general)
                })
            
            elif strategy == 'multiplicative':
                targets.update({
                    'multiplicative_potential': g1_scores['immunology'] * g1_scores['general'] * 
                                              g2_scores['immunology'] * g2_scores['general']
                })
            
            elif strategy == 'complementary':
                # Genes que se complementan (uno fuerte en inmuno, otro en general)
                targets.update({
                    'complementary_score': (min(g1_scores['immunology'], g2_scores['general']) + 
                                          min(g1_scores['general'], g2_scores['immunology'])) / 2
                })
        
        return targets
    
    def create_tensors(self, train_size: float = 0.7, val_size: float = 0.15, 
                      random_state: int = 42, device: str = 'cpu',
                      save_csvs: bool = True, csv_prefix: str = 'gene_pair_data') -> 'GenePairDataProcessor':
        """
        Crear tensores PyTorch con train/val/test splits y guardar CSVs
        """
        
        if self.features_df is None:
            raise ValueError("Primero ejecuta load_and_process()")
        
        logger.info("🔢 Creando tensores PyTorch y CSVs...")
        
        # Escalar features
        X = self.scaler.fit_transform(self.features_df.values)
        y = self.targets_df.values
        
        # Crear DataFrame completo con features escalados
        X_scaled_df = pd.DataFrame(X, columns=self.feature_names)
        
        # Splits
        test_size = 1 - train_size - val_size
        
        # Crear índices para splits
        indices = np.arange(len(X))
        
        # Train/temp split
        train_idx, temp_idx = train_test_split(
            indices, train_size=train_size, random_state=random_state
        )
        
        # Val/test split
        val_ratio = val_size / (val_size + test_size)
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=val_ratio, random_state=random_state
        )
        
        # Crear tensores
        self.tensors = {
            'X_train': torch.tensor(X[train_idx], dtype=torch.float32, device=device),
            'y_train': torch.tensor(y[train_idx], dtype=torch.float32, device=device),
            'X_val': torch.tensor(X[val_idx], dtype=torch.float32, device=device),
            'y_val': torch.tensor(y[val_idx], dtype=torch.float32, device=device),
            'X_test': torch.tensor(X[test_idx], dtype=torch.float32, device=device),
            'y_test': torch.tensor(y[test_idx], dtype=torch.float32, device=device),
            'train_indices': train_idx,
            'val_indices': val_idx,
            'test_indices': test_idx,
            'gene_pairs_train': [self.gene_pairs[i] for i in train_idx],
            'gene_pairs_val': [self.gene_pairs[i] for i in val_idx],
            'gene_pairs_test': [self.gene_pairs[i] for i in test_idx]
        }
        
        # Guardar CSVs si se requiere
        if save_csvs:
            self._save_processed_csvs(X_scaled_df, csv_prefix, train_idx, val_idx, test_idx)
        
        logger.info(f"✅ Tensores creados:")
        logger.info(f"   🏋️  Train: {self.tensors['X_train'].shape} → {self.tensors['y_train'].shape}")
        logger.info(f"   🔍 Val: {self.tensors['X_val'].shape} → {self.tensors['y_val'].shape}")
        logger.info(f"   🧪 Test: {self.tensors['X_test'].shape} → {self.tensors['y_test'].shape}")
        logger.info(f"   🖥️  Device: {device}")
        
        return self
    
    def get_data_loaders(self, batch_size: int = 32, shuffle_train: bool = True) -> Dict[str, DataLoader]:
        """
        Crear DataLoaders para entrenamiento
        """
        
        if not self.tensors:
            raise ValueError("Primero ejecuta create_tensors()")
        
        # Crear datasets
        train_dataset = TensorDataset(self.tensors['X_train'], self.tensors['y_train'])
        val_dataset = TensorDataset(self.tensors['X_val'], self.tensors['y_val'])
        test_dataset = TensorDataset(self.tensors['X_test'], self.tensors['y_test'])
        
        # Crear loaders
        self.data_loaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        }
        
        logger.info(f"📦 DataLoaders creados (batch_size={batch_size})")
        
        return self.data_loaders
    
    def get_model_specs(self) -> Dict:
        """
        Obtener especificaciones para crear modelos
        """
        
        if not self.tensors:
            raise ValueError("Primero ejecuta create_tensors()")
        
        return {
            'input_size': self.tensors['X_train'].shape[1],
            'output_size': self.tensors['y_train'].shape[1],
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'n_samples': {
                'train': len(self.tensors['X_train']),
                'val': len(self.tensors['X_val']),
                'test': len(self.tensors['X_test'])
            },
            'stats': self.stats
        }
    
    def predict_new_pair(self, gene1: str, gene2: str, 
                        individual_scores_df: pd.DataFrame,
                        interaction_features: Dict[str, float]) -> torch.Tensor:
        """
        Preparar tensor para predecir un nuevo par de genes
        """
        
        if not self.tensors:
            raise ValueError("Primero ejecuta create_tensors()")
        
        # Crear lookup para scores individuales
        individual_lookup = {}
        for _, row in individual_scores_df.iterrows():
            gene = row['gene_symbol']
            individual_lookup[gene] = {
                'immunology': row.get('development_score_log_norm_immunology', 0.0),
                'general': row.get('development_score_log_norm', 0.0)
            }
        
        if gene1 not in individual_lookup or gene2 not in individual_lookup:
            raise ValueError(f"Genes {gene1} o {gene2} no encontrados en individual_scores")
        
        # Crear features igual que en entrenamiento
        features = self._extract_interaction_features(pd.Series(interaction_features))
        
        g1_scores = individual_lookup[gene1]
        g2_scores = individual_lookup[gene2]
        
        individual_features = {
            'gene1_immunology_score': g1_scores['immunology'],
            'gene2_immunology_score': g2_scores['immunology'],
            'gene1_general_score': g1_scores['general'],
            'gene2_general_score': g2_scores['general'],
            'genes_immunology_diff': abs(g1_scores['immunology'] - g2_scores['immunology']),
            'genes_general_diff': abs(g1_scores['general'] - g2_scores['general']),
            'genes_avg_immunology': (g1_scores['immunology'] + g2_scores['immunology']) / 2,
            'genes_avg_general': (g1_scores['general'] + g2_scores['general']) / 2,
            'genes_min_immunology': min(g1_scores['immunology'], g2_scores['immunology']),
            'genes_max_immunology': max(g1_scores['immunology'], g2_scores['immunology']),
            'both_high_immunology': int(g1_scores['immunology'] > 0.7 and g2_scores['immunology'] > 0.7),
            'both_low_immunology': int(g1_scores['immunology'] < 0.3 and g2_scores['immunology'] < 0.3),
            'complementary_specialization': abs(g1_scores['immunology'] - g1_scores['general']) + 
                                           abs(g2_scores['immunology'] - g2_scores['general'])
        }
        
        all_features = {**features, **individual_features}
        
        # Ordenar features igual que en entrenamiento
        feature_values = [all_features.get(name, 0.0) for name in self.feature_names]
        
        # Escalar y convertir a tensor
        X_scaled = self.scaler.transform([feature_values])
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.tensors['X_train'].device)
        
        return X_tensor
    
    def save_state(self, filepath: str = None):
        """Guardar estado completo"""
        
        if filepath is None:
            filepath = self.cache_dir / "gene_data_processor.pkl"
        
        state = {
            'features_df': self.features_df,
            'targets_df': self.targets_df,
            'gene_pairs': self.gene_pairs,
            'scaler': self.scaler,
            'tensors': self.tensors,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'stats': self.stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"💾 Estado guardado: {filepath}")
    
    def load_state(self, filepath: str):
        """Cargar estado completo"""
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.features_df = state['features_df']
        self.targets_df = state['targets_df']
        self.gene_pairs = state['gene_pairs']
        self.scaler = state['scaler']
        self.tensors = state['tensors']
        self.feature_names = state['feature_names']
        self.target_names = state['target_names']
        self.stats = state['stats']
        
        logger.info(f"📁 Estado cargado: {filepath}")
    
    def get_summary(self) -> str:
        """Obtener resumen completo"""
        
        lines = [
            "🧬 RESUMEN DEL PROCESADOR DE DATOS DE GENES",
            "=" * 50,
            "",
            f"📊 Datos procesados:",
            f"   Pares válidos: {self.stats.get('n_valid_pairs', 'N/A')}",
            f"   Coverage rate: {self.stats.get('coverage_rate', 0):.1f}%",
            f"   Features: {self.stats.get('n_features', 'N/A')}",
            f"   Targets: {self.stats.get('n_targets', 'N/A')}",
            "",
        ]
        
        if self.tensors:
            lines.extend([
                f"🔢 Tensores:",
                f"   Train: {self.tensors['X_train'].shape} → {self.tensors['y_train'].shape}",
                f"   Val: {self.tensors['X_val'].shape} → {self.tensors['y_val'].shape}",
                f"   Test: {self.tensors['X_test'].shape} → {self.tensors['y_test'].shape}",
                f"   Device: {self.tensors['X_train'].device}",
                "",
            ])
        
        if self.feature_names:
            lines.extend([
                f"📈 Features principales:",
                f"   {', '.join(self.feature_names[:8])}{'...' if len(self.feature_names) > 8 else ''}",
                "",
            ])
        
        if self.target_names:
            lines.extend([
                f"🎯 Targets:",
                f"   {', '.join(self.target_names)}",
                "",
            ])
        
        return "\n".join(lines)

# ============================================================================
# EJEMPLO DE USO DIRECTO
# ============================================================================

def ejemplo_uso_completo():
    """
    Ejemplo de uso completo del procesador
    """
    
    print("🧬 PROCESADOR DE DATOS PARA PARES DE GENES")
    print("=" * 50)
    
    # 1. Inicializar procesador
    processor = GenePairDataProcessor()
    
    # 2. Cargar y procesar datos (reemplaza con tus archivos)
    processor.load_and_process(
        interactions_file='tu_interactions.csv',
        individual_scores_file='tu_individual_scores.csv',
        target_strategies=['average', 'synergy', 'multiplicative']
    )
    
    # 3. Crear tensores
    processor.create_tensors(
        train_size=0.7,
        val_size=0.15,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 4. Crear data loaders
    data_loaders = processor.get_data_loaders(batch_size=32)
    
    # 5. Obtener especificaciones para modelos
    model_specs = processor.get_model_specs()
    
    # 6. Mostrar resumen
    print(processor.get_summary())
    
    # 7. Guardar para uso futuro
    processor.save_state('mi_procesador_genes.pkl')
    
    print("✅ Procesamiento completo!")
    print(f"📊 Listo para entrenar modelos con:")
    print(f"   Input size: {model_specs['input_size']}")
    print(f"   Output size: {model_specs['output_size']}")
    
    return processor, data_loaders, model_specs

if __name__ == "__main__":
    # Ejecutar ejemplo
    
    processor = GenePairDataProcessor()

    processor.load_and_process(
        interactions_file='../cloud-data/its-cmo-darwin-magellan-workspaces-folders/WS_PMCB/BisCiT_Repository/results/current_version/v2.0c/aa/dtl-surfaceome-secretome-equal_0.1/results-with-evidence-full.txt',        # TXT
        individual_scores_file='GENS_DEVELOPMENT_SCORE.csv',         # CSV
        interactions_separator='\t',                    # Tab para TXT
        individual_separator=','                        # Coma para CSV
    )