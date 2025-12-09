#!/usr/bin/env python3
"""
🍄 Psychedelic LLM Explorer - MLX Edition
==========================================

Simule des états de conscience altérés dans un LLM en perturbant
les représentations internes, inspiré par la théorie de l'entropie
cérébrale de Carhart-Harris.

Usage:
    python psychedelic_llm_mlx.py
    
    # Ou en mode interactif:
    python psychedelic_llm_mlx.py --interactive

Requires: mlx, mlx-lm, matplotlib, numpy
    pip install mlx mlx-lm matplotlib numpy

Testé avec SmolLM3-3B-Base sur Apple Silicon.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Callable
import time
import os
import argparse

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TripReport:
    """Stocke les métriques d'une génération"""
    intensity: float
    prompt: str
    output: str
    hidden_state_entropies: List[float] = field(default_factory=list)
    attention_entropies: List[float] = field(default_factory=list)
    token_entropies: List[float] = field(default_factory=list)
    layer_entropies: Dict[int, List[float]] = field(default_factory=dict)
    generation_time: float = 0.0


# =============================================================================
# ENTROPY TRACKING
# =============================================================================

class EntropyTracker:
    """Collecte les métriques d'entropie pendant la génération"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.hidden_states: List[np.ndarray] = []
        self.attention_weights: List[np.ndarray] = []
        self.logits: List[np.ndarray] = []
        self.layer_activations: Dict[int, List[np.ndarray]] = {}
    
    def record_hidden_state(self, hs: mx.array, layer_idx: int = -1):
        """Enregistre un hidden state"""
        # Convertir en float32 car numpy ne supporte pas bfloat16
        arr = np.array(hs.astype(mx.float32))
        self.hidden_states.append(arr)

        if layer_idx >= 0:
            if layer_idx not in self.layer_activations:
                self.layer_activations[layer_idx] = []
            self.layer_activations[layer_idx].append(arr)

    def record_attention(self, attn: mx.array):
        """Enregistre les poids d'attention"""
        self.attention_weights.append(np.array(attn.astype(mx.float32)))

    def record_logits(self, logits: mx.array):
        """Enregistre les logits de sortie"""
        self.logits.append(np.array(logits.astype(mx.float32)))
    
    @staticmethod
    def compute_entropy(probs: np.ndarray, axis=-1) -> np.ndarray:
        """Calcule l'entropie de Shannon"""
        probs = np.clip(probs, 1e-10, 1.0)
        return -np.sum(probs * np.log2(probs + 1e-10), axis=axis)
    
    @staticmethod
    def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
        """Softmax numériquement stable"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def get_hidden_state_entropies(self) -> List[float]:
        """Entropie des activations (via softmax normalization)"""
        entropies = []
        for hs in self.hidden_states:
            # On prend un échantillon pour éviter les calculs trop lourds
            if hs.ndim >= 2:
                sample = hs.reshape(-1, hs.shape[-1])[:100]
                probs = self.softmax(sample.astype(np.float64))
                ent = self.compute_entropy(probs).mean()
                entropies.append(float(ent))
        return entropies if entropies else [0.0]
    
    def get_attention_entropies(self) -> List[float]:
        """Entropie des patterns d'attention"""
        entropies = []
        for attn in self.attention_weights:
            if attn.size > 0:
                # Attention est déjà une distribution
                ent = self.compute_entropy(attn.astype(np.float64)).mean()
                entropies.append(float(ent))
        return entropies if entropies else [0.0]
    
    def get_token_entropies(self) -> List[float]:
        """Entropie des distributions de tokens"""
        entropies = []
        for logits in self.logits:
            probs = self.softmax(logits.astype(np.float64))
            ent = self.compute_entropy(probs).mean()
            entropies.append(float(ent))
        return entropies if entropies else [0.0]
    
    def get_layer_entropies(self) -> Dict[int, float]:
        """Entropie moyenne par couche"""
        result = {}
        for layer_idx, activations in self.layer_activations.items():
            ents = []
            for act in activations:
                if act.ndim >= 2:
                    sample = act.reshape(-1, act.shape[-1])[:50]
                    probs = self.softmax(sample.astype(np.float64))
                    ents.append(float(self.compute_entropy(probs).mean()))
            if ents:
                result[layer_idx] = np.mean(ents)
        return result


# =============================================================================
# PSYCHEDELIC LAYER WRAPPER
# =============================================================================

class PsychedelicLayer(nn.Module):
    """
    Wrapper pour une couche transformer avec effets psychédéliques.

    Effets implémentés:
    - Injection de bruit gaussien (dissolution des frontières conceptuelles)
    - Rotation de phase (synesthésie sémantique)
    - Mélange dimensionnel (connexions inhabituelles)
    - Aplatissement de l'attention (hyperconnectivité)
    """

    def __init__(self, original_layer, layer_idx: int, total_layers: int,
                 tracker: Optional[EntropyTracker] = None):
        # IMPORTANT: définir _layer AVANT super().__init__() car nn.Module
        # peut déclencher __getattr__ pendant l'initialisation
        object.__setattr__(self, '_layer', original_layer)
        super().__init__()
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.tracker = tracker

        # Paramètres d'effet (modifiés dynamiquement)
        self.intensity = 0.0
        self.noise_scale = 0.0
        self.phase_rotation = 0.0
        self.dimension_shuffle_rate = 0.0

        # Position relative (0-1) pour modulation par couche
        self.relative_pos = layer_idx / total_layers

        # Sensibilité de cette couche (courbe gaussienne centrée sur 0.5)
        # Les couches médianes sont plus affectées
        self.layer_sensitivity = float(np.exp(-((self.relative_pos - 0.5) ** 2) / 0.08))

    def __getattr__(self, name: str):
        """Délègue les attributs inconnus à la couche originale (ex: use_sliding)"""
        # Accéder à _layer de manière sûre pour éviter la récursion
        _layer = object.__getattribute__(self, '_layer')
        return getattr(_layer, name)
    
    def set_intensity(self, intensity: float):
        """Configure l'intensité de l'effet (0.0 à 1.0)"""
        self.intensity = max(0.0, min(1.0, intensity))

        # Moduler par la sensibilité de la couche
        effective = self.intensity * self.layer_sensitivity

        # Paramètres réduits pour garder une cohérence minimale
        self.noise_scale = effective * 0.25  # Réduit de 0.5
        self.phase_rotation = effective * 0.1  # Réduit de 0.2
        self.dimension_shuffle_rate = effective * 0.08  # Réduit de 0.15
    
    def _inject_noise(self, x: mx.array) -> mx.array:
        """
        Injection de bruit gaussien dans les représentations.
        Simule: dissolution des frontières entre concepts
        """
        if self.noise_scale > 0:
            # Bruit proportionnel à la magnitude des activations
            std = mx.sqrt(mx.mean(x * x) + 1e-8)
            noise = mx.random.normal(shape=x.shape) * self.noise_scale * std
            return x + noise
        return x
    
    def _phase_scramble(self, x: mx.array) -> mx.array:
        """
        Rotation de phase dans l'espace des représentations.
        Simule: synesthésie, mélange des modalités sensorielles
        """
        if self.phase_rotation > 0 and x.shape[-1] >= 2:
            # Angle de rotation avec perturbation aléatoire
            base_angle = self.phase_rotation * np.pi * 0.5
            noise_angle = float(mx.random.uniform(low=-0.15, high=0.15).item()) * self.intensity
            angle = base_angle + noise_angle
            
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            
            d = x.shape[-1]
            half = d // 2
            
            x1 = x[..., :half]
            x2 = x[..., half:2*half]
            rest = x[..., 2*half:] if d > 2*half else None
            
            # Rotation 2D sur les paires de dimensions
            rotated_1 = x1 * cos_a - x2 * sin_a
            rotated_2 = x1 * sin_a + x2 * cos_a
            
            if rest is not None:
                rotated = mx.concatenate([rotated_1, rotated_2, rest], axis=-1)
            else:
                rotated = mx.concatenate([rotated_1, rotated_2], axis=-1)
            
            # Mélange progressif avec l'original
            blend = self.phase_rotation * 0.7
            return x * (1 - blend) + rotated * blend
        return x
    
    def _dimension_shuffle(self, x: mx.array) -> mx.array:
        """
        Mélange partiel des dimensions cachées.
        Simule: connexions inhabituelles entre concepts normalement séparés
        """
        if self.dimension_shuffle_rate > 0 and self.intensity > 0.4:
            d = x.shape[-1]
            shuffle_size = max(1, int(d * self.dimension_shuffle_rate))
            
            if shuffle_size > 1:
                # Roll cyclique sur une portion des features
                shift = int(mx.random.randint(low=1, high=max(2, shuffle_size // 2), shape=()).item())
                
                shuffled_part = mx.roll(x[..., :shuffle_size], shift=shift, axis=-1)
                
                # Mélange partiel (pas remplacement total)
                blend_factor = self.intensity * 0.5
                blended_part = x[..., :shuffle_size] * (1 - blend_factor) + shuffled_part * blend_factor
                
                return mx.concatenate([blended_part, x[..., shuffle_size:]], axis=-1)
        return x
    
    def _cross_dimension_bleed(self, x: mx.array) -> mx.array:
        """
        Fait "saigner" l'information entre dimensions adjacentes.
        Simule: dissolution des frontières perceptuelles
        """
        if self.intensity > 0.6:
            bleed_factor = (self.intensity - 0.6) * 0.3
            
            # Moyenne glissante légère sur les dimensions
            d = x.shape[-1]
            if d > 2:
                left = mx.concatenate([x[..., :1], x[..., :-1]], axis=-1)
                right = mx.concatenate([x[..., 1:], x[..., -1:]], axis=-1)
                
                blurred = (x + left * bleed_factor + right * bleed_factor) / (1 + 2 * bleed_factor)
                return blurred
        return x
    
    def __call__(self, x: mx.array, mask: Optional[mx.array] = None,
                 cache: Optional[Tuple] = None):
        """Forward pass avec effets psychédéliques"""

        # Forward original de la couche
        # Certains modèles retournent (output, cache), d'autres juste output
        result = self._layer(x, mask=mask, cache=cache)

        if isinstance(result, tuple):
            output, new_cache = result
        else:
            output = result
            new_cache = None
        
        # Appliquer les perturbations si intensité > 0
        if self.intensity > 0:
            output = self._inject_noise(output)
            output = self._phase_scramble(output)
            output = self._dimension_shuffle(output)
            output = self._cross_dimension_bleed(output)
        
        # Tracking pour visualisation (seulement quelques couches pour éviter surcharge)
        if self.tracker is not None:
            # Enregistrer pour les couches clés: début, milieu, fin
            if self.layer_idx in [0, self.total_layers // 4, self.total_layers // 2,
                                   3 * self.total_layers // 4, self.total_layers - 1]:
                self.tracker.record_hidden_state(output, layer_idx=self.layer_idx)

        # Retourner dans le même format que l'original
        if new_cache is not None:
            return output, new_cache
        else:
            return output


# =============================================================================
# MAIN PSYCHEDELIC LLM CLASS
# =============================================================================

class PsychedelicLLM:
    """
    Interface principale pour l'exploration psychédélique de LLM.
    
    Exemple d'utilisation:
        psy = PsychedelicLLM("HuggingFaceTB/SmolLM3-3B-Base")
        
        # Génération sobre
        report_sober = psy.generate("The meaning of life is", intensity=0.0)
        
        # Génération altérée
        report_trip = psy.generate("The meaning of life is", intensity=0.8)
        
        # Comparaison visuelle
        psy.visualize_comparison([report_sober, report_trip])
    """
    
    def __init__(self, model_name: str = "./SmolLM3-3B-pseudoLuc-merged"):
        print(f"🧠 Chargement de {model_name}...")
        from mlx_lm import load
        
        self.model, self.tokenizer = load(model_name)
        self.model_name = model_name
        
        self.tracker = EntropyTracker()
        self.trip_reports: List[TripReport] = []
        self.original_layers = None
        
        # Wrapper les couches
        self._setup_psychedelic_layers()
        print(f"✨ Modèle prêt ({self.n_layers} couches)")
    
    def _setup_psychedelic_layers(self):
        """Remplace les couches par des versions psychédéliques"""
        # Accéder aux couches du modèle
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
        else:
            raise ValueError("Architecture de modèle non supportée")
        
        self.n_layers = len(layers)
        self.original_layers = list(layers)
        
        new_layers = []
        for i, layer in enumerate(layers):
            psy_layer = PsychedelicLayer(
                layer, i, self.n_layers, 
                tracker=self.tracker
            )
            new_layers.append(psy_layer)
        
        # Remplacer les couches
        if hasattr(self.model, 'model'):
            self.model.model.layers = new_layers
        else:
            self.model.layers = new_layers
        
        self.psychedelic_layers = new_layers
    
    def set_intensity(self, intensity: float):
        """
        Ajuste l'intensité globale (0.0 à 1.0)
        
        Guide des intensités:
        - 0.0: Sobre, fonctionnement normal
        - 0.1-0.3: Micro-dose, subtil
        - 0.4-0.6: Dose modérée, altérations notables
        - 0.7-0.8: Dose forte, patterns inhabituels
        - 0.9-1.0: Peak experience, génération très chaotique
        """
        for layer in self.psychedelic_layers:
            layer.set_intensity(intensity)
    
    def generate(self, prompt: str, intensity: float = 0.0,
                 max_tokens: int = 100, verbose: bool = False) -> TripReport:
        """
        Génère du texte avec l'intensité spécifiée.

        Args:
            prompt: Le prompt d'entrée
            intensity: 0.0 (sobre) à 1.0 (peak experience)
            max_tokens: Nombre max de tokens à générer
            verbose: Afficher le texte pendant la génération

        Returns:
            TripReport avec le texte et les métriques
        """
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        self.tracker.reset()
        self.set_intensity(intensity)

        # Température adaptative selon l'intensité
        # Plus d'intensité = plus d'entropie dans le sampling aussi
        temp = 0.7 + intensity * 1.3  # 0.7 à 2.0
        top_p = 0.85 + intensity * 0.13  # 0.85 à 0.98

        # Créer le sampler avec les paramètres
        # Note: repetition_penalty est géré via repetition_context_size dans make_sampler
        # ou via logits_processors si disponible
        sampler = make_sampler(temp=temp, top_p=top_p)

        start_time = time.time()

        output = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=verbose
        )
        
        generation_time = time.time() - start_time
        
        # Créer le rapport
        report = TripReport(
            intensity=intensity,
            prompt=prompt,
            output=output,
            hidden_state_entropies=self.tracker.get_hidden_state_entropies(),
            attention_entropies=self.tracker.get_attention_entropies(),
            token_entropies=self.tracker.get_token_entropies(),
            layer_entropies=self.tracker.get_layer_entropies(),
            generation_time=generation_time
        )
        
        self.trip_reports.append(report)
        return report
    
    def comparative_trip(self, prompt: str,
                         intensities: List[float] = [0.0, 0.1, 0.2, 0.3],
                         max_tokens: int = 80) -> List[TripReport]:
        """
        Compare les générations à différentes intensités.
        """
        reports = []
        
        for intensity in intensities:
            level_name = self._intensity_name(intensity)
            emoji = self._intensity_emoji(intensity)
            print(f"\n{emoji} Génération [{level_name}] (intensité={intensity})...")
            
            report = self.generate(prompt, intensity, max_tokens)
            reports.append(report)
            
            # Afficher un extrait
            preview = report.output
            if len(preview) > 200:
                preview = preview[:200] + "..."
            print(f"   {preview}")
            print(f"   ⏱️  {report.generation_time:.2f}s")
        
        return reports
    
    @staticmethod
    def _intensity_name(intensity: float) -> str:
        if intensity == 0:
            return "SOBRE"
        elif intensity <= 0.1:
            return "MICRO-DOSE"
        elif intensity <= 0.2:
            return "DOSE"
        elif intensity <= 0.3:
            return "2x DOSE"
        elif intensity < 0.7:
            return "MODÉRÉ"
        elif intensity < 0.9:
            return "INTENSE"
        else:
            return "BREAKTHROUGH"
    
    @staticmethod
    def _intensity_emoji(intensity: float) -> str:
        if intensity == 0:
            return "🧠"
        elif intensity < 0.4:
            return "🌀"
        elif intensity < 0.7:
            return "🌈"
        else:
            return "🍄"


# =============================================================================
# VISUALIZATION
# =============================================================================

class TripVisualizer:
    """Génère des visualisations des métriques psychédéliques"""
    
    def __init__(self, figsize: Tuple[int, int] = (16, 14)):
        self.figsize = figsize
        self.colors = {
            0.0: '#2ecc71',   # Vert - sobre
            0.1: '#3498db',   # Bleu - micro-dose
            0.2: '#9b59b6',   # Violet - dose
            0.3: '#e74c3c',   # Rouge - 2x dose
        }
        self.default_color = '#f39c12'  # Orange pour autres intensités
    
    def _get_color(self, intensity: float) -> str:
        """Retourne la couleur pour une intensité donnée"""
        # Chercher la couleur la plus proche
        closest = min(self.colors.keys(), key=lambda x: abs(x - intensity))
        if abs(closest - intensity) < 0.15:
            return self.colors[closest]
        return self.default_color
    
    def plot_comparative_analysis(self, reports: List[TripReport], 
                                   save_path: Optional[str] = None,
                                   show: bool = True) -> plt.Figure:
        """
        Crée une visualisation complète comparative.
        """
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
        
        # Style sombre
        fig.patch.set_facecolor('#1a1a2e')
        
        # 1. Entropie des hidden states par intensité
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_entropy_evolution(ax1, reports, 'hidden_state_entropies',
                                      'Entropie des Hidden States')
        
        # 2. Distribution des entropies  
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_entropy_distribution(ax2, reports)
        
        # 3. Barres d'entropie moyenne
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_mean_entropy_bars(ax3, reports)
        
        # 4. Radar chart des métriques
        ax4 = fig.add_subplot(gs[1, 1], projection='polar')
        self._plot_radar(ax4, reports)
        
        # 5. Textes générés (en bas)
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_text_samples(ax5, reports)
        
        # Titre global
        fig.suptitle('🍄 Analyse Psychédélique du LLM - Comparaison des Intensités',
                     fontsize=16, fontweight='bold', color='white', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor=fig.get_facecolor(), edgecolor='none')
            print(f"📊 Visualisation sauvegardée: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def _plot_entropy_evolution(self, ax, reports: List[TripReport], 
                                 metric: str, title: str):
        """Plot l'évolution de l'entropie au cours de la génération"""
        ax.set_facecolor('#16213e')
        
        for report in reports:
            values = getattr(report, metric)
            if values and len(values) > 0:
                color = self._get_color(report.intensity)
                label = f"i={report.intensity:.1f}"
                x = range(len(values))
                ax.plot(x, values, color=color, linewidth=2, alpha=0.8, 
                       label=label, marker='o', markersize=4)
                ax.fill_between(x, values, alpha=0.15, color=color)
        
        ax.set_xlabel('Étape', color='white', fontsize=10)
        ax.set_ylabel('Entropie (bits)', color='white', fontsize=10)
        ax.set_title(title, color='white', fontweight='bold', fontsize=11)
        ax.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#444',
                  labelcolor='white', fontsize=9)
        ax.tick_params(colors='white', labelsize=9)
        ax.grid(True, alpha=0.2, color='white')
        
        for spine in ax.spines.values():
            spine.set_color('#444')
    
    def _plot_entropy_distribution(self, ax, reports: List[TripReport]):
        """Boxplot de la distribution des entropies"""
        ax.set_facecolor('#16213e')
        
        data = []
        labels = []
        colors = []
        
        for report in reports:
            values = report.hidden_state_entropies
            if values:
                data.append(values)
                labels.append(f"i={report.intensity:.1f}")
                colors.append(self._get_color(report.intensity))
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            for element in ['whiskers', 'caps', 'medians']:
                for item in bp[element]:
                    item.set_color('white')
                    item.set_alpha(0.8)
        
        ax.set_xlabel('Intensité', color='white', fontsize=10)
        ax.set_ylabel('Entropie (bits)', color='white', fontsize=10)
        ax.set_title('Distribution des Entropies', color='white', 
                     fontweight='bold', fontsize=11)
        ax.tick_params(colors='white', labelsize=9)
        
        for spine in ax.spines.values():
            spine.set_color('#444')
    
    def _plot_mean_entropy_bars(self, ax, reports: List[TripReport]):
        """Barres d'entropie moyenne par intensité"""
        ax.set_facecolor('#16213e')
        
        intensities = [r.intensity for r in reports]
        hs_means = [np.mean(r.hidden_state_entropies) if r.hidden_state_entropies else 0 
                    for r in reports]
        
        x = np.arange(len(intensities))
        colors = [self._get_color(i) for i in intensities]
        
        bars = ax.bar(x, hs_means, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
        
        # Ajouter les valeurs sur les barres
        for bar, val in zip(bars, hs_means):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{val:.2f}', ha='center', va='bottom', color='white', fontsize=9)
        
        ax.set_xlabel('Intensité', color='white', fontsize=10)
        ax.set_ylabel('Entropie moyenne (bits)', color='white', fontsize=10)
        ax.set_title('Entropie Moyenne par Intensité', color='white', 
                     fontweight='bold', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{i:.1f}' for i in intensities], color='white')
        ax.tick_params(colors='white', labelsize=9)
        
        for spine in ax.spines.values():
            spine.set_color('#444')
    
    def _plot_radar(self, ax, reports: List[TripReport]):
        """Radar chart des métriques normalisées"""
        categories = ['Entropie\nHS', 'Variabilité', 'Température\neffective', 'Chaos']
        n_cats = len(categories)
        
        angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
        angles += angles[:1]  # Fermer le polygone
        
        ax.set_facecolor('#16213e')
        
        for report in reports:
            hs_ent = np.mean(report.hidden_state_entropies) if report.hidden_state_entropies else 0
            variability = np.std(report.hidden_state_entropies) if len(report.hidden_state_entropies) > 1 else 0
            temp_effect = report.intensity
            chaos = report.intensity ** 1.5  # Non-linéaire
            
            # Normaliser (0-1)
            values = [
                min(hs_ent / 12, 1),
                min(variability / 3, 1),
                temp_effect,
                chaos
            ]
            values += values[:1]
            
            color = self._get_color(report.intensity)
            ax.plot(angles, values, color=color, linewidth=2, alpha=0.8,
                   label=f'i={report.intensity:.1f}')
            ax.fill(angles, values, color=color, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, color='white', size=9)
        ax.set_title('Profil Psychédélique', color='white', fontweight='bold', 
                     pad=20, fontsize=11)
        
        # Légende
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0),
                  facecolor='#1a1a2e', edgecolor='#444', labelcolor='white', fontsize=8)
        
        # Grille
        ax.yaxis.grid(True, color='white', alpha=0.2)
        ax.xaxis.grid(True, color='white', alpha=0.2)
        ax.set_ylim(0, 1)
        ax.tick_params(colors='white', labelsize=8)
    
    def _plot_text_samples(self, ax, reports: List[TripReport]):
        """Affiche les textes générés en entier"""
        ax.set_facecolor('#16213e')
        ax.axis('off')

        # Calculer l'espace disponible par rapport
        n_reports = len(reports)
        space_per_report = 0.95 / max(n_reports, 1)

        y_pos = 0.98

        for report in reports:
            emoji = PsychedelicLLM._intensity_emoji(report.intensity)
            level = PsychedelicLLM._intensity_name(report.intensity)

            # Texte complet (juste nettoyé des sauts de ligne)
            sample = report.output.replace('\n', ' ').strip()

            color = self._get_color(report.intensity)

            # Titre de l'intensité
            ax.text(0.01, y_pos, f"{emoji} {level} (i={report.intensity:.1f}):",
                   transform=ax.transAxes, fontsize=9, fontweight='bold',
                   color=color, family='sans-serif')

            # Texte généré complet avec wrapping automatique
            text_obj = ax.text(0.01, y_pos - 0.06, sample, transform=ax.transAxes,
                   fontsize=8, color='white', family='monospace', style='italic',
                   wrap=True, verticalalignment='top')
            text_obj.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='#0d1117',
                                   edgecolor=color, alpha=0.7))

            y_pos -= space_per_report

        ax.set_title('Textes Générés (complets)', color='white',
                     fontweight='bold', loc='left', fontsize=11, pad=10)
    
    def plot_entropy_heatmap(self, reports: List[TripReport],
                             save_path: Optional[str] = None,
                             show: bool = True) -> plt.Figure:
        """
        Heatmap de l'entropie au fil de la génération.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        
        # Préparer les données
        valid_reports = [r for r in reports if r.hidden_state_entropies]
        if not valid_reports:
            ax.text(0.5, 0.5, "Pas de données d'entropie disponibles",
                   ha='center', va='center', color='white', fontsize=14)
            return fig
        
        max_len = max(len(r.hidden_state_entropies) for r in valid_reports)
        matrix = []
        
        for report in valid_reports:
            values = report.hidden_state_entropies
            # Padding si nécessaire
            if len(values) < max_len:
                padded = values + [values[-1]] * (max_len - len(values))
            else:
                padded = values[:max_len]
            matrix.append(padded)
        
        matrix = np.array(matrix)
        
        im = ax.imshow(matrix, aspect='auto', cmap='magma', interpolation='bilinear')
        
        ax.set_yticks(range(len(valid_reports)))
        ax.set_yticklabels([f'i={r.intensity:.1f}' for r in valid_reports], color='white')
        ax.set_xlabel('Étape de génération', color='white', fontsize=11)
        ax.set_ylabel('Intensité', color='white', fontsize=11)
        ax.set_title('🌡️ Heatmap de l\'Entropie', 
                     color='white', fontweight='bold', fontsize=13)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Entropie (bits)', color='white', fontsize=10)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        ax.tick_params(colors='white')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor=fig.get_facecolor())
            print(f"📊 Heatmap sauvegardée: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_layer_analysis(self, reports: List[TripReport],
                            save_path: Optional[str] = None,
                            show: bool = True) -> plt.Figure:
        """
        Analyse de l'entropie par couche du modèle.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        
        for report in reports:
            if report.layer_entropies:
                layers = sorted(report.layer_entropies.keys())
                values = [report.layer_entropies[l] for l in layers]
                
                color = self._get_color(report.intensity)
                ax.plot(layers, values, color=color, linewidth=2, alpha=0.8,
                       label=f'i={report.intensity:.1f}', marker='s', markersize=6)
        
        ax.set_xlabel('Numéro de couche', color='white', fontsize=11)
        ax.set_ylabel('Entropie moyenne (bits)', color='white', fontsize=11)
        ax.set_title('📊 Entropie par Couche du Modèle', 
                     color='white', fontweight='bold', fontsize=13)
        ax.legend(facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='white')
        
        for spine in ax.spines.values():
            spine.set_color('#444')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor=fig.get_facecolor())
            print(f"📊 Analyse par couche sauvegardée: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_full_experiment(
    model_name: str = "./SmolLM3-3B-pseudoLuc-merged",
    prompts: Optional[List[str]] = None,
    intensities: List[float] = [0.0, 0.1, 0.2, 0.3],
    max_tokens: int = 80,
    output_dir: str = "./psychedelic_outputs",
    show_plots: bool = True
) -> List[TripReport]:
    """
    Exécute une expérience complète avec visualisations.
    
    Args:
        model_name: Nom du modèle HuggingFace
        prompts: Liste de prompts à tester
        intensities: Intensités à comparer
        max_tokens: Tokens max par génération
        output_dir: Dossier pour sauvegarder les visualisations
        show_plots: Afficher les plots interactivement
    
    Returns:
        Liste de tous les TripReports générés
    """
    
    if prompts is None:
        prompts = [
            "La nature de la conscience est",
            "Quand j'ai regardé mes mains, j'ai réalisé que",
            "Les couleurs et les sons ont commencé à fusionner, et j'ai compris que",
            "La frontière entre soi et l'univers se dissout quand",
        ]
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("🍄 PSYCHEDELIC LLM EXPLORER - EXPÉRIENCE COMPLÈTE")
    print("=" * 70)
    print(f"📦 Modèle: {model_name}")
    print(f"📝 Prompts: {len(prompts)}")
    print(f"🎚️  Intensités: {intensities}")
    print(f"📊 Max tokens: {max_tokens}")
    print(f"💾 Output: {output_dir}")
    print("=" * 70)
    
    # Initialiser
    llm = PsychedelicLLM(model_name)
    viz = TripVisualizer()
    
    all_reports = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n{'─' * 70}")
        print(f"📝 PROMPT {i+1}/{len(prompts)}")
        print(f"   \"{prompt}\"")
        print('─' * 70)
        
        reports = llm.comparative_trip(prompt, intensities, max_tokens)
        all_reports.extend(reports)
        
        # Nom de fichier sécurisé
        safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:25])
        
        # Visualisations
        viz.plot_comparative_analysis(
            reports,
            save_path=f"{output_dir}/analysis_{i+1}_{safe_prompt}.png",
            show=show_plots
        )
        
        viz.plot_entropy_heatmap(
            reports,
            save_path=f"{output_dir}/heatmap_{i+1}_{safe_prompt}.png",
            show=show_plots
        )
        
        viz.plot_layer_analysis(
            reports,
            save_path=f"{output_dir}/layers_{i+1}_{safe_prompt}.png",
            show=show_plots
        )

        # Sauvegarder les textes en markdown
        md_path = f"{output_dir}/textes_{i+1}_{safe_prompt}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# 🍄 Psychedelic LLM Explorer\n\n")
            f.write(f"## Prompt\n\n> {prompt}\n\n")
            f.write(f"---\n\n")

            for report in reports:
                emoji = PsychedelicLLM._intensity_emoji(report.intensity)
                level = PsychedelicLLM._intensity_name(report.intensity)
                avg_entropy = np.mean(report.hidden_state_entropies) if report.hidden_state_entropies else 0

                f.write(f"### {emoji} {level} (intensité={report.intensity:.1f})\n\n")
                f.write(f"**Entropie moyenne:** {avg_entropy:.2f} bits | ")
                f.write(f"**Temps:** {report.generation_time:.2f}s\n\n")
                f.write(f"```\n{report.output}\n```\n\n")

        print(f"📝 Textes sauvegardés: {md_path}")

    # Résumé final
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DE L'EXPÉRIENCE")
    print("=" * 70)
    
    for intensity in intensities:
        int_reports = [r for r in all_reports if r.intensity == intensity]
        
        if int_reports:
            avg_hs_entropy = np.mean([
                np.mean(r.hidden_state_entropies) 
                for r in int_reports if r.hidden_state_entropies
            ])
            avg_time = np.mean([r.generation_time for r in int_reports])
            
            emoji = PsychedelicLLM._intensity_emoji(intensity)
            level = PsychedelicLLM._intensity_name(intensity)
            
            print(f"{emoji} {level:12} (i={intensity:.1f}): "
                  f"Entropie={avg_hs_entropy:.2f} bits, "
                  f"Temps={avg_time:.2f}s")
    
    print(f"\n✅ {len(all_reports)} générations complétées")
    print(f"📁 Visualisations sauvegardées dans {output_dir}/")
    
    return all_reports


def interactive_mode(model_name: str = "./SmolLM3-3B-pseudoLuc-merged"):
    """
    Mode interactif pour explorer le modèle.
    """
    print("=" * 70)
    print("🍄 PSYCHEDELIC LLM EXPLORER - MODE INTERACTIF")
    print("=" * 70)
    
    llm = PsychedelicLLM(model_name)
    viz = TripVisualizer()
    
    print("\nCommandes:")
    print("  - Entrez un prompt pour générer")
    print("  - 'i=X' pour changer l'intensité (ex: i=0.7)")
    print("  - 'compare' pour comparer plusieurs intensités")
    print("  - 'quit' pour quitter")
    print()
    
    current_intensity = 0.0
    
    while True:
        try:
            user_input = input(f"\n🎚️  [intensité={current_intensity:.1f}] > ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("👋 Au revoir!")
                break
            
            if user_input.lower().startswith('i='):
                try:
                    current_intensity = float(user_input[2:])
                    current_intensity = max(0.0, min(1.0, current_intensity))
                    emoji = PsychedelicLLM._intensity_emoji(current_intensity)
                    print(f"{emoji} Intensité réglée à {current_intensity:.1f}")
                except ValueError:
                    print("❌ Format invalide. Utilisez: i=0.5")
                continue
            
            if user_input.lower() == 'compare':
                prompt = input("📝 Prompt pour comparaison: ").strip()
                if prompt:
                    reports = llm.comparative_trip(prompt)
                    viz.plot_comparative_analysis(reports, show=True)
                continue
            
            # Sinon, c'est un prompt
            print(f"\n🌀 Génération en cours...")
            report = llm.generate(user_input, intensity=current_intensity, max_tokens=100)
            
            print(f"\n{'─' * 50}")
            print(report.output)
            print(f"{'─' * 50}")
            print(f"⏱️  {report.generation_time:.2f}s | "
                  f"📊 Entropie: {np.mean(report.hidden_state_entropies):.2f} bits")
            
        except KeyboardInterrupt:
            print("\n👋 Interruption. Au revoir!")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🍄 Psychedelic LLM Explorer")
    parser.add_argument("--model", type=str, default="./SmolLM3-3B-pseudoLuc-merged",
                        help="Chemin vers le modèle (défaut: pseudoLuc fine-tuné)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Mode interactif")
    parser.add_argument("--output-dir", type=str, default="./psychedelic_outputs",
                        help="Dossier de sortie pour les visualisations")
    parser.add_argument("--no-show", action="store_true",
                        help="Ne pas afficher les plots (seulement sauvegarder)")
    parser.add_argument("--max-tokens", type=int, default=80,
                        help="Nombre max de tokens à générer")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt unique à tester")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.model)
    else:
        prompts = [args.prompt] if args.prompt else None
        run_full_experiment(
            model_name=args.model,
            prompts=prompts,
            max_tokens=args.max_tokens,
            output_dir=args.output_dir,
            show_plots=not args.no_show
        )