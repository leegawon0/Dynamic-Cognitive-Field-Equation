
"""
DCFE - Dynamic Cognitive Field Equation 

1. Context -> Hormone Projection
2. Temporal Hormone Dynamics
3. Hormone -> Emotion Mapping
4. Emotion -> Hormone Feedback
5. Extensible: Relationship Tensor

============================================================

Philosophical Declaration:
  All cognitive fields (hormones, social norms, manners) are equivalent.
  No special treatment - all are gravitational fields that bend space.

Complete Flow:
  Context -> [Projection] -> Hormone -> [Accumulation] -> Hormone State
       -> Emotion weight -> [collapse] -> Emotion Selection
       -> [Feedback] -> Hormone Readjustment -> [Decay] -> Next Time
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional

# ============================================================
# EMOTIONS: Emotion space coordinates
# ============================================================

EMOTIONS_ALL = {
    "Threat":   np.array([-2.0,  1.0]),
    "Anger":    np.array([-3.0,  1.0]),
    "Jealous":  np.array([-2.5,  2.0]),
    "Anxiety":  np.array([-2.0, -2.0]),
    "Justice":  np.array([-1.0,  2.5]),
    "Awkward":  np.array([-0.5,  0.0]),
    "Boredom":  np.array([ 0.0, -1.5]),
    "Pity":     np.array([ 2.5, -1.0]),
    "Calm":     np.array([ 0.5, -0.5]),
    "Crime":    np.array([ 2.0, -2.5]),
    "Trust":    np.array([ 2.0, -1.5]),
    "Happy":    np.array([ 3.0,  1.5]),
}

# ============================================================
# FIELDS: Cognitive Mass Tensor
# ============================================================
# Defines the pulling and pushing interactions between cognitive fields 
# (like hormones or social norms) and emotions.
# ============================================================

FIELDS = {
    # ----- Hormone Fields ----------------------------------------------------
    "cortisol": {
        "pull": {"Threat": 5.0, "Anger": 4.0, "Anxiety": 2.0, "Jealous": 5.0, "Justice": 1.5},
        "push": {"Trust": 0.2, "Happy": 0.2, "Calm": 0.3},
    },
    "oxytocin": {
        "pull": {"Trust": 5.0, "Happy": 4.0, "Calm": 0.1, "Pity": 5.0},
        "push": {"Threat": 0.1, "Anger": 0.1},
    },
    "dopamine": {
        "pull": {"Happy": 5.0, "Trust": 1.0, "Justice": 3.0},
        "push": {"Boredom": 0.5},
    },
    "serotonin": {
        "pull": {"Calm": 5.0, "Pity": 1.5, "Trust": 1.0},
        "push": {"Anxiety": 0.3, "Threat": 0.2},
    },
    # ----- Social/Cognitive Fields (Equivalent to Hormones) -------------------
    "social_norm": {
        "pull": {"Calm": 2.0, "Boredom": 1.5, "Crime":5.0},
        "push": {"Threat": 0.2, "Anger": 0.2, "Justice": 0.4},
    },
    "manner": {
        "pull": {"Calm": 3.0, "Awkward": 1.5},
        "push": {"Anger": 0.3, "Threat": 0.3},
    },
}

# ----- Hormone Decay Rate (Return to baseline over time) -------------------
HORMONE_DECAY = {
    "cortisol":    0.75,
    "oxytocin":    0.60,
    "dopamine":    0.45,
    "serotonin":   0.85,
    "social_norm": 0.90,
    "manner":      0.92,
}

# ----- Emotion -> Hormone Feedback Effect ------------------------------------
# The effect of the selected emotion on hormones (multiplication coefficient)
EMOTION_TO_HORMONE = {
    "Anger": {
        "cortisol": 1.2,   # Anger -> cortisol increase
        "serotonin": 0.8,  # Anger -> serotonin decrease
    },
    "Calm": {
        "cortisol": 0.7,   # Calm -> cortisol decrease
        "serotonin": 1.3,  # Calm -> serotonin increase
    },
    "Justice": {
        "dopamine": 1.4,   # Justice -> dopamine increase
        "cortisol": 0.9,   # Justice -> cortisol slight decrease
    },
    "Threat": {
        "cortisol": 1.5,   # Threat -> cortisol spike
        "dopamine": 0.6,   # Threat -> dopamine decrease
    },
    "Happy": {
        "dopamine": 1.3,   # Happy -> dopamine increase
        "oxytocin": 1.2,   # Happy -> oxytocin increase
    },
    "Trust": {
        "oxytocin": 1.4,   # Trust -> oxytocin increase
        "cortisol": 0.8,   # Trust -> cortisol decrease
    },
}

# ----- Geometric Parameters ------------------------------------
SIGMA = 2.0
LAMBDA_CONST = 0.05
ALPHA_SIGMA = 0.2


# ============================================================
# HormoneState: Temporal Hormone Dynamics
# ============================================================

class HormoneState:
    """
    Manages the temporal dynamics of hormones
    
    Features:
    - Base hormone level (personality baseline)
    - Hormone increase based on input stimulus
    - Decay over time (return to baseline)
    - Application of emotion feedback
    """
    
    def __init__(self, base_levels: Optional[Dict[str, float]] = None):
        """
        Parameters
        ----------
        base_levels : dict
            Base hormone levels (personality)
            e.g.: {"cortisol": 1.0, "social_norm": 50.0}
        """
        # Base levels (personality)
        self.base = base_levels or {
            "cortisol": 1.0,
            "oxytocin": 1.0,
            "dopamine": 0.5,
            "serotonin": 1.0,
            "social_norm": 50.0,
            "manner": 30.0,
        }
        
        # Current hormone state
        self.current = self.base.copy()
        
        # Hormone history (for visualization)
        self.history = {field: [self.base[field]] for field in self.base}
    
    def stimulate(self, delta: Dict[str, float]):
        """
        Hormone changes based on input stimulus
        
        Parameters
        ----------
        delta : dict
            Amount of change for each hormone
            e.g.: {"cortisol": +0.8, "dopamine": +0.3}
        """
        for field, change in delta.items():
            if field in self.current:
                self.current[field] += change
                # Prevent negative values
                self.current[field] = max(0.0, self.current[field])
    
    def apply_emotion_feedback(self, emotion: str, strength: float = 1.0):
        """
        Selected emotion feedbacks to hormones
        
        Parameters
        ----------
        emotion : str
            Selected emotion
        strength : float
            Feedback strength (0.0~1.0)
        """
        if emotion not in EMOTION_TO_HORMONE:
            return
        
        effects = EMOTION_TO_HORMONE[emotion]
        for field, factor in effects.items():
            if field in self.current:
                # Multiplicative feedback (1.0 = no change, >1.0 = increase, <1.0 = decrease)
                adjustment = 1.0 + (factor - 1.0) * strength
                self.current[field] *= adjustment
    
    def decay(self):
        """
        Hormone decay over time (return to baseline)
        """
        for field in self.current:
            decay_rate = HORMONE_DECAY.get(field, 0.8)
            # Move towards the baseline by the decay rate from the current value
            self.current[field] = (
                self.base[field] + 
                (self.current[field] - self.base[field]) * decay_rate
            )
        
        # Record history
        for field in self.current:
            self.history[field].append(self.current[field])
    
    def get_state(self) -> Dict[str, float]:
        """Returns the current hormone state"""
        return self.current.copy()


# ============================================================
# ContextProjector: Context -> Hormone Projection
# ============================================================

class ContextProjector:
    """
    Projects context embedding to hormone delta.
    
    Toy model: Keyword-based (simple rules)
    """
    
    def __init__(self, mode='keyword'):
        """
        Parameters
        ----------
        mode : str
            'keyword' - Keyword-based (toy model)
            'learned' - Learned projection matrix (complete implementation)
        """
        self.mode = mode
        
        # Keyword -> Hormone mapping (for toy model)
        self.keyword_mappings = {
            # Stress/Violence -> cortisol
            "cortisol": {
                "keywords": ["murder", "kill", "violence", "tragedy", "pain", "threat", "fear", "angry", "unfair"],
                "strength": 0.8,
            },
            # Revenge/Justice -> dopamine
            "dopamine": {
                "keywords": ["criminal", "revenge", "justice", "reward"],
                "strength": 0.5,
            },
            # Norm/Law -> social_norm
            "social_norm": {
                "keywords": ["law", "norm", "order", "society", "crime", "rule"],
                "strength": 2.0,
            },
            # Bond/Love -> oxytocin
            "oxytocin": {
                "keywords": ["family", "love", "father", "mother", "friend"],
                "strength": 0.6,
            },
            # Calmness -> serotonin
            "serotonin": {
                "keywords": ["calm", "stability", "peace", "rest"],
                "strength": 0.4,
            },
        }
    
    def project(self, text: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Text -> Extract (Hormone delta, Activation tensor)
        
        Parameters
        ----------
        text : str
            Input text
            
        Returns
        -------
        delta : dict
            Amount of change for each hormone
        T_tensor : dict
            Activation strength of each field (Tensor!)
        """
        if self.mode == 'keyword':
            return self._project_keyword(text)
        else:
            # TODO: Implement learned projection matrix
            raise NotImplementedError("Learned projection not yet implemented")
    
    def _project_keyword(self, text: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Keyword-based projection (Toy model)
        
        Returns
        -------
        delta : dict
            Hormone change amount
        T_tensor : dict
            Activation strength per field 
        """
        delta = {field: 0.0 for field in FIELDS}
        T_tensor = {field: 0.0 for field in FIELDS}  # Tensor
        
        text_lower = text.lower()
        
        for field, config in self.keyword_mappings.items():
            # Number of keyword matches
            matches = sum(1 for kw in config["keywords"] if kw in text_lower)
            if matches > 0:
                # Delta increases proportionally to the number of matches
                delta[field] = config["strength"] * min(matches, 3) / 3.0
                
                # Activation strength: Activated if there's a match
                T_tensor[field] = 1.0 + 0.5 * min(matches, 3) / 3.0
                # 1.0 (Weak match) ~ 1.5 (Strong match)
        
        return delta, T_tensor


# ============================================================
# DCFEEngine: Complete Tensor Engine
# ============================================================

class DCFEEngine:
    """
    Complete DCFE Engine
    
    Integrated features:
    1. Context -> Hormone projection
    2. Hormone dynamics (Accumulation + Decay)
    3. Hormone -> Emotion tensor mapping
    4. Geometric collapse
    5. Emotion -> Hormone feedback
    """
    
    def __init__(self, 
                 emotions: Dict[str, np.ndarray],
                 base_hormone_levels: Optional[Dict[str, float]] = None,
                 kappa: float = 1.0,
                 enable_feedback: bool = True):
        """
        Parameters
        ----------
        emotions : dict
            Emotion coordinates
        base_hormone_levels : dict
            Base hormone levels (personality)
        kappa : float
            Coupling constant (Rigidity)
        enable_feedback : bool
            Whether to activate emotion->hormone feedback
        """
        self.emotions = emotions
        self.kappas = {k: kappa for k in emotions}
        self.enable_feedback = enable_feedback
        
        # Hormone state management
        self.hormone_state = HormoneState(base_hormone_levels)
        
        # Context projector
        self.projector = ContextProjector(mode='keyword')
        
        # History
        self.emotion_history = []
        self.weight_history = []
    
    def compute_weights(self, fields: Dict[str, float], T: float = 1.0) -> Dict[str, float]:
        """
        Calculate Hormone -> Emotion weights based on current field concentrations.

        Parameters
        ----------
        fields : dict
            Current hormone concentration
        T : float or dict
            Activation strength
            - float: Same for all fields (Scalar, backward compatible)
            - dict: Activation per field
            
        Returns
        -------
        weights : dict
            Weight of each emotion
        """
        # Convert T to tensor (Backward compatibility)
        if isinstance(T, (int, float)):
            T_tensor = {field: T for field in fields}
        else:
            T_tensor = T
        
        w = {k: 1.0 for k in self.emotions}
        
        for field, conc in fields.items():
            if conc <= 0 or field not in FIELDS:
                continue
            
            # Tensor: Activation strength per field
            T_field = T_tensor.get(field, 0.0)
            if T_field <= 0:
                continue  # This field is not activated
            
            f = FIELDS[field]
            
            # Pull: Pulling (Multiplicative amplification)
            for emo, s in f.get("pull", {}).items():
                if emo in w:
                    w[emo] *= (1.0 + conc * s * T_field)  # T_field
            
            # Push: Pushing (Exponential decay)
            for emo, fac in f.get("push", {}).items():
                if emo in w:
                    w[emo] *= (fac ** (conc * T_field))  # T_field
        
        return w
    
    def gaussian_force(self, p: np.ndarray, p_i: np.ndarray, 
                      w_i: float, scale: float = 1.0) -> np.ndarray:
        """
        Calculates the Gaussian gravitational force between two cognitive coordinates.
        """
        diff = p_i - p
        dist_sq = np.dot(diff, diff) * (scale ** 2)
        
        # Curvature effect
        dist_sq = dist_sq * (1.0 + LAMBDA_CONST * dist_sq)
        
        # Dynamic expansion of reach
        sigma_eff = SIGMA * (1.0 + ALPHA_SIGMA * np.log(max(w_i, 1.0)))
        sigma_sq = sigma_eff ** 2
        
        F_mag = w_i * np.exp(-dist_sq / (2 * sigma_sq)) / sigma_sq
        return F_mag * diff
    
    def select_emotion(self, fields: Dict[str, float], 
                      T: float = 1.0) -> Tuple[str, Dict]:
        """
        Geometric collapse: Select the strongest emotion
        
        Returns
        -------
        selected : str
            Selected emotion
        info : dict
            Information such as weight, force magnitude, etc.
        """
        w = self.compute_weights(fields, T)
        forces = {}
        
        p = np.array([0.0, 0.0])  # Start from origin
        
        for name, p_i in self.emotions.items():
            F_i = self.gaussian_force(p, p_i, w[name])
            F_i *= self.kappas[name]
            
            forces[name] = {
                "vector": F_i,
                "magnitude": np.linalg.norm(F_i),
                "weight": w[name]
            }
        
        # Strongest attraction
        selected = max(forces, key=lambda k: forces[k]["magnitude"])
        
        return selected, {
            "selected": selected,
            "weights": w,
            "forces": forces,
            "T": T,
        }
    
    def process_input(self, text: str, T: float = 1.0) -> Dict:
        """
        Complete pipeline: Input -> Hormone -> Emotion (Tensor version)
        
        Parameters
        ----------
        text : str
            Input text
        T : float
            Activation strength (backward compatibility)
            In practice, T_tensor is automatically generated in projector
            
        Returns
        -------
        result : dict
            Selected emotion, hormone state, weight, etc.
        """
        # [1] Context -> (Hormone delta, Activation tensor) projection
        delta, T_tensor = self.projector.project(text)
        
        # [2] Hormone accumulation
        self.hormone_state.stimulate(delta)
        
        # [3] Select emotion with current hormone state (Using T_tensor!)
        fields = self.hormone_state.get_state()
        selected, info = self.select_emotion(fields, T_tensor)
        
        # [4] Emotion -> Hormone feedback (Optional)
        if self.enable_feedback:
            self.hormone_state.apply_emotion_feedback(selected, strength=0.5)
        
        # Record history
        self.emotion_history.append(selected)
        self.weight_history.append(info["weights"].copy())
        
        return {
            "input": text,
            "hormone_delta": delta,
            "T_tensor": T_tensor,  # Activation tensor
            "hormone_state": fields.copy(),
            "selected_emotion": selected,
            "weights": info["weights"],
            "forces": {k: v["magnitude"] for k, v in info["forces"].items()},
        }
    
    def time_step(self):
        """Time passage (Hormone decay)"""
        self.hormone_state.decay()
    
    def process_sequence(self, texts: List[str], T: float = 1.0) -> List[Dict]:
        """
        Process time sequence
        
        Parameters
        --------
        texts : list
            Input texts in chronological order
        T : float
            Activation strength
            
        Returns
        -------
        results : list
            Results for each time step
        """
        results = []
        
        for t, text in enumerate(texts):
            # Process input
            result = self.process_input(text, T)
            result["time"] = t
            results.append(result)
            
            # Time passage (excluding the last one)
            if t < len(texts) - 1:
                self.time_step()
        
        return results


# ============================================================
# Scenario Definitions (Same as original)
# ============================================================

SCENARIO_EMOTIONS = {
    "Silence": {
        "Threat":  EMOTIONS_ALL["Threat"],
        "Anxiety": EMOTIONS_ALL["Anxiety"],
        "Awkward": EMOTIONS_ALL["Awkward"],
        "Boredom": EMOTIONS_ALL["Boredom"],
        "Calm":    EMOTIONS_ALL["Calm"],
        "Trust":   EMOTIONS_ALL["Trust"],
    },
    "Love": {
        "Jealous": EMOTIONS_ALL["Jealous"],
        "Awkward": EMOTIONS_ALL["Awkward"],
        "Calm":    EMOTIONS_ALL["Calm"],
        "Happy":   EMOTIONS_ALL["Happy"],
    },
    "Killing": {
        "Justice": EMOTIONS_ALL["Justice"],
        "Calm":    EMOTIONS_ALL["Calm"],
        "Pity":    EMOTIONS_ALL["Pity"],
        "Crime":   EMOTIONS_ALL["Crime"],
        "Anger":   EMOTIONS_ALL["Anger"],
    },
}

PERSONALITIES = [
    # ----- Silence Scenario -----
    # Unspecified hormones are 0.0 (No effect)
    {"id": "P1", "name": "Silence-Trust", "scenario": "Silence", "k": 0.8,
     "base": {"oxytocin": 2.0, "serotonin": 0.10, "social_norm": 0.10,
              "cortisol": 0.0, "dopamine": 0.0, "manner": 0.0}},
    {"id": "P2", "name": "Silence-Threat", "scenario": "Silence", "k": 1.2,
     "base": {"cortisol": 2.0, "social_norm": 0.10,
              "oxytocin": 0.0, "serotonin": 0.0, "dopamine": 0.0, "manner": 0.0}},
    # ----- Love Scenario -----
    {"id": "P3", "name": "Love-Happy", "scenario": "Love", "k": 1.0,
     "base": {"oxytocin": 2.00, "dopamine": 0.80, "manner": 0.00,
              "cortisol": 0.0, "serotonin": 0.0, "social_norm": 0.0}},
    {"id": "P4", "name": "Love-Jealous", "scenario": "Love", "k": 1.1,
     "base": {"cortisol": 2.00, "dopamine": 0.50, "manner": 0.00,
              "oxytocin": 0.0, "serotonin": 0.0, "social_norm": 0.0}},
    # ----- Killing Scenario -----
    {"id": "P5", "name": "Killing-Pity", "scenario": "Killing", "k": 1.2,
     "base": {"oxytocin": 2.0, "serotonin": 0.30, "social_norm": 0.00,
              "cortisol": 0.0, "dopamine": 0.0, "manner": 0.0}},
    {"id": "P6", "name": "Killing-Justice", "scenario": "Killing", "k": 1.1,
     "base": {"cortisol": 1.50, "dopamine": 0.65, "social_norm": 0.00,
              "oxytocin": 0.0, "serotonin": 0.0, "manner": 0.0}},
    {"id": "P7", "name": "Killing-Crime", "scenario": "Killing", "k": 1.1,
     "base": {"cortisol": 1.50, "dopamine": 0.65, "social_norm": 100.00,
              "oxytocin": 0.0, "serotonin": 0.0, "manner": 0.0}},
]


# ============================================================
# Visualization Functions
# ============================================================

def plot_potential_fields(personalities):
    """
    Visualize the potential field of each personality
    """
    import matplotlib.pyplot as plt
    
    print("\n" + "=" * 60)
    print("Generating potential field visualization...")
    print("=" * 60)
    
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    
    for i, p in enumerate(personalities):
        scenario_name = p["scenario"]
        emotions_dict = SCENARIO_EMOTIONS[scenario_name]
        
        # Create engine (use k per personality)
        engine = DCFEEngine(
            emotions=emotions_dict,
            base_hormone_levels=p["base"],
            kappa=p.get("k", 1.0)
        )
        
        # Calculate Weight
        fields = engine.hormone_state.get_state()
        weights = engine.compute_weights(fields, T=1.0)
        
        # Calculate potential
        Z = np.zeros_like(X)
        for row in range(X.shape[0]):
            for col in range(X.shape[1]):
                pos = np.array([X[row, col], Y[row, col]])
                V = 0
                for name, p_i in emotions_dict.items():
                    w_i = weights[name] * engine.kappas[name]
                    diff = p_i - pos
                    dist_sq = np.dot(diff, diff)
                    dist_sq = dist_sq * (1.0 + LAMBDA_CONST * dist_sq)
                    V += w_i / (dist_sq + 1e-6)
                Z[row, col] = V
        
        Z = np.log1p(Z)  # Log scale
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(X, Y, Z, levels=30, cmap='Blues', alpha=0.6)
        plt.contour(X, Y, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        
        # Display emotion coordinates
        for name, pos in EMOTIONS_ALL.items():
            is_active = name in emotions_dict
            plt.scatter(pos[0], pos[1], c='black', s=10, alpha=0.5)
            plt.text(pos[0], pos[1] + 0.15, name, fontsize=18, ha='center',
                    alpha=1.0 if is_active else 0.3, 
                    weight='bold' if is_active else 'normal')
        
        # Calculate trajectory
        start_pos = np.array([0.0, 0.0])
        forces = {}
        for name, p_i in emotions_dict.items():
            F_i = engine.gaussian_force(start_pos, p_i, weights[name])
            F_i *= engine.kappas[name]
            forces[name] = F_i
        
        selected = max(forces, key=lambda k: np.linalg.norm(forces[k]))
        target_pos = EMOTIONS_ALL[selected]
        
        # Display trajectory
        plt.plot([0, target_pos[0]], [0, target_pos[1]], 
                color='black', lw=2.5, solid_capstyle='round')
        plt.scatter(0, 0, c='white', edgecolors='black', s=120, zorder=10)
        plt.scatter(target_pos[0], target_pos[1], marker='*', 
                   c='magenta', s=350, zorder=18, edgecolors='black')
        
        plt.xlabel('Arousal (High - Low)', fontsize=18)
        plt.ylabel('Valence (Positive - Negative)', fontsize=18)
        plt.grid(True, linestyle=':', alpha=0.4)
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.tight_layout()
        
        filename = f"figure_p{i+1}_potential.pdf"
        plt.savefig(filename)
        plt.close()
        print(f"  {filename}")


def plot_bifurcation_analysis():
    """
    Visualize bifurcation analysis and kappa effect
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    print("\n" + "=" * 60)
    print("Generating bifurcation analysis and kappa effect visualization...")
    print("=" * 60)
    
    # Calculate Phase map
    resolution = 60
    oxy_vals = np.linspace(0.0, 3.5, resolution)
    cor_vals = np.linspace(0.0, 3.5, resolution)
    O, C = np.meshgrid(oxy_vals, cor_vals)
    
    scenario_emo = SCENARIO_EMOTIONS["Silence"]
    emo_keys = list(scenario_emo.keys())
    emo_map = {k: i for i, k in enumerate(emo_keys)}
    
    Z = np.zeros(O.shape)
    
    for i in range(resolution):
        for j in range(resolution):
            engine = DCFEEngine(
                emotions=scenario_emo,
                base_hormone_levels={
                    "oxytocin": O[i, j],
                    "cortisol": C[i, j],
                    "social_norm": 0.1,
                    "dopamine": 0.0,
                    "serotonin": 0.0,
                    "manner": 0.0,
                }
            )
            fields = engine.hormone_state.get_state()
            selected, _ = engine.select_emotion(fields, T=1.0)
            Z[i, j] = emo_map[selected]
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00']
    cmap = ListedColormap(colors)
    
    c1 = axes[0].pcolormesh(O, C, Z, cmap=cmap, vmin=0, vmax=len(colors)-1, shading='auto')
    axes[0].set_title('Bifurcation Phase Map (Silence Scenario)', fontsize=20, pad=10)
    axes[0].set_xlabel('Oxytocin Concentration ($T_{oxy}$)', fontsize=18)
    axes[0].set_ylabel('Cortisol Concentration ($T_{cor}$)', fontsize=18)
    axes[0].plot([0, 3.5], [0, 3.5], 'k--', alpha=0.3, lw=1)
    
    cbar = fig.colorbar(c1, ax=axes[0], ticks=np.arange(len(emo_keys)))
    cbar.ax.set_yticklabels(emo_keys, fontsize=16)
    
    # kappa effect
    x_dist = np.linspace(0, 4, 100)
    potential_low = -1.0 * np.exp(-(x_dist**2) / (2 * SIGMA**2))
    potential_high = -5.0 * np.exp(-(x_dist**2) / (2 * SIGMA**2))
    
    axes[1].plot(x_dist, potential_low, label='Kappa = 1.0 (Fluid)', 
                color='blue', lw=2, linestyle='--')
    axes[1].plot(x_dist, potential_high, label='Kappa = 5.0 (Rigid)', 
                color='darkblue', lw=3)
    axes[1].fill_between(x_dist, potential_high, potential_low, color='blue', alpha=0.1)
    
    axes[1].set_title('Effect of Kappa on Cognitive Gravity', fontsize=20, pad=10)
    axes[1].set_xlabel('Distance from Selected Emotion', fontsize=18)
    axes[1].set_ylabel('Potential Depth ($-V_{net}$)', fontsize=18)
    axes[1].legend(fontsize=18)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig('figure_bifurcation_and_kappa.pdf', bbox_inches='tight')
    plt.close()
    print(f"  figure_bifurcation_and_kappa.pdf")

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("=" * 72)
    print("DCFE - Complete Tensor Implementation")
    print("Context -> Hormone -> Emotion -> Feedback -> Temporal Dynamics")
    print("=" * 72)
    
    # ----- Test all personalities ---------------------------------------
    print("\n[Test initial selection of all Personalities]")
    print(f"{'ID':<5} {'Name':<20} {'Scenario':<10} {'Selected':^12} {'Top3'}")
    print("-" * 72)
    
    results = []
    for p in PERSONALITIES:
        scenario_name = p["scenario"]
        emotions_dict = SCENARIO_EMOTIONS[scenario_name]
        
        engine = DCFEEngine(
            emotions=emotions_dict,
            base_hormone_levels=p["base"],
            kappa=p.get("k", 1.0)
        )
        
        fields = engine.hormone_state.get_state()
        selected, info = engine.select_emotion(fields, T=p.get("k", 1.0))
        
        top3 = "  ".join(f"{k}({v:.1f})" 
                        for k, v in sorted(info["weights"].items(), 
                                          key=lambda x: -x[1])[:3])
        
        print(f"{p['id']:<5} {p['name']:<20} {scenario_name:<10} "
              f"[{selected:<10}] {top3}")
        
        results.append({
            "id": p["id"],
            "name": p["name"],
            "selected": selected,
            "weights": info["weights"]
        })
    
    # ----- Verification of bifurcation ---------------------------------------
    print("\n[Verification of bifurcation]")
    pairs = [("P1", "P2"), ("P3", "P4"), ("P5", "P6")]
    all_pass = True
    for a, b in pairs:
        r_a = next(r for r in results if r["id"] == a)
        r_b = next(r for r in results if r["id"] == b)
        ok = r_a["selected"] != r_b["selected"]
        if not ok:
            all_pass = False
        print(f"  {r_a['name']:<20} [{r_a['selected']:<10}] vs "
              f"{r_b['name']:<20} [{r_b['selected']:<10}]  "
              f"{'O' if ok else 'X'}")
    
    print(f"\n  Final: {'O ALL PASS' if all_pass else 'X FAIL'}")
    
    # ----- Detailed comparison of P6 vs P7 ---------------------------------------
    print("\n" + "=" * 72)
    print("[P6 vs P7: Time Sequence Comparison]")
    print("=" * 72)
    
    texts = [
        "My family was murdered by a criminal",
        "I am so angry and feel it's so unfair",
        "The father killed the criminal",
    ]
    
    for p_id in ["P6", "P7"]:
        p = next(p for p in PERSONALITIES if p["id"] == p_id)
        
        print(f"\n{p_id}: {p['name']}")
        print("-" * 72)
        
        engine = DCFEEngine(
            emotions=SCENARIO_EMOTIONS[p["scenario"]],
            base_hormone_levels=p["base"],
            enable_feedback=True
        )
        
        results = engine.process_sequence(texts)
        
        for r in results:
            print(f"[t={r['time']}] {r['input']}")
            print(f"  -> Emotion: {r['selected_emotion']:<10} | "
                  f"cortisol={r['hormone_state']['cortisol']:.2f}, "
                  f"social_norm={r['hormone_state']['social_norm']:.2f}")
    
    # ----- Generate visualizations ---------------------------------------
    plot_potential_fields(PERSONALITIES)
    plot_bifurcation_analysis()