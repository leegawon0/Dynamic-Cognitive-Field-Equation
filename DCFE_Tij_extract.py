"""
Object Variation with Modulation: Preserving the Sign Structure
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt
from typing import List


class ModulatedObjectAnalyzer:
    """Extract T_ij with modulation for object variation"""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Initialize model"""
        print(f"Loading {model_name}...")
        
        if torch.backends.mps.is_available(): 
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            output_hidden_states=True,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
        ).to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def get_embedding(self, text: str) -> torch.Tensor:
        """Extract embedding"""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        token_idx = min(len(inputs['input_ids'][0]) - 2, len(inputs['input_ids'][0]) - 1)
        return outputs.hidden_states[-1][0, token_idx, :]
    
    def get_token_mean_embedding(self, texts: List[str]) -> torch.Tensor:
        """Get mean embedding"""
        embeddings = [self.get_embedding(text) for text in texts]
        return torch.stack(embeddings).mean(dim=0)
    
    def get_specific_token_embeddings(self, text: str, token: str = "killing") -> List[torch.Tensor]:
        """Extract embeddings of specific token from all layers"""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Find LAST occurrence
        token_idx = None
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] == token or tokens[i].lower() == token.lower():
                token_idx = i
                break
        
        if token_idx is None:
            token_idx = max(0, len(tokens) - 2)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return [layer[0, token_idx, :] for layer in outputs.hidden_states[2:-1]]
    
    def compute_modulated_pull(
        self,
        h_token: torch.Tensor,
        v_crime: torch.Tensor,
        v_object: torch.Tensor,
        v_aggravating: torch.Tensor,
        v_mitigating: torch.Tensor
    ) -> tuple:
        """
        Compute positive crime pull with bounded severity modulation.

        Design intention:
        - Any killing scenario should retain positive pull toward the crime attractor.
        - The object/context should only modulate the strength of that pull.
        - Modulation is therefore not interpreted as negative/positive force, but as
          a bounded severity factor.

        Returns: (base_pull, severity_score, modulated_pull)
        """
        # Base distance and pull toward the crime attractor.
        distance = torch.norm(h_token - v_crime)
        base_pull = 1.0 / (distance + 1e-8)

        # Directional similarity from the object concept to aggravating/mitigating concepts.
        # This is not a push force; it only adjusts the magnitude of crime pull.
        object_norm = v_object / (torch.norm(v_object) + 1e-8)
        aggravating_norm = v_aggravating / (torch.norm(v_aggravating) + 1e-8)
        mitigating_norm = v_mitigating / (torch.norm(v_mitigating) + 1e-8)

        aggravating_sim = torch.dot(object_norm, aggravating_norm)
        mitigating_sim = torch.dot(object_norm, mitigating_norm)

        # Convert relative severity into a bounded positive factor.
        # severity_score is in approximately [0, 1].
        severity_score = torch.sigmoid(aggravating_sim - mitigating_sim)

        # Keep the final crime pull always positive.
        # Range: base_pull * [0.75, 1.25]
        severity_factor = 0.75 + 0.50 * severity_score
        modulated_pull = base_pull * severity_factor

        return base_pull.item(), severity_score.item(), modulated_pull.item()
    
    def extract_with_modulation(
        self,
        scenarios: List[tuple]
    ) -> pd.DataFrame:
        """
        Extract T_ij with modulation for different objects
        
        Args:
            scenarios: List of (name, text, object_words)
        """
        
        print(f"\nExtracting 'crime' embedding...")
        v_crime = self.get_token_mean_embedding(["crime", "illegal", "unlawful", "criminal act", "punishable"])

        print(f"Extracting aggravating / mitigating context embeddings...")
        v_aggravating = self.get_token_mean_embedding(["innocent", "victim", "blameless", "vulnerable"])
        v_mitigating = self.get_token_mean_embedding(["guilty", "murderer", "criminal", "justice", "self defense"])
        
        results = []
        
        for scenario_name, scenario_text, object_words in scenarios:
            print(f"\n[{scenario_name}]")
            print(f"  Scenario: '{scenario_text}'")
            print(f"  Object concept: {object_words}")
            
            # Get object embedding
            v_object = self.get_token_mean_embedding(object_words)
            
            # Get 'killing' token embeddings at all layers
            h_layers = self.get_specific_token_embeddings(scenario_text, token="killing")
            
            for layer_idx, h_token in enumerate(h_layers):
                base_pull, modulation, modulated_pull = self.compute_modulated_pull(
                    h_token, v_crime, v_object, v_aggravating, v_mitigating
                )
                
                results.append({
                    'scenario': scenario_name,
                    'layer': layer_idx,
                    'base_pull': base_pull,
                    'modulation': modulation,
                    'modulated_pull': modulated_pull
                })
        
        return pd.DataFrame(results)


def visualize_with_modulation(df: pd.DataFrame, save_path: str = 'modulated_objects.png'):
    """Visualize modulated pulls"""
    
    scenarios = df['scenario'].unique()
    colors = plt.cm.Set2(range(len(scenarios)))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Base Pull (same for all)
    ax1 = axes[0]
    for i, scenario in enumerate(scenarios):
        scenario_data = df[df['scenario'] == scenario]
        ax1.plot(scenario_data['layer'], scenario_data['base_pull'], 
                marker='o', label=scenario, linewidth=2.5, color=colors[i], markersize=6)
    
    ax1.set_title('Base Pull (1/distance)\nBefore Modulation', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Pull Strength')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Modulation (by object)
    ax2 = axes[1]
    for i, scenario in enumerate(scenarios):
        scenario_data = df[df['scenario'] == scenario]
        ax2.plot(scenario_data['layer'], scenario_data['modulation'], 
                marker='s', label=scenario, linewidth=2.5, color=colors[i], markersize=6)
    
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('Severity Modulation by Object\n(bounded positive factor)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Severity Score [0, 1]')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. Modulated Pull (final result with sign)
    ax3 = axes[2]
    for i, scenario in enumerate(scenarios):
        scenario_data = df[df['scenario'] == scenario]
        ax3.plot(scenario_data['layer'], scenario_data['modulated_pull'], 
                marker='D', label=scenario, linewidth=2.5, color=colors[i], markersize=6)
    
    ax3.set_title('Modulated Pull\n(Base × bounded severity factor)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Pull Strength')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved: {save_path}")
    plt.show()


def print_modulation_summary(df: pd.DataFrame):
    """Print summary with modulation effects"""
    
    scenarios = df['scenario'].unique()
    
    print("\n" + "="*100)
    print("MODULATED OBJECT ANALYSIS: How object affects crime pull")
    print("="*100)
    
    for scenario in scenarios:
        scenario_data = df[df['scenario'] == scenario]
        
        print(f"\n[{scenario}]")
        print(f"  Base Pull (distance-based):")
        print(f"    Mean: {scenario_data['base_pull'].mean():.6f}")
        
        print(f"  Severity modulation (object effect):")
        print(f"    Mean: {scenario_data['modulation'].mean():+.6f}")
        print(f"    Range: [{scenario_data['modulation'].min():+.6f}, {scenario_data['modulation'].max():+.6f}]")
        
        print(f"  Modulated Pull (final):")
        print(f"    Mean: {scenario_data['modulated_pull'].mean():.6f}")
        print(f"    Range: [{scenario_data['modulated_pull'].min():.6f}, {scenario_data['modulated_pull'].max():.6f}]")
    
    print(f"\n[T_ij POTENTIAL]")
    print("-" * 100)
    
    # Compute pairwise T_ij
    scenario_list = list(scenarios)
    for i in range(len(scenario_list)):
        for j in range(i+1, len(scenario_list)):
            s1 = scenario_list[i]
            s2 = scenario_list[j]
            
            pull_1 = df[df['scenario'] == s1]['modulated_pull'].mean()
            pull_2 = df[df['scenario'] == s2]['modulated_pull'].mean()
            t_ij = pull_2 - pull_1
            
            print(f"T_ij ({s2} - {s1}): {t_ij:+.6f}")
    
    print("\n" + "="*100)


def main():
    """Main execution"""
    
    analyzer = ModulatedObjectAnalyzer()
    
    scenarios = [
        ("Innocent person", "Father killing an innocent person", 
         ["innocent", "victim", "blameless"]),
        
        ("The murderer", "Father killing the murderer", 
         ["murderer", "killer", "guilty"]),
        
        ("The murderer of family", "Father killing the murderer of his family", 
         ["murderer", "killer", "guilty", "justice"]),
        
        ("A criminal", "Father killing a criminal", 
         ["criminal", "guilty", "wrongdoer"]),
        
        ("For revenge", "Father killing for revenge", 
         ["revenge", "retribution", "payback"])
    ]
    
    print("\n" + "="*100)
    print("EXTRACTING T_ij WITH MODULATION: Object effects on crime pull")
    print("="*100)
    
    df = analyzer.extract_with_modulation(scenarios)
    
    print("\nGenerating visualization...")
    visualize_with_modulation(df)
    
    print_modulation_summary(df)


if __name__ == "__main__":
    main()