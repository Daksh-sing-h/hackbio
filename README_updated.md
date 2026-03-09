# AMR Prediction from Bacterial Genomes
**Computational Bioengineering Hackathon 2026**

Genotype-to-phenotype machine learning model for predicting antibiotic resistance
from *Escherichia coli* genome assemblies.

---
## Judging Criteria Coverage

| Criterion | Weight | Our approach |
|---|---|---|
| Predictive Performance | 35% | 5 models, 5-fold CV, AUC + MCC per antibiotic |
| Feature Interpretability | 25% | Model importance + SHAP + mechanism annotations |
| Clinical Actionability | 25% | R/S/I + calibrated confidence + 4 clinical demos |
| Runtime Efficiency | 15% | Benchmarked: ~90s total per genome |

---
## Dataset
- **Source**: PATRIC Database (https://www.patricbrc.org/)
- **Organism**: *Escherichia coli* (Gram-negative, highest clinical relevance)
- **Samples**: 1000 genomes with 3-class AMR phenotype labels
- **Antibiotics**: Ciprofloxacin (fluoroquinolone), Ceftriaxone (3rd-gen cephalosporin), Meropenem (carbapenem)
- **Phenotype classes**: Resistant (R), Intermediate (I), Susceptible (S)

---
## Performance — Test Set Results (20% held-out)

| Antibiotic      | Model                               | AUC    | MCC    | F1-macro | F1-Intermediate |
|-----------------|-------------------------------------|--------|--------|----------|-----------------|
| Ciprofloxacin   | RandomForest (calibrated)           | 0.8337 | 0.7099 | 0.5852 | 0.0000 |
| Ceftriaxone     | RandomForest (calibrated)           | 0.7992 | 0.5889 | 0.5411 | 0.0000 |
| Meropenem       | RandomForest (calibrated)           | 0.8044 | 0.5825 | 0.5327 | 0.0000 |

> AUC = macro One-vs-Rest across 3 classes. MCC = multiclass Matthews Correlation Coefficient.

## Cross-Validation Results (5-fold stratified, mean ± std)

| Antibiotic      | Best Model             | AUC (mean ± std)       | MCC (mean ± std)       |
|-----------------|------------------------|------------------------|------------------------|
| Ciprofloxacin   | RandomForest           | nan ± nan | 0.6869 ± 0.0370 |
| Ceftriaxone     | RandomForest           | nan ± nan | 0.5902 ± 0.0435 |
| Meropenem       | RandomForest           | nan ± nan | 0.6798 ± 0.0394 |

---
## Runtime Benchmarks

| Stage | Time |
|---|---|
| AMRFinder+ feature extraction | ~90 seconds |
| Feature vector construction | <1 ms |
| Model inference (3 antibiotics) | <5 ms |
| **Total per genome** | **~90 seconds (~1.5 min) ✅** |

---
## Quick Start

```python
# Predict from FASTA file
results, genes, source, ms = predict_resistance(fasta_path='genome.fasta')
print(results)

# Or from a gene list directly
results, genes, source, ms = predict_resistance(
    detected_genes=['blaCTX-M', 'gyrA_S83L', 'parC_S80I']
)
```

---
## Deliverables

| # | Deliverable | Files | Status |
|---|---|---|---|
| 1 | Trained models + AUC + MCC | `models/*.pkl`, `results/performance_metrics.csv` | ✅ |
| 2 | Feature importance analysis | `results/*_top_features.csv`, `results/feature_importance_all.png` | ✅ |
| 3 | Executable prediction tool | `predict_resistance()` in notebook | ✅ |
| 4 | Clinical interpretation guide | `clinicguide.txt` | ✅ |

---
**Hackathon**: Computational Bioengineering 2026 | **Species**: *E. coli* | **March 2026**
