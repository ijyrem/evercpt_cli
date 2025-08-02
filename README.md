##  EVERCPT: Extracellular Vesicles Enrichment of RNA Cargo Prediction Tool
This tool allows the prediction of the enrichment of RNAs in EVs based on its seqeunce features.

The tool uses a machine learning model trained on RNA sequences to predict whether an RNA is likely to be enriched in extracellular vesicles (EVs) or retained in cells. The tool also provides detailed features extracted from the RNA sequence, including length, GC and, AT percentages, secondary structure features, and nucleotide frequencies as well as RNA binding protein (RBP) motifs.

---
### ⚠️ Check 
If you don't have conda already installed, you can install it here: [Install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/#regular-installation)

---

## 🧬 Installation (requires conda)

### 1. Clone the repo
```bash
git clone https://github.com/ijyrem/evercpt_cli.git
cd evercpt_cli

```

### 2. Install dependencies
```bash
chmod +x setup.sh
./setup.sh
```

### 3. Run EVERCPT
```bash
conda activate evenv
python evercpt.py -i fasta.fa -t [mrna|circ] -o output.csv
```
---

## 📝 Cite
In progress.

---


