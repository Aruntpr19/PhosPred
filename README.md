# PhosPred

**PhosPred** is a deep learning tool for predicting **anion binding sites** in proteins.  
It uses state-of-the-art protein language models (ESM2) to classify binding sites as one of five anions: phosphate, sulfate, chloride, nitrate, or carbonate.

---

## 🚀 Features

- **ESM2 Integration** – Leverages ESM2 protein language model for high-quality sequence embeddings
- **Multi-class Classification** – Predicts anion binding site: phosphate, sulfate, chloride, nitrate, carbonate
- **Flexible Input** – Accepts protein sequences and binding site residue indices
- **GPU Support** – Automatically detects and utilizes available GPU
- **Easy Integration** – Use as a Python library or a command-line tool

---

## 📋 Table of Contents

- [Installation](#️installation)
- [Quick Start](#️quick-start)
- [Usage](#️usage)
- [API Reference](#️api-reference)
- [Examples](#️examples)
- [Training Your Own Model](#️training-your-own-model)
- [Contributing](#️contributing)
- [License](#️license)

---

## 🛠️ Installation

### From PyPI (coming soon)
```bash
pip install protein-binding-classifier
```

### From Source
```bash
git clone https://github.com/Aruntpr19/PhosPred.git
cd protein-binding-classifier
pip install -e .
```

### Requirements

All required packages are listed in the `requirements.txt` file.  
To install them manually:
```bash
pip install -r requirements.txt
```

---

## ⚡ Quick Start (Python)

```python
from protein_classifier import ProteinBindingSiteClassifier

# Initialize classifier
classifier = ProteinBindingSiteClassifier()

# Example protein sequence
sequence = "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL"

# Binding site residue indices (1-based)
binding_sites = [23, 45, 67, 89, 102]

# Predict binding site type
results = classifier.predict(sequence, binding_sites)
print(f"Predicted probabilities: {results}")
```

---

## 📖 Usage

### Command Line Interface (CLI)

#### Basic prediction
```bash
python -m protein_classifier --sequence "YOUR_SEQUENCE" --binding_sites "1,2,3,4,5"
```

#### With custom model
```bash
python -m protein_classifier \
  --sequence "YOUR_SEQUENCE" \
  --binding_sites "1,2,3,4,5" \
  --model_path "path/to/model.pth"
```

#### Batch processing
```bash
python -m protein_classifier \
  --input_file "sequences.json" \
  --output_file "results.json"
```

---

## 🧪 Python API

```python
from protein_classifier import ProteinBindingSiteClassifier

# Load a pre-trained model (optional)
classifier = ProteinBindingSiteClassifier(model_path="path/to/model.pth")

# Single prediction
results = classifier.predict(sequence, binding_site_indices)

# Batch predictions
sequences = ["SEQUENCE1", "SEQUENCE2"]
binding_sites_list = [[1, 2, 3], [4, 5, 6]]
batch_results = classifier.predict_batch(sequences, binding_sites_list)
```

---

## 🧠 Training Your Own Model

Custom training scripts will be added in the next release.  
For now, you can fine-tune on your own dataset using `classifier.train(...)` (WIP).


---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
