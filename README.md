# ASJC Multi-Label Classification with SciBERT

This repository contains the resources used to deploy and run inference with a multi-label ASJC classification model fine-tuned on SciBERT. The model itself is hosted on Hugging Face at:

➡️ https://huggingface.co/asjc-classification/scibert_multilabel_asjc_classifier

The model can be used to assess the disciplinary orientation of research documents or collections.

The notebook hugging_face_deployment.ipynb documents the process used to upload and deploy the model to Hugging Face. An example inference workflow is provided in model_inference.py. Additional details and usage guidance are available on the Hugging Face model page.

---

## Citation

If you use this work, please cite:

```bibtex
@article{gusenbauer2025open,
  author    = {Gusenbauer, Michael and Endermann, Jochen and Huber, Harald and Strasser, Simon and Granitzer, Andreas-Nizar and Ströhle, Thomas},
  title     = {Fine-tuning SciBERT to enable ASJC-based assessments of the disciplinary orientation of research collections},
  journal   = {Scientometrics},
  year      = {2025},
  doi       = {10.1007/s11192-025-05490-0}
}
```

---

## Installation

It is recommended to use a virtual environment.

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.\.venv\Scripts\activate

# Activate virtual environment (Linux / Mac)
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

To run a sample inference using the deployed Hugging Face model, execute:

```python
python model_inference.py

# --- Example model input ---

# text = (
#     "title={Jodometrie}, "
#     "container_title={Fresenius' Zeitschrift für analytische Chemie, Zeitschrift für analytische Chemie}, "
#     "abstract={}"
# )


# --- Example model output ---

# [
#   {'label': 'Analytical Chemistry', 'score': 0.933479368686676}, 
#   {'label': 'Clinical Biochemistry', 'score': 0.9108470678329468}, 
#   {'label': 'Biochemistry', 'score': 0.494137704372406}
# ]


# --- True labels ---

# - Clinical Biochemistry  
# - Analytical Chemistry  
```

---

## Notes

- This pipeline is **multi-label**, meaning multiple ASJC categories can be predicted for a single text including Title, Container Title and / or Abstract.  
- The threshold for selecting labels can be adjusted in the pipeline constructor:  

```python
pipe = pipeline(
    task="text-classification",
    model="asjc-classification/scibert_multilabel_asjc_classifier",
    pipeline_class=ASJCMultiLabelPipeline,
    threshold=0.5  # Example threshold
)
```