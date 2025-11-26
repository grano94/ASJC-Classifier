# ASJC Multi-Label Classification with SciBERT

This repository provides a Python implementation of a **multi-label classification pipeline** for **ASJC categories**, fine-tuned on SciBERT. The model is deployed on [Hugging Face](https://huggingface.co/asjc-classification/scibert_multilabel_asjc_classifier) and can be used for assessing the disciplinary orientation of research collections.

The deployment is provided in `hugging_face_deployment.ipynb`. The notebook includes both the model deployment and example inference. Further information can be retrieved from the Hugging Face model page.

---

## Citation

If you use this work, please cite:

```bibtex
@article{gusenbauer2025open,
  author    = {Gusenbauer, Michael and Endermann, Jochen and Huber, Harald and Strasser, Simon and Granitzer, Andreas-Nizar and Ströhle, Thomas},
  title     = {Fine-tuning SciBERT to enable ASJC-based assessments of the disciplinary orientation of research collections},
  journal   = {Scientometrics},
  year      = {2025}
}
```

---

## Installation

Requires **Python 3.11+**. It is recommended to use a virtual environment.

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

```python
from transformers import TextClassificationPipeline, pipeline
import torch

# --- Custom multi-label pipeline ---
class ASJCMultiLabelPipeline(TextClassificationPipeline):
    """
    Multi-label classification pipeline for ASJC categories.
    Uses a configurable threshold to return all labels with scores above the threshold.
    """
    def __init__(self, *args, **kwargs):
        # Allow threshold override; default falls back to model config
        self.threshold = kwargs.pop("threshold", None)
        super().__init__(*args, **kwargs)
        if self.threshold is None:
            self.threshold = getattr(self.model.config, "threshold", 0.3)

    def postprocess(self, model_outputs, **kwargs):
        # Convert logits to probabilities using sigmoid
        scores = torch.sigmoid(torch.tensor(model_outputs["logits"])).tolist()

        results = []
        for i, score in enumerate(scores[0]):
            if score >= self.threshold:
                label = self.model.config.id2label[i]
                results.append({"label": label, "score": float(score)})

        # Sort by descending score
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results

# --- Create the pipeline explicitly using the custom class ---
pipe = pipeline(
    task="text-classification",
    model="asjc-classification/scibert_multilabel_asjc_classifier",
    pipeline_class=ASJCMultiLabelPipeline
)

# --- Example text input ---
text = (
    "title={Jodometrie}, "
    "container_title={Fresenius' Zeitschrift für analytische Chemie, Zeitschrift für analytische Chemie}, "
    "abstract={}"
)

# --- Get multi-label predictions ---
result = pipe(text)
print(result)
```

**Example output:**

```python
[
  {'label': 'Analytical Chemistry', 'score': 0.933479368686676}, 
  {'label': 'Clinical Biochemistry', 'score': 0.9108470678329468}, 
  {'label': 'Biochemistry', 'score': 0.494137704372406}
]
```

**Expected labels:**

- Clinical Biochemistry  
- Analytical Chemistry  

---

## Notes

- This pipeline is **multi-label**, meaning multiple ASJC categories can be predicted for a single text.  
- The threshold for selecting labels can be adjusted in the pipeline constructor:  

```python
pipe = pipeline(
    task="text-classification",
    model="asjc-classification/scibert_multilabel_asjc_classifier",
    pipeline_class=ASJCMultiLabelPipeline,
    threshold=0.5  # Example threshold
)
```