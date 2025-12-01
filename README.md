# ASJC Multi-Label Classification with SciBERT

This repository contains the resources used to [deploy](./hugging_face_deployment.ipynb), [test](./data/test/) and [run inference](./model_inference.py) with a multi-label ASJC classification model fine-tuned on SciBERT. The model itself is hosted on Hugging Face at:

➡️ https://huggingface.co/asjc-classification/scibert_multilabel_asjc_classifier

The model can be used to assess the disciplinary orientation of research documents or collections. Additional details and usage guidance are available on the Hugging Face model page.

---

## Citation

If you use this work, please cite:

```bibtex
@article{Gusenbauer.2025,
author = {Gusenbauer, Michael and Endermann, Jochen and Huber, Harald and Strasser, Simon and Granitzer, Andreas-Nizar and Ströhle, Thomas},
year = {2025},
title = {Fine-tuning SciBERT to enable ASJC-based assessments of the disciplinary orientation of research collections},
keywords = {All Science Journal Classification;Disciplinary coverage;Fine-tuning;multi-label classification;SciBERT;Transformer-based language models},
issn = {0138-9130},
journal = {Scientometrics},
doi = {10.1007/s11192-025-05490-0},
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

# --- Example text input ---
# text = (
#     "title={Dose optimization of β-lactams antibiotics in pediatrics and adults: A systematic review}, "
#     "container_title={Frontiers in Pharmacology}, "
#     "abstract={Background: β-lactams remain the cornerstone of the empirical therapy to treat various bacterial infections. This systematic review aimed to analyze the data describing the dosing regimen of β-lactams.Methods: Systematic scientific and grey literature was performed in accordance with Preferred Items for Systematic Reviews and Meta-Analysis (PRISMA) guidelines. The studies were retrieved and screened on the basis of pre-defined exclusion and inclusion criteria. The cohort studies, randomized controlled trials (RCT) and case reports that reported the dosing schedule of β-lactams are included in this study.Results: A total of 52 studies met the inclusion criteria, of which 40 were cohort studies, 2 were case reports and 10 were RCTs. The majority of the studies (34/52) studied the pharmacokinetic (PK) parameters of a drug. A total of 20 studies proposed dosing schedule in pediatrics while 32 studies proposed dosing regimen among adults. Piperacillin (12/52) and Meropenem (11/52) were the most commonly used β-lactams used in hospitalized patients. As per available evidence, continuous infusion is considered as the most appropriate mode of administration to optimize the safety and efficacy of the treatment and improve the clinical outcomes.Conclusion: Appropriate antibiotic therapy is challenging due to pathophysiological changes among different age groups. The optimization of pharmacokinetic/pharmacodynamic parameters is useful to support alternative dosing regimens such as an increase in dosing interval, continuous infusion, and increased bolus doses.}"
# )

# --- Get multi-label predictions ---
# result = pipe(text)
# print(result)

# Predicted labels:
# [
#   {'label': 'Pharmacology (medical)', 'score': 0.9922493696212769}, 
#   {'label': 'Pharmacology', 'score': 0.902540922164917}
# ]

# Expected labels:
# - Pharmacology (medical)
# - Pharmacology 
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