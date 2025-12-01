'''
 # @ Author: Andreas-Nizar Granitzer
 # @ Create Time: 2025-11-27 09:39:39
 # @ Modified by: Andreas-Nizar Granitzer
 # @ Modified time: 2025-11-27 09:40:03
 # @ Description: Model inference script for deployed Hugging Face model.
 '''

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
                label = self.model.config.id2label[(i)]
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
    "title={Dose optimization of β-lactams antibiotics in pediatrics and adults: A systematic review}, "
    "container_title={Frontiers in Pharmacology}, "
    "abstract={Background: β-lactams remain the cornerstone of the empirical therapy to treat various bacterial infections. This systematic review aimed to analyze the data describing the dosing regimen of β-lactams.Methods: Systematic scientific and grey literature was performed in accordance with Preferred Items for Systematic Reviews and Meta-Analysis (PRISMA) guidelines. The studies were retrieved and screened on the basis of pre-defined exclusion and inclusion criteria. The cohort studies, randomized controlled trials (RCT) and case reports that reported the dosing schedule of β-lactams are included in this study.Results: A total of 52 studies met the inclusion criteria, of which 40 were cohort studies, 2 were case reports and 10 were RCTs. The majority of the studies (34/52) studied the pharmacokinetic (PK) parameters of a drug. A total of 20 studies proposed dosing schedule in pediatrics while 32 studies proposed dosing regimen among adults. Piperacillin (12/52) and Meropenem (11/52) were the most commonly used β-lactams used in hospitalized patients. As per available evidence, continuous infusion is considered as the most appropriate mode of administration to optimize the safety and efficacy of the treatment and improve the clinical outcomes.Conclusion: Appropriate antibiotic therapy is challenging due to pathophysiological changes among different age groups. The optimization of pharmacokinetic/pharmacodynamic parameters is useful to support alternative dosing regimens such as an increase in dosing interval, continuous infusion, and increased bolus doses.}"
)

# --- Get multi-label predictions ---
result = pipe(text)
print(result)

# Predicted labels:
# [
#   {'label': 'Pharmacology (medical)', 'score': 0.9922493696212769}, 
#   {'label': 'Pharmacology', 'score': 0.902540922164917}
# ]

# Expected labels:
# - Pharmacology (medical)
# - Pharmacology