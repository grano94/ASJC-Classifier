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