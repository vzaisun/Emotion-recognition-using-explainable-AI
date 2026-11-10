# Does Grad-CAM Catch a Shortcut?

Facial emotion recognition on FER2013 with Grad-CAM, plus an experiment that tests whether Grad-CAM reveals a shortcut the model is known to use.

Final project for DSC 291: Interpretable Machine Learning, UC San Diego (Fall 2026). Vijay Sundar (vsundar@ucsd.edu)

## Notebooks

| Notebook | What it does | Colab |
|---|---|---|
| `Emotion_recognition_using_gradcam.ipynb` | Trains a CNN on FER2013, evaluates it, and visualizes Grad-CAM (examples, per-class average maps, layer comparison) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vzaisun/Emotion-recognition-using-explainable-AI/blob/main/Emotion_recognition_using_gradcam.ipynb) |
| `Gradcam_shortcut_experiment.ipynb` | Plants a shortcut marker in the training data and checks whether Grad-CAM finds it; also runs a weight-randomization sanity check and a pixel-deletion test | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vzaisun/Emotion-recognition-using-explainable-AI/blob/main/Gradcam_shortcut_experiment.ipynb) |

## The experiment

A small checkerboard marker is added to the top-left corner of 80% of the *Sad* training images. Three models are trained with the same settings:

- **Clean**: unmodified data
- **Strong**: marker at full contrast
- **Faint**: marker blended in at 30% contrast

Since we planted the shortcut, we know the right answer. We add the marker to non-Sad test faces and see if the prediction flips to Sad. On those flipped images, we measure how much of the Grad-CAM heatmap lands on the marker.

## Results

Classifier (clean model): **68.1%** test accuracy with flip test-time augmentation (majority-class baseline: 24.5%).

| Model | Test acc. | Flip rate | Grad-CAM energy on marker | Peak on marker |
|---|---|---|---|---|
| Clean (control) | 67.5% | 0.9% | 2–6% | 0.5–19% |
| Strong | 61.7% | 82.5% | 18–25% | 82–99% |
| Faint | 67.1% | 21.8% | 13–19% | 86–92% |

The energy chance level is 4.3%. Ranges cover the two Grad-CAM layers (block 3 and block 4).

Main findings:
- Grad-CAM does locate the shortcut, well above chance and the clean control.
- It understates the shortcut. The marker alone caused these predictions, yet it gets only 13–25% of the heatmap, and most of the map still covers the face.
- Grad-CAM passes the weight-randomization sanity check (Adebayo et al., 2018).
- In the deletion test, Grad-CAM's pixel ranking is much better than random for the shortcut models but close to random for the clean model.

## How to run

1. Open a notebook in Colab and select a GPU runtime (Runtime > Change runtime type > T4 GPU).
2. **Data:** FER2013 from Kaggle. The shortcut notebook downloads `deadskull7/fer2013` automatically through `kagglehub` if `CSV_PATH` doesn't exist. The main notebook expects the CSV path set in `CSV_PATH`. Both `fer2013.csv` and `icml_face_data.csv` work as they are.
3. Run all cells. The main notebook takes about 20–30 minutes. The shortcut experiment trains three models, about 50 minutes in total. Its models are saved to Google Drive, so a re-run skips models that already finished.

Requirements: TensorFlow 2.x with Keras 3, NumPy, pandas, scikit-learn, SciPy, matplotlib, seaborn. All are preinstalled on Colab.

## References

- Selvaraju et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. ICCV 2017.
- Adebayo et al. Sanity Checks for Saliency Maps. NeurIPS 2018.
- Adebayo et al. Post Hoc Explanations may be Ineffective for Detecting Unknown Spurious Correlation. ICLR 2022.
- Goodfellow et al. Challenges in Representation Learning: A Report on Three Machine Learning Contests. ICONIP 2013.
