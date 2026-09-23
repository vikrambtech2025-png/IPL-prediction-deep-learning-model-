# IPL Prediction — Deep Learning Model

Predicts **IPL match outcomes** using a **PyTorch** deep learning model — includes the training/architecture code, a pre-processing pipeline, a trained model checkpoint, and a Streamlit dashboard deployable on Render.

## Project structure

| File | Description |
|---|---|
| `model.py` | PyTorch model architecture & training logic |
| `preprocess.py` | Feature engineering / data preparation |
| `models/saved_model.pth` | Trained model checkpoint |
| `app.py` | Streamlit prediction UI |
| `render.yaml` | Render deployment configuration |

## Quick start

```bash
pip install -r requirements.txt
python model.py           # train (or load the saved checkpoint)
streamlit run app.py      # launch the dashboard
```

## Stack

PyTorch · pandas · NumPy · scikit-learn · Matplotlib · Streamlit

## License

No license specified — for learning/reference use.