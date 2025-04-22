from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict

from load_model import get_prediction

app = FastAPI()


class InputText(BaseModel):
    text: str = Field(..., min_length=1, example="ระบบการทำงานที่ กฟผ")


class PredictionResponse(BaseModel):
    input_text: str
    sentiment_score: Dict[str, float]  # e.g. [0.73, 0.1, 0.17]


@app.post("/predict-sentiment", response_model=PredictionResponse)
def predict(input_data: InputText):
    labels = ["positive", "neutral", "negative"]

    # raw_scores = [random.random() for _ in labels]
    # total = sum(raw_scores)
    # probs = [round(score / total, 4) for score in raw_scores]
    output = get_prediction(input_data.text)
    return output

    # sentiment_scores = dict(zip(labels, probs))

    # return PredictionResponse(
    #     input_text=input_data.text,
    #     sentiment_score=sentiment_scores,
    # )
