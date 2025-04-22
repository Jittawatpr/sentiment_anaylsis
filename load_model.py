import joblib
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn


def create_model(input_size, hidden_size, num_layers, dropout_rate=0.5):
    class LSTMWithDropout(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
            super(LSTMWithDropout, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.dropout = nn.Dropout(dropout_rate)
            self.linear = nn.Linear(hidden_size, 3)

        def forward(self, x):
            if x.dim() == 2:  # [batch_size, input_size] -> [batch_size, 1, input_size]
                x = x.unsqueeze(1)
            output, (hidden, cell) = self.lstm(x)
            output = output[:, -1, :]  # Take last time step
            output = self.dropout(output)
            output = self.linear(output)
            return output, (hidden, cell)  # Returns tuple

    return LSTMWithDropout(input_size, hidden_size, num_layers, dropout_rate)


# Load the model
def load_model(file_path, input_size, hidden_size, num_layers, dropout_rate):
    model = create_model(input_size, hidden_size, num_layers, dropout_rate)
    model.load_state_dict(torch.load(file_path))
    model.eval()
    print(f"Model loaded from {file_path}")
    return model


def get_wangchanberta_embeddings(
    texts, model_name="airesearch/wangchanberta-base-att-spm-uncased", max_length=128
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModel.from_pretrained(model_name)
    encoded_inputs = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length
    )
    with torch.no_grad():
        model_output = model(**encoded_inputs)
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings.numpy()


def get_prediction(text):
    model_file = "best_model_lstm.pth"

    embeddings = get_wangchanberta_embeddings(text)

    # Load the model for inference
    input_size = embeddings.shape[1]  # Assuming embeddings are already defined
    hidden_size = 256
    num_layers = 3
    dropout_rate = 0.18
    loaded_model = load_model(
        model_file, input_size, hidden_size, num_layers, dropout_rate
    )

    # Predict sentiment for a new text
    embedding_tensor = torch.tensor(embeddings, dtype=torch.float32)
    with torch.no_grad():
        output = loaded_model(embedding_tensor)
        if isinstance(output, tuple):
            output = output[0]
        best_scaler = joblib.load("scaler.pkl")
        predicted_scores = best_scaler.inverse_transform(output.cpu().numpy())

        return predicted_scores
