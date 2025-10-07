# 🗣️ Gani Hausa TTS Backend (Flask)

This backend provides a **Text-to-Speech (TTS)** API for generating **Hausa audio** responses using the **CLEAR-Global Hausa TTS model**.  
It powers the Gani AI assistant by converting Hausa text into high-quality spoken audio.

---

## 🚀 Features

- Converts Hausa text to natural speech.
- Uses **Hugging Face CLEAR-Global Hausa TTS model**.
- Returns audio in `.wav` format.
- Simple REST API endpoint: `/api/tts`.

---

## 🧰 Requirements

Make sure you have the following installed:

- **Python 3.9+**
- **pip** (Python package manager)
- **virtualenv** (recommended)

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ogayanfe/hausa-tts-server.git
cd hausa-tts-backend
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
# Activate it:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` File

In the project root, create a `.env` file and add the following:

```bash
FLASK_ENV=development
FLASK_APP=app.py
```

(Optional) Add your Hugging Face token if needed:

```bash
HUGGINGFACE_HUB_TOKEN=your_hf_token_here
```

---

## 🧩 Project Structure

```
app/
├── __init__.py
├── routes.py        # Defines the /api/tts route
config.py
model.py             # Loads and runs the Hausa TTS model
requirements.txt
app.py               # Entry point for Flask app
```

---

## ▶️ Running the Server Locally

### Option 1: Using Flask’s built-in server

```bash
flask run
```

### Option 2: Using Python directly

```bash
python app.py
```

You should see:

```
 * Running on http://127.0.0.1:5000
```

---

## 🔉 API Endpoint

### `POST /api/tts`

**Description:**  
Converts Hausa text to audio and returns a `.wav` file.

**Request Body:**

```json
{
  "text": "Ina kwana?"
}
```

**Response:**

- Returns binary WAV audio data.
- You can play it directly or convert it to Base64 in your frontend.

**Example using cURL:**

```bash
curl -X POST http://127.0.0.1:5000/api/tts \
     -H "Content-Type: application/json" \
     -d '{"text": "Ina kwana?"}' \
     --output output.wav
```

---

## 🧪 Testing

After running the server, test with the cURL command above or integrate it with your frontend.

---

## 🛠️ Troubleshooting

- If you see `gunicorn: command not found`, install it with:
  ```bash
  pip install gunicorn
  ```
- If the TTS model download is slow, ensure your internet connection is stable or pre-download the model from [Hugging Face](https://huggingface.co/CLEAR-Global/TWB-Voice-Hausa-TTS-1.0).

---

## 📜 License

This project uses the CLEAR Global Hausa TTS model, released under its respective license.  
You are free to use it for research or non-commercial applications unless stated otherwise.
