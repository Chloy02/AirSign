# ASL Detection API

A FastAPI service for real-time ASL (American Sign Language) word detection using Roboflow's hosted model.

## Features

- **Real-time ASL Detection**: Processes individual video frames to detect ASL words
- **Highest Confidence Output**: Returns only the most confident detection as text
- **Meeting App Integration**: Designed for live video call ASL translation
- **CORS Enabled**: Ready for web application integration

## API Endpoints

### `POST /detect-asl`
- **Purpose**: Detect ASL words from an image frame
- **Input**: Image file (PNG, JPG, etc.)
- **Output**: JSON with detected word
- **Example Response**: `{"detected_word": "hello"}`

### `GET /health`
- Health check endpoint

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
uvicorn main:app --reload
```

3. Test the API:
```bash
curl -X POST "http://localhost:8000/detect-asl" -F "image=@your_image.png"
```

## Deployment on Render

1. Connect your GitHub repository to Render
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `./start.sh`
4. Deploy!

## Usage in Meeting App

Send video frames to the `/detect-asl` endpoint at 5-second intervals for live ASL translation.
