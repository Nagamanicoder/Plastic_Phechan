 AI voice bot, Plastic Pehchan, helps rural communities identify and recycle plastic waste. Users get local recycler info and 'Eco-Points' for proper disposal, turning a problem into a community-led circular economy. It's a low-cost, high-impact solution that turns a common problem into community-led circularity.

✨ Built with
Languages: Python, HTML, CSS, JavaScript

Frameworks & Platforms: Flask, Ollama (gemma:2b)

APIs & Libraries: ollama Python library, requests, MediaRecorder API, Speech-to-Text (STT) API, Text-to-Speech (TTS) API

 Getting Started
 
This project uses Ollama as its local AI engine. You must have it installed and running on your machine to use the project.

Prerequisites
Ollama: Follow the official guide to install Ollama on your machine.

gemma:2b model: In your terminal, run ollama pull gemma:2b.

Installation
Clone the repository to your local machine:

git clone https://github.com/Nagamanicoder/Plastic_Phechan.git
Navigate into the project directory:

cd Plastic_Phechan

Install the necessary Python libraries:

pip install -r requirements.txt

 How to Use
Make sure your local Ollama server is running in the background.

Start the Flask server from your terminal:

python app.py
Open your web browser and navigate to http://127.0.0.1:5000 (or the address shown in your terminal) and open another tab, type http://127.0.0.1:5000/static/index.html 

Click the microphone button and speak to the bot to identify your plastic item. The bot will respond with text and audio.
