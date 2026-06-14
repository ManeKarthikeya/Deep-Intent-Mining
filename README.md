# Deep Intent Mining from Student Doubt Messages

A Streamlit NLP project that detects:
- Topic
- Hidden intent
- Emotion
- Sentiment
- Aspect cues
- Learning-risk signal

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Input supported
- Single chat message
- Multiple messages pasted line-by-line
- CSV upload with a `text` column

## Notes
- Topic and intent are now computed with cached TF-IDF resources for much faster response time.
- Optional Hugging Face models can still be enabled for emotion and sentiment.
- No NLP system can guarantee 100% accuracy on every real-world message, but this version is designed to be fast, stable, and easy to demo.
