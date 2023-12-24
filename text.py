import cv2
import pytesseract
from nltk.sentiment import SentimentIntensityAnalyzer


pytesseract.pytesseract.tesseract_cmd = r'/Users/chira/anaconda3/bin/tesseract'


cap = cv2.VideoCapture(0)


sia = SentimentIntensityAnalyzer()

while True:

    ret, frame = cap.read()


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)


    custom_config = r'--oem 3 --psm 6 outputbase digits'
    text = pytesseract.image_to_string(binary, config=custom_config)


    nltk_sentiment = sia.polarity_scores(text)


    cv2.imshow('Webcam', frame)


    print("Detected Text:")
    print(text)


    print("Sentiment Analysis (NLTK):")
    print(f"Compound Score: {nltk_sentiment['compound']}")
    if nltk_sentiment['compound'] >= 0.05:
        print("Sentiment: Positive")
    elif nltk_sentiment['compound'] <= -0.05:
        print("Sentiment: Negative")
    else:
        print("Sentiment: Neutral")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
