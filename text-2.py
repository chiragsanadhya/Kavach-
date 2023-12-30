import cv2
import easyocr
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en','hi'])

# Initialize NLTK SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Perform text detection and recognition
    results = reader.readtext(frame)

    # Display the original frame
    cv2.imshow('Webcam', frame)

    # Perform sentiment analysis for each detected text
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # Extract the text for sentiment analysis
        sentiment_text = text

        # Perform sentiment analysis using NLTK
        nltk_sentiment = sia.polarity_scores(sentiment_text)

        # Draw a red bounding box only for negative sentiment
        if nltk_sentiment['compound'] <= -0.05:
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
            cv2.putText(frame, f"Negative: {text}", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display the result
    cv2.imshow('Webcam with Text Detection and Sentiment Analysis', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
