import cv2

cap = cv2.VideoCapture(0)  # Open the default camera (0)

while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    # Perform your ML processing here on 'frame'
    fontText = cv2.FONT_HERSHEY_COMPLEX

    # text, coordinate, font, size of text,color,thickness of font
    cv2.putText(frame,'Hello There', (100,100), fontText, 2, (255,255,255), 3)  
    # out.write(frame)
    cv2.imshow('Webcam', frame)  # Display the frame

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()