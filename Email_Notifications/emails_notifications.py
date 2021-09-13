import smtplib
import os

EMAIL_ADDRESS = 'stockprediction.notification@gmail.com'
EMAIL_PASSWORD = 'zmffldterjzsryjv'

with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
    smtp.ehlo()
    smtp.starttls()
    smtp.ehlo()

    smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)

    subject = 'Prediction'
    body = 'Buy'

    msg = f'Subject: {subject}\n\n{body}'

    smtp.sendmail(EMAIL_ADDRESS, 'sanchezgamezjc@gmail.com', msg)
