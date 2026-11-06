import smtplib
from email.message import EmailMessage
from config import EMAIL_FROM, EMAIL_APP_PASSWORD, EMAIL_TO

msg = EmailMessage()
msg["Subject"] = "Digital Eye test"
msg["From"] = EMAIL_FROM
msg["To"] = EMAIL_TO
msg.set_content("This is a test email from Digital Eye.")

with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
    server.starttls()
    server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
    server.send_message(msg)
print("Email sent")