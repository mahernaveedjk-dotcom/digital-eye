import os
import smtplib
import time
from email.message import EmailMessage

from config import EMAIL_FROM, EMAIL_APP_PASSWORD, EMAIL_TO


def send_email(subject, body, image_path=None):
    """Send an email, retrying up to three times."""

    for attempt in range(1, 4):
        try:
            message = EmailMessage()

            message["Subject"] = subject
            message["From"] = EMAIL_FROM
            message["To"] = EMAIL_TO

            message.set_content(body)

            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as file:
                    message.add_attachment(
                        file.read(),
                        maintype="image",
                        subtype="jpeg",
                        filename=os.path.basename(image_path)
                    )

            with smtplib.SMTP(
                "smtp.gmail.com",
                587,
                timeout=30
            ) as server:
                server.starttls()
                server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
                server.send_message(message)

            print("Email sent successfully.")
            return True

        except Exception as error:
            print(f"Email attempt {attempt} failed: {error}")
            time.sleep(5)

    print("Email could not be sent. Incident remains saved locally.")
    return False