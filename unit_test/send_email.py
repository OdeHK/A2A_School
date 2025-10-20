import base64
from email.message import EmailMessage
import os
from typing import List, Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/gmail.compose"]


def _token_path() -> str:
  """Return absolute path to token.json stored alongside this file."""
  return os.path.join(os.path.dirname(__file__), "token.json")


def send_message(sender: str, recipients: List[str], subject: str, content: str) -> Optional[List[dict]]:
  """
  Gửi email qua Gmail API.

  Tham số:
  - sender: email người gửi (str)
  - recipients: danh sách email người nhận (list[str])
  - subject: tiêu đề email (str)
  - content: nội dung văn bản thuần (str)

  Trả về:
  - List[dict] phản hồi từ Gmail API cho mỗi email đã gửi, hoặc None nếu có lỗi.
  """
  if not sender or not isinstance(sender, str):
    raise ValueError("sender must be a non-empty string")
  if not recipients or not isinstance(recipients, list):
    raise ValueError("recipients must be a non-empty list of emails")

  # Load authorized credentials from local token file.
  creds = Credentials.from_authorized_user_file(_token_path(), SCOPES)

  try:
    service = build("gmail", "v1", credentials=creds)
    sent_messages = []

    # Duyệt qua từng recipient để gửi email riêng lẻ
    for recipient in recipients:
      if not recipient or not isinstance(recipient, str):
        print(f"Skipping invalid recipient: {recipient}")
        continue
        
      message = EmailMessage()
      message.set_content(content)
      message["From"] = sender
      message["To"] = recipient
      message["Subject"] = subject

      # Encode message per Gmail API requirements.
      encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
      create_message = {"raw": encoded_message}

      # Send the message.
      sent = (
          service.users().messages().send(userId="me", body=create_message).execute()
      )
      sent_messages.append(sent)
      # Optional: print message id for visibility in manual runs.
      print(f"Message sent to {recipient}. Id: {sent.get('id')}")

    return sent_messages if sent_messages else None
  except HttpError as error:
    # Log error to stdout for test visibility and return None.
    print(f"An error occurred: {error}")
    return None


if __name__ == "__main__":
  # Demo usage (commented out to avoid accidental sends):
  send_message(
      sender="congvuthanh1209@gmail.com",
      recipients=["tranbadong9471@gmail.com", "congvuthanh1209@gmail.com"],
      subject="Test Subject",
      content="Hello from Gmail API",
  )
