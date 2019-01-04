def send_message(message, token=YOUR_TOKEN):
    import requests

    line_notify_token = token
    line_notify_api = 'https://notify-api.line.me/api/notify'

    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}
    line_notify = requests.post(line_notify_api, data=payload, headers=headers)

    print("message sent")
