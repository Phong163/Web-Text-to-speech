import requests
import json
url = "https://viettelai.vn/tts/speech_synthesis"
payload = json.dumps({
"text": "Văn bản cần đọc",
"voice": "hcm-diemmy",
"speed": 1,
"tts_return_option": 3,
"token": "d21644415aa3383d460ebaf81b86f42b",
"without_filter": False
})
headers = {
'accept': '*/*',
'Content-Type': 'application/json'
}
response = requests.request("POST", url, headers=headers, data=payload)
with open("audio.mp3", "wb") as file:
    file.write(response.content)