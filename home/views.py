import json
import os
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.conf import settings

import requests
import torch
from vits2.utils.task import load_vocab
from vits2.utils.task import load_checkpoint
from vits2.utils.hparams import get_hparams_from_file
from vits2.model.models import SynthesizerTrn
from vits2.text import tokenizer
from scipy.io.wavfile import write
import gdown

audio_url="static/audio/out.wav"
audio_path = "home/static/audio/out.wav"

def download_from_google_drive(file_id):
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    output = 'G_7000.pth'  # You can specify the output filename here
    gdown.download(url, output, quiet=False)
    return output

def get_home(request):
    return render(request, 'home.html')

def clear_text(text: str) -> torch.LongTensor:
    text_norm = tokenizer(text, vocab, hps.data.text_cleaners, language="vi")
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def translate(text):
    text_clear = clear_text(text)
    x_tst = text_clear.unsqueeze(0)
    x_tst_lengths = torch.LongTensor([text_clear.size(0)])
    out = net_g.infer(x_tst, x_tst_lengths, noise_scale=0, noise_scale_w=0, length_scale=1)
    audio = out[0][0, 0].data.cpu().float().numpy()
    write(audio_path, 22050, (audio * 32767).astype("int16"))

def translate_api_viettel(text):
    url = "https://viettelai.vn/tts/speech_synthesis"
    payload = json.dumps({
    "text": text,
    "voice": "hcm-minhquan",
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
    with open("home/static/audio/out.wav", "wb") as file:
        file.write(response.content)
@csrf_exempt
def process_text(request):
    if request.method == "POST":
        text_input = request.POST.get("text_input", "")
        voice_option = request.POST.get("voiceOption", "customTrain")  # Get the selected voice option

        if voice_option == "customTrain":
            translate(text_input)  # Call the custom-trained voice synthesis function
        else:
            translate_api_viettel(text_input)
        try:
            return JsonResponse({"audio_url": audio_url})
        except Exception as e:
            return JsonResponse({"error": f"Lỗi vị trí số 1: {str(e)}"}, status=500)

# Load model
vocab = load_vocab("vits2/config/vocab2.txt")
hps = get_hparams_from_file("vits2/config/config.yaml")
filter_length = hps.data.n_mels if hps.data.use_mel else hps.data.n_fft // 2 + 1
segment_size = hps.train.segment_size // hps.data.hop_length
net_g = SynthesizerTrn(342, filter_length, segment_size, **hps.model)
_ = net_g.eval()
check_point = "vits2/logs/G_13000.pth"
# Check if the checkpoint exists, otherwise download it
if check_point is None or not os.path.exists(check_point):
    # Assuming you have a function or method to download from Google Drive
    check_point = download_from_google_drive('1V9ou7CON54GY3SFekxHAYK0XGR8LJFAi')
_ = load_checkpoint(check_point, net_g, None)
