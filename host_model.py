import glob
import os
import io
import datetime
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ElementTree
import re
import threading

from matplotlib import image as mpimg
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, BitsAndBytesConfig
from PIL import Image
from flask import Flask, request, jsonify, render_template_string, send_file
from llm_client import (
    MessageHistory,
)

processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# Need to save it this way or else it doesn't load (some issue with the names not matching)
# processor.save_pretrained("modelq2/allenai.Molmo-7B-D-0924.1721478b71306fb7dc671176d5c204dc7a4d27d7")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="fp4",
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto',
    quantization_config=quantization_config
)

image_buffer = io.BytesIO()
done_writing = False


def parse_points(xml_string: str) -> list[tuple[float, float]]:
    points = []

    # Extract XML segments
    point_segments = re.findall(r'<point.*?>.*?</point>', xml_string)
    points_segments = re.findall(r'<points.*?>.*?</points>', xml_string)

    for segment in point_segments:
        root = ElementTree.fromstring(segment)
        x = float(root.get('x'))
        y = float(root.get('y'))
        text = root.text
        points.append((x, y))

    for segment in points_segments:
        root = ElementTree.fromstring(segment)
        index = 1
        while True:
            x = root.get(f'x{index}')
            y = root.get(f'y{index}')
            if x is None or y is None:
                break
            x = float(x)
            y = float(y)
            text = root.text.strip()  # Assuming all points share the same text
            points.append((x, y))
            index += 1

    return points


def draw_dot(image, point, radius):
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    mask = (x - point[0]) ** 2 + (y - point[1]) ** 2 <= radius ** 2
    image[mask] = [0.94, 0.01, 0.99, 1.0]  # Set dot color to red
    return image


def generate_image(message: str) -> None:
    global image_buffer, done_writing

    points = parse_points(message)

    done_writing = False

    if len(points) != 0:
        print("POINTS:", points)

    list_of_screenshots = glob.glob('/mnt/c/Users/m/AppData/Roaming/.minecraft/screenshots/*.png')
    latest_screenshot = max(list_of_screenshots, key=os.path.getctime)
    image = mpimg.imread(latest_screenshot)  # Load your image

    # Get image dimensions
    height, width, _ = image.shape

    # global image
    # points = [(100, 100), (150, 200), (200, 300)]  # Example points
    updated_image = image.copy()
    for point in points:
        px = int(point[0] / 100 * width)
        py = int(point[1] / 100 * height)

        updated_image = np.array(updated_image)
        updated_image = draw_dot(updated_image, (px, py), 10)

    fig, ax = plt.subplots(figsize=(32, 18))
    ax.imshow(updated_image)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    image_buffer = io.BytesIO()
    plt.savefig(image_buffer, format='png')
    image_buffer.seek(0)
    plt.close(fig)

    done_writing = True


def chatbot(messages: str) -> str:
    """
    Call the loaded language model with the given messages and return the response.

    This function should take a string of messages in the following format:
    USER: message
    ASSISTANT: message
    ...
    """
    print(messages)

    list_of_screenshots = glob.glob('/mnt/c/Users/m/AppData/Roaming/.minecraft/screenshots/*.png')
    latest_screenshot = max(list_of_screenshots, key=os.path.getctime)

    print("LATEST SCREENSHOT:", latest_screenshot)

    inputs = processor.process(
        images=[Image.open(latest_screenshot)],
        text=messages,
    )

    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    # with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )

    # only get generated tokens; decode them to text
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print("MODEL:", generated_text)

    threading.Thread(target=generate_image, args=[generated_text]).start()

    return generated_text


app = Flask(__name__)


@app.route('/')
def home():
    return render_template_string('''
        <!doctype html>
        <html style="height: 100%; margin: 0;">
        <head>
            <title>Model Output Viewer</title>
            <style>
                body, html {
                    height: 100%;
                    margin: 0;
                }
                .fullscreen-image {
                    position: absolute;
                    top: 0;
                    bottom: 0;
                    left: 0;
                    right: 0;
                    width: 100%;
                    height: 100%;
                    object-fit: contain;
                }
            </style>
        </head>
        <body>
            <img src="{{ url_for('update_image') }}" class="fullscreen-image">
            <script>
                setInterval(function() {
                    document.querySelector('img').src = '{{ url_for('update_image') }}' + '?' + new Date().getTime();
                }, 2000);
            </script>
        </body>
        </html>
    ''')


@app.route('/update_image')
def update_image():
    if image_buffer is None or not done_writing:
        return send_file(io.BytesIO(), mimetype='image/png')

    # copy the image buffer
    buf = io.BytesIO()
    image_buffer.seek(0)
    buf.write(image_buffer.read())
    buf.seek(0)

    response = send_file(buf, mimetype='image/png')
    expires = datetime.datetime.now() + datetime.timedelta(seconds=60)
    response.headers['Expires'] = expires.strftime('%a, %d %b %Y %H:%M:%S GMT')

    return response


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    print("Completion request received.")

    data = request.get_json()
    if not data or "model" not in data or "messages" not in data:
        return jsonify({"error": "Invalid request format"}), 400

    model_name = "allenai/Molmo-7B-D-0924"
    messages = data["messages"]

    history = MessageHistory()
    for message in messages:
        history.add(message["role"], message["content"])

    # Convert message history to the required string format for the chatbot function
    message_str = history.chat_str()

    assistant_response = chatbot(message_str)

    return jsonify(
        {
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": assistant_response},
                    "finish_reason": "stop",
                }
            ],
        }
    )


if __name__ == "__main__":
    generate_image("")

    app.run(host="0.0.0.0", port=1234)
