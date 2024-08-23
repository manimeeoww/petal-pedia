from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import traceback

app = Flask(__name__)

# Update CORS to match your front-end URL
CORS(app, resources={r"/*": {"origins": "https://petal-pedia-bsv8-j452etjiu-manimeeowws-projects.vercel.app"}})

# Load the model
model = tf.keras.models.load_model('mymodel.h5')

# Define flower categories and detailed info
categories = {
    'Chamomile': {
        'scientific_name': 'Matricaria chamomilla (German chamomile) or Chamaemelum spp. (Roman chamomile)',
        'origin': 'Native to Europe, western Asia, and northern Africa',
        'family': 'Asteraceae (Compositae)',
        'symbolism': 'Known for its soothing properties, it symbolizes relaxation, peace, and healing. It\'s often used in herbal remedies and teas.',
        'link': 'https://en.wikipedia.org/wiki/Chamomile',
        'image': 'https://i.pinimg.com/564x/26/d8/e8/26d8e8760154d8229ef3ae6b3a078f6b.jpg'
    },
    'Chrysanthemum': {
        'scientific_name': 'Chrysanthemum spp. (Various species)',
        'origin': 'Native to Asia and northeastern Europe',
        'family': 'Asteraceae (Compositae)',
        'symbolism': 'Symbolizes longevity, loyalty, joy, and optimism. In some cultures, it\'s associated with death and used in funeral rituals.',
        'link': 'https://en.wikipedia.org/wiki/Chrysanthemum',
        'image': 'https://i.pinimg.com/564x/20/63/25/206325bae5f105956f255845eb42bbcc.jpg'
    },
    'French Marigold': {
        'scientific_name': 'Tagetes patula',
        'origin': 'Native to Mexico and Central America',
        'family': 'Asteraceae (Compositae)',
        'symbolism': 'Symbolizes passion and creativity. It\'s also associated with positive energy and good fortune.',
        'link': 'https://en.wikipedia.org/wiki/Marigold',
        'image': 'https://i.pinimg.com/564x/20/63/25/206325bae5f105956f255845eb42bbcc.jpg'
    },
    'Lavender': {
        'scientific_name': 'Lavandula spp. (Various species)',
        'origin': 'Native to the Mediterranean region, Africa, and India',
        'family': 'Lamiaceae (mint family)',
        'symbolism': 'Represents calmness, serenity, and grace. It\'s often associated with cleanliness and relaxation.',
        'link': 'https://en.wikipedia.org/wiki/Lavender',
        'image': 'https://i.pinimg.com/564x/20/63/25/206325bae5f105956f255845eb42bbcc.jpg'
    },
    'Lotus': {
        'scientific_name': 'Nelumbo nucifera',
        'origin': 'Native to Asia and Australia',
        'family': 'Nelumbonaceae',
        'symbolism': 'Represents purity, enlightenment, and rebirth in various cultures, particularly in Asian religions like Buddhism and Hinduism.',
        'link': 'https://en.wikipedia.org/wiki/Nelumbo_nucifera',
        'image': 'https://i.pinimg.com/564x/54/1f/e7/541fe7d7d8a5f3db275336ae7894dc17.jpg'
    },
    'Passion Flower': {
        'scientific_name': 'Passiflora spp. (Various species)',
        'origin': 'Native to tropical and subtropical regions of the Americas',
        'family': 'Passiflorine (passionflower family)',
        'symbolism': 'Represents faith, passion, and spirituality. The intricate structure of its flower is often seen as a symbol of the Passion of Christ.',
        'link': 'https://en.wikipedia.org/wiki/Passion_flower',
        'image': 'https://i.pinimg.com/564x/eb/bb/9d/ebbb9d014988fc6e8f6990c959b0761f.jpg'
    },
    'Poppy': {
        'scientific_name': 'Eschscholzia californica',
        'origin': 'Native to California, USA, and adjacent areas of Mexico',
        'family': 'Papaveraceae (poppy family)',
        'symbolism': 'Represents imagination, success, and remembrance. It\'s also associated with relaxation and restful sleep.',
        'link': 'https://en.wikipedia.org/wiki/California_poppy',
        'image': 'https://i.pinimg.com/564x/7b/94/aa/7b94aa41ccccba116748f8e58aa24e2c.jpg'
    },
    'Purple coneflower': {
        'scientific_name': 'Echinacea purpurea',
        'origin': 'Native to eastern and central North America',
        'family': 'Asteraceae (Compositae)',
        'symbolism': 'Known for its medicinal properties and immune-boosting benefits. Represents strength, health, and healing.',
        'link': 'https://en.wikipedia.org/wiki/Echinacea',
        'image': 'https://i.pinimg.com/564x/84/28/b6/8428b6a28d45bd3467326c94d7953a7c.jpg'
    },
    'Rose': {
        'scientific_name': 'Rosa spp. (Various species)',
        'origin': 'Roses are native to various world regions, depending on the species.',
        'family': 'Rosaceae',
        'symbolism': 'Roses have numerous meanings depending on color and context. Generally, they symbolize love, beauty, and passion.',
        'link': 'https://en.wikipedia.org/wiki/Rose',
        'image': 'https://i.pinimg.com/564x/a1/68/cf/a168cfe922bf440d68dea91b80bc62ed.jpg'
    },
    'Sunflower': {
        'scientific_name': 'Helianthus annuus',
        'origin': 'Native to North America',
        'family': 'Asteraceae (Compositae)',
        'symbolism': 'Represents adoration, loyalty, and longevity. The sunflower\'s vibrant yellow color symbolizes vitality and happiness.',
        'link': 'https://en.wikipedia.org/wiki/Common_sunflower',
        'image': 'https://i.pinimg.com/564x/fa/33/cc/fa33ccdefc6f4c5c696d6af334cfc5a5.jpg'
    }
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Process the image
        img = Image.open(file.stream).resize((224, 224))  # Resize as per your model's requirement
        img_array = np.array(img) / 255.0  # Normalize if required
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Map prediction to flower category
        flower_name = list(categories.keys())[predicted_class]
        flower_info = categories.get(flower_name, {})

        # Return prediction and flower details
        return jsonify({
            'prediction': flower_name,
            'scientific_name': flower_info['scientific_name'],
            'origin': flower_info['origin'],
            'family': flower_info['family'],
            'symbolism': flower_info['symbolism'],
            'link': flower_info['link'],
            'image': flower_info['image']
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run()
