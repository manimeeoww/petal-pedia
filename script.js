// script.js

async function getPrediction() {
    const fileInput = document.getElementById('file-input');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('https://petal-pedia.vercel.app/predict', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            document.getElementById('prediction-result').innerText = 'Prediction: ' + result.prediction;

            const flowerInfoDiv = document.getElementById('flower-info');
            flowerInfoDiv.innerHTML = `
                <img src="${result.image}" alt="${result.prediction}">
                <h2>${result.prediction}</h2>
                <p><strong>Scientific Name:</strong> ${result.scientific_name}</p>
                <p><strong>Origin:</strong> ${result.origin}</p>
                <p><strong>Family:</strong> ${result.family}</p>
                <p><strong>Symbolism:</strong> ${result.symbolism}</p>
                <p><a href="${result.link}" target="_blank">Learn More</a></p>
            `;
        } else {
            const error = await response.json();
            document.getElementById('prediction-result').innerText = 'Error: ' + error.error;
        }
    } catch (error) {
        document.getElementById('prediction-result').innerText = 'Error: ' + error.message;
    }
}
