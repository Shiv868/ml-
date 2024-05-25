let model;
const metadata = {
    labels: ["one piece", "college"] // Replace with your actual class labels
};

const loadModel = async () => {
    model = await tf.loadLayersModel('model/model.json');
    document.getElementById('prediction').innerText = "Model loaded!";
};

const loadImage = (event) => {
    const image = document.getElementById('input-image');
    image.src = URL.createObjectURL(event.target.files[0]);
    image.onload = () => classifyImage(image);
};

const classifyImage = async (image) => {
    const tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .expandDims();
    const predictions = await model.predict(tensor).data();
    const results = Array.from(predictions)
        .map((p, i) => ({ probability: p, className: metadata.labels[i] }))
        .sort((a, b) => b.probability - a.probability);
    document.getElementById('prediction').innerText = 
        `Prediction: ${results[0].className} - Probability: ${results[0].probability.toFixed(2)}`;
};

window.onload = loadModel;
