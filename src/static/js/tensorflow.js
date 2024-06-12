characters = ['0','1','2','3','4','5','6','7','8','9',
'A','B','C','D','E','F','G','H','I','J','K','L','M','N',
'O','P','Q','R','S','T','U','V','W','X','Y','Z',
'a','b','d','e','f','g','h','n','q','r','t']


const modelUrl = "https://raw.githubusercontent.com/vinccrn/ScribMaps/main/src/static/model.json";
let model;

document.addEventListener('DOMContentLoaded', (event) => {
    (async function(){  
        console.log("model loading...");
    
        try {
            // Check if the model is already stored in localStorage
            if (localStorage.getItem('modelTopology') && localStorage.getItem('modelWeightData')) {
                const modelTopology = JSON.parse(localStorage.getItem('modelTopology'));
                const weightData = new Uint8Array(JSON.parse(localStorage.getItem('modelWeightData')));
    
                model = await tf.loadLayersModel(tf.io.fromMemory(modelTopology, weightData));
                console.log("Model loaded from localStorage");
            } else {
                model = await tf.loadLayersModel(modelUrl);
                console.log("Model loaded from URL");
    
                // Save model to localStorage
                const modelArtifacts = await model.save(tf.io.withSaveHandler(async (artifacts) => artifacts));
                localStorage.setItem('modelTopology', JSON.stringify(modelArtifacts.modelTopology));
                localStorage.setItem('modelWeightData', JSON.stringify(Array.from(new Uint8Array(modelArtifacts.weightData))));
                console.log("Model saved to localStorage");
            }
    
            setInterval(async function(){     
                let tensor = preprocessCanvas(document.getElementById('canvas')); 
                console.log(tensor);   
                let predictions = await model.predict(tensor).data();
                console.log(predictions);
                let results = Array.from(predictions);       
                console.log(results);
                predictShow(results);
                clearCanvas(canvas);
            }, 4000);
    
        } catch (error) {
            console.error("Error loading model:", error);
        }
    })(); 
});

function clearCanvas(canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function preprocessCanvas(image) { 
    const ctx = image.getContext('2d');
    const width = image.width;
    const height = image.height;

    // Get the image data
    let imageData = ctx.getImageData(0, 0, width, height);
    let data = imageData.data;

    // Create a temporary canvas to resize the image
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = 28;
    tempCanvas.height = 28;

    // Draw the current image onto the temporary canvas
    tempCtx.drawImage(image, 0, 0, 28, 28);

    // Get the image data from the temporary canvas
    imageData = tempCtx.getImageData(0, 0, 28, 28);
    data = imageData.data;

    // Create a 1D array of grayscale values
    let grayData = [];
    for (let i = 0; i < data.length; i += 4) {
        let r = data[i];
        let g = data[i + 1];
        let b = data[i + 2];
        let alpha = data[i + 3];
        // Convert to grayscale
        let gray = (0.299 * r + 0.587 * g + 0.114 * b) * (alpha / 255);
        grayData.push(gray);
    }

    // Create a 2D tensor from the grayscale values and reshape it to 28x28x1
    let tensor = tf.tensor4d(grayData, [1, 28, 28, 1]);
    return tensor.div(255.0);
}

function predictShow(data){
    try {
        var max = data[0];    
        var maxIndex = 0;     
        for (var i = 1; i < data.length; i++) {        
            if (data[i] > max) {            
                maxIndex = i;            
                max = data[i];  
            }
        }
        console.log(max);
        console.log(maxIndex);
        const resultElement = document.getElementById('result');
        const confidenceElement = document.getElementById('confidence');
        const recognizedTextElement = document.getElementById('recognizedText');
        
        if (!resultElement || !confidenceElement || !recognizedTextElement) {
            throw new Error("Result, confidence or recognizedText element not found in DOM");
        }
        
        const recognizedCharacter = characters[maxIndex];
            if (max > 0.017) {  // Set your desired threshold here
                resultElement.innerHTML = recognizedCharacter;  
                confidenceElement.innerHTML = "Confidence: " + (max * 100).toFixed(2) + "%";
                // Append the recognized character to the input field
                recognizedTextElement.value += recognizedCharacter;
            } else {
                resultElement.innerHTML = '';  
                confidenceElement.innerHTML = "Confidence: 0%";
            }
    } catch (error) {
        console.error("Error in predictShow function:", error);
    }
}