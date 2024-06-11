characters = ['0','1','2','3','4','5','6','7','8','9',
'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


var base_url = window.location.origin;
let model;
(async function(){  
    console.log("model loading...");  
    model = await tf.loadLayersModel('______________')
    console.log("model loaded..");
})();

function preprocessCanvas(image) { 
    let tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([28, 28])
        .mean(2)
        .expandDims(2)
        .expandDims()
        .toFloat(); 
    return tensor.div(255.0);
}

setInterval(function(){     
    var imageData = canvas.toDataURL();    
    let tensor = preprocessCanvas(canvas); 
    console.log(tensor)   
    let predictions = await model.predict(tensor).data();
    let results = Array.from(predictions);       
    console.log(results);
    predictShow(results)
}, 3000);

function predictShow(data){
    var max = data[0];    
    var maxIndex = 0;     
    for (var i = 1; i < data.length; i++) {        
      if (data[i] > max) {            
        maxIndex = i;            
        max = data[i];  

      }
    }
    console.log(max)
    console.log(maxIndex)
    document.getElementById('result').innerHTML = characters[maxIndex];  
    document.getElementById('confidence').innerHTML = "Confidence: "+(max*100).toFixed(2) + "%";
}