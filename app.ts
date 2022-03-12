const bodyPix = require('@tensorflow-models/body-pix');

const img = document.getElementById('image');

async function loadAndPredict() {
  const net = await bodyPix.load({
    architecture: 'ResNet50',
    outputStride: 32,
    quantBytes: 2
  });
  const segmentation = await net.segmentPerson(img);
  console.log(segmentation);
}
loadAndPredict();