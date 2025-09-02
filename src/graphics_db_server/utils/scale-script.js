// import { NodeIO } from '@gltf-transform/node';
import { transform } from '@gltf-transform/functions';

// --- Get arguments from the command line ---
// Usage: node scale-script.js <input> <output> <factor>
const [_node, _script, inputPath, outputPath, factorStr] = process.argv;

if (!inputPath || !outputPath || !factorStr) {
  console.error('Usage: node scale-script.js <input.glb> <output.glb> <factor>');
  process.exit(1);
}

const factor = Number(factorStr);
if (isNaN(factor)) {
  console.error('Error: Scale factor must be a number.');
  process.exit(1);
}

// --- Define the scaling function ---
const rescale = (factor) => {
  return transform((node) => {
    // Only affect root nodes in the scene
    if (node.getParent()) return;

    const scale = node.getScale();
    node.setScale([
      scale[0] * factor,
      scale[1] * factor,
      scale[2] * factor,
    ]);
  });
};


// --- Read, Transform, and Write the model ---
const io = new NodeIO();

console.log(`Reading from: ${inputPath}`);

const document = await io.read(inputPath);

console.log(`Applying scale factor: ${factor}`);

await document.transform(
  rescale(factor)
);

console.log(`Writing to: ${outputPath}`);

await io.write(outputPath, document);

console.log('✅ Done.');