import coremltools as ct
from PIL import Image
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
coreml_dir = os.path.join(SCRIPT_DIR, "CoreML_Models")

# Dynamically discover mlpackage model filenames from CoreML_Models
model_filenames = [
    fname for fname in os.listdir(coreml_dir)
    if fname.endswith(".mlpackage")
]

# Load a sample image
if len(sys.argv) > 1:
    input_image_path = sys.argv[1]
else:
    input_image_path = os.path.join("samples", "inputs", "1.jpg")
input_image = Image.open(input_image_path).convert("RGB").resize((1024, 1024))

for filename in model_filenames:
    print(f"Testing model: {filename}")
    # Load Core ML model
    model_path = os.path.join(coreml_dir, filename)
    mlmodel = ct.models.MLModel(model_path)

    # Print the model specification for verification
    print(mlmodel.get_spec())

    # Run a prediction. Provide the PIL Image since the model expects an ImageType.
    result = mlmodel.predict({"input_image": input_image})

    # Extract the output image
    output_image = result["output_image"]
    output_image.show()
    print(f"Finished testing {filename}\n")
