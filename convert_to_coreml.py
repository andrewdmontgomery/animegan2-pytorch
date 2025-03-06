#!/usr/bin/env python3

import torch
import torchvision.io as io
import coremltools as ct
import os
from CoreMLWrapper import CoreMLWrapper

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
weights_dir = os.path.join(SCRIPT_DIR, "weights")
coreml_dir = os.path.join(SCRIPT_DIR, "CoreML_Models")

def convert_to_coreml(
    pytorch_weights_filename="face_paint_512_v1.pt",
    coreml_package_filename="FacePaintV1.mlpackage"
):
    pytorch_model_path = os.path.join(weights_dir, pytorch_weights_filename)
    coreml_model_path = os.path.join(coreml_dir, coreml_package_filename)

    # Instantiate model and load weights.
    model = CoreMLWrapper()
    model.generator.load_state_dict(torch.load(pytorch_model_path, map_location="cpu", weights_only=True))
    model.eval()

    # Create a dummy input of the fixed shape used during export.
    example_input = torch.randn(1, 3, 1024, 1024)

    # Use a tensor from a real image instead of a randomly generated tensor
    #
    # input_image = Image.open("/path/to/sample.jpg").resize((1024, 1024))
    # to_tensor = transforms.ToTensor()
    # input_tensor = to_tensor(input_image)
    # input_batch = input_tensor.unsqueeze(0)

    traced_model = torch.jit.trace(model, example_input)

    # Supporting dynamic images
    #
    # Set the input_shape to use RangeDim for each dimension.
    # input_shape = ct.Shape(shape=(1,
    #                             3,
    #                             ct.RangeDim(lower_bound=25, upper_bound=2048, default=1024),
    #                             ct.RangeDim(lower_bound=25, upper_bound=2048, default=1024)))
    
    input_shape = ct.EnumeratedShapes(shapes=[[1, 3, 256, 256],
                                              [1, 3, 1024, 1024]],
                                              default=[1, 3, 256, 256])

    # Convert to Core ML
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="input_image",
                shape=[1, 3, 1024, 1024],
                scale=2/255.0,
                bias=[-1.0, -1.0, -1.0],
            )
        ],
        outputs=[
            ct.ImageType(
                name="output_image",
                # scale=1.0,      # CoreML requires a scale of 1.0
                # bias=[0.0, 0.0, 0.0],
                color_layout=ct.colorlayout.RGB
            )
        ]
    )

    mlmodel.save(coreml_model_path)
    print(f"Core ML model saved to {coreml_model_path}")

if __name__ == "__main__":
    if not os.path.exists(coreml_dir):
        os.makedirs(coreml_dir)

    convert_to_coreml("face_paint_512_v1.pt", "FacePaintV1.mlpackage")
    convert_to_coreml("face_paint_512_v2.pt", "FacePaintV2.mlpackage")
    convert_to_coreml("paprika.pt", "Paprika.mlpackage")
    convert_to_coreml("celeba_distill.pt", "CelebA_Distill.mlpackage")