from transformers import AutoImageProcessor, FlaxResNetModel, ResNetConfig
from PIL import Image
import requests
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
import jax.random
from enzyme_ad.jax import JaXPipeline, hlo_opts, enzyme_jax_ir
from test_utils import *
import llama

# Define the pipelines as per your framework
pipelines = [
    ("JaX", None, CurBackends),
    ("JaXPipe", JaXPipeline(), CurBackends),
    (
        "HLOOpt",
        JaXPipeline(
            "inline{default-pipeline=canonicalize max-iterations=4},"
            + "canonicalize,cse,enzyme-hlo-opt,cse"
        ),
        CurBackends,
    ),
    ("PartOpt", JaXPipeline(llama.partialopt), CurBackends),
    ("DefOpt", JaXPipeline(hlo_opts()), CurBackends),
    (
        "EqSat",
        JaXPipeline(
            "inline{default-pipeline=canonicalize max-iterations=4},"
            + "equality-saturation-pass"
        ),
        CurBackends,
    ),
]

# Load the image processor and model
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = FlaxResNetModel.from_pretrained("microsoft/resnet-50")

# Define the ResNet forward pass
def resnet_forward(pixel_values, weights):
    """ ResNet forward pass """
    model.params = weights  # Set the weights to the loaded weights
    outputs = model(pixel_values=pixel_values)
    return outputs.last_hidden_state

# Define the test class for ResNet using absltest
class ResNetTest(EnzymeJaxTest):
    def setUp(self):
        # Create a ResNetConfig object
        self.config = ResNetConfig.from_pretrained("microsoft/resnet-50")

        # Load the weights for ResNet model (Microsoft pretrained)
        self.weights = model.params

        # Sample image from COCO dataset
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        pil_image = Image.open(requests.get(url, stream=True).raw)
        np_image = np.array(pil_image)

        # Process the image
        inputs = processor(images=np_image, return_tensors="np")
        pixel_values = inputs["pixel_values"]  # Extract the actual NumPy array

        # Run the ResNet forward pass with the processed input
        self.logits = resnet_forward(pixel_values, self.weights)

        # Setup for the test harness
        self.fn = resnet_forward
        self.name = "resnet"
        self.count = 1000
        self.revprimal = False
        self.AllPipelines = pipelines
        self.AllBackends = CurBackends

        # Input and output setup
        self.ins = [pixel_values, self.weights]
        self.dins = [pixel_values, self.weights]
        self.douts = [self.logits]
        self.tol = 5e-5

        # Optimized function using enzyme_jax
        self.opt_fn = enzyme_jax_ir(
            pipeline_options=JaXPipeline(
                "inline{default-pipeline=canonicalize max-iterations=4},equality-saturation-pass"
            )
        )(self.fn)

    def test_optimized_forward(self):
        # Test the optimized function against the standard one
        opt_outputs = self.opt_fn(*self.ins)
        recursive_check(self, opt_outputs, self.logits, self.tol)

# Running the tests with absltest
if __name__ == "__main__":
    absltest.main()
