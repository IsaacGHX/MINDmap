# MINDmap
*<span style="color:lightblue;">Faster</span>
<span style="color:lightpink;">Auto-Regression</span>
<span style="color:lightgreen;">Image</span>
<span style="color:lightcoral;">Generation</span>*

## I. Project Overview
This project is dedicated to exploring the application of autoregressive (AR) models in image generation, aiming to achieve several key objectives:

- **<span style="color:lightblue;">Better Results</font>**: By improving and optimizing the relevant mechanisms of autoregressive models, we strive to enhance the quality of generated images, making them more realistic and in line with expectations.
- **<span style="color:lightblue;">More Controllable Output</span>**: We are exploring ways to precisely control various attributes of image generation, such as style and content, enabling the generated results to be more customizable.
- **<span style="color:lightblue;">Faster Inference</span>**: Researching methods that can accelerate the inference process of autoregressive models, reducing the time consumed in generating images and improving efficiency.
- **<span style="color:lightblue;">In-depth Exploration of the AR Paradigm in Image Generation</span>**: Analyzing the internal working mechanisms of autoregressive models when applied to image generation from a theoretical perspective, providing a solid foundation for subsequent improvements.
- **<span style="color:lightblue;">Empirical Verification</span>**: Through a large number of experiments and actual data, we aim to test our proposed ideas and improvement measures to ensure their effectiveness and reliability.

## II. Related Research Readings
During the progress of the project, we have read several relevant articles. Here is a partial list of the reference articles:

### Image Encoders & Decoders, based on VQ-VAE
- [VQ-VAE](https://arxiv.org/pdf/1711.00937) (*Neural Discrete Representation Learning*, Arxiv-version) | [\[CODE\]](https://github.com/MishaLaskin/vqvae)
- [VQ-VAE2](https://arxiv.org/pdf/1906.00446) (*Generating Diverse High-Fidelity Images with VQ-VAE-2*, Arxiv-version) | [\[CODE\]](https://github.com/rosinality/vq-vae-2-pytorch)
- [VQ-GAN](https://arxiv.org/pdf/2012.09841) (*Taming Transformers for High-Resolution Image Synthesis*, Arxiv-version) | [\[CODE\]](https://github.com/Westlake-AI/VQGAN)
- [RQ-VAE](https://arxiv.org/pdf/2203.01941) (*Autoregressive Image Generation using Residual Quantization*, Arxiv-version) | [\[CODE\]](https://github.com/kakaobrain/rq-vae-transformer)
- [MoVQ-VAE](https://arxiv.org/pdf/2209.09002)(*MoVQ: Modulating Quantized Vectors for High-Fidelity Image Generation*, Arxiv-version) | [\[CODE\]](https://github.com/ai-forever/MoVQGAN)

### Overall Generation Paradigm
- [LlamaGen](https://arxiv.org/pdf/2406.06525)(*Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation*, Arxiv-version) | [\[CODE\]](https://github.com/FoundationVision/LlamaGen/tree/main)
- [VAR](https://arxiv.org/pdf/2404.02905)(*Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction*, Arxiv-version) | [\[CODE\]](https://github.com/FoundationVision/VAR/tree/main)
- [MAR](https://arxiv.org/pdf/2406.11838)(*Autoregressive Image Generation without Vector Quantization*, Arxiv-version) | [\[CODE\]](https://github.com/LTH14/mar)


### Faster
- [MaskGiT](https://arxiv.org/pdf/2202.04200)(*MaskGIT: Masked Generative Image Transformer*, Arxiv-version) | [\[CODE\]](https://github.com/google-research/maskgit)


## III. Current Work Progress
Currently, we are carrying out the following key tasks:

- **Enhancing the Information Content of Discretized Latent in VQ-VAE from the Perspective of Better idx Encoding (Partly Done. )**: We are exploring ways to optimize the indexing (idx) encoding to increase the information content of the discretized latent representation in the Vector Quantized-Variational AutoEncoder (VQ-VAE), thereby assisting autoregressive models to generate high-quality images better.
- **Verifying the Proposed New Methods (Doing...)**: For the new methods we have conceived to improve the performance of autoregressive models in image generation, we are verifying their practical effects in enhancing image generation quality, controllability, and inference speed through rigorous experimental design and comparative analysis.

## IV. Future Prospects
We expect that through the continuous progress of this project, valuable results can be achieved in the field of autoregressive model image generation, providing useful references and inspiration for subsequent related research and practical applications. Meanwhile, we also plan to open-source and share our achievements, so as to jointly promote the development of this field with more researchers and developers.

## V. How to Participate (Optional part, if you want others to participate in the project)
If you are interested in this project, you are welcome to participate in the following ways:

- **Submitting Issues**: If you find problems in the project or have suggestions for improvement, you can submit relevant content at [the specific Issue address (such as the Issue page corresponding to the GitHub repository)].
- **Contributing Code**: We warmly welcome capable developers to contribute code. The specific code contribution process and specifications can be viewed at [the address of the code contribution guide document (if available)].

## VI. Contact Us =w=/
If you have any questions or want to further understand the relevant situation of the project, you can contact us in the following ways :

- **Email**: isaacpfino [at] gmail [dot] com
