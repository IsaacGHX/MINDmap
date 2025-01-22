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
- [VQ-VAE](https://arxiv.org/abs/1711.00937) (*Neural Discrete Representation Learning*, NIPS2017) | [\[CODE\]](https://github.com/MishaLaskin/vqvae)
- [VQ-VAE2](https://arxiv.org/abs/1906.00446) (*Generating Diverse High-Fidelity Images with VQ-VAE-2*, ArXiv-version) | [\[CODE\]](https://github.com/rosinality/vq-vae-2-pytorch)
- [VQ-GAN](https://arxiv.org/abs/2012.09841) (*Taming Transformers for High-Resolution Image Synthesis*, ArXiv-version) | [\[CODE\]](https://github.com/Westlake-AI/VQGAN)
- [RQ-VAE](https://arxiv.org/abs/2203.01941) (*Autoregressive Image Generation using Residual Quantization*, CVPR2022) | [\[CODE\]](https://github.com/kakaobrain/rq-vae-transformer)
- [MoVQ-VAE](https://arxiv.org/abs/2209.09002)(*MoVQ: Modulating Quantized Vectors for High-Fidelity Image Generation*, NIPS2022) | [\[CODE\]](https://github.com/ai-forever/MoVQGAN)

### Overall Generation Paradigm
- [LlamaGen](https://arxiv.org/abs/2406.06525)(*Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation*, ArXiv-version) | [\[CODE\]](https://github.com/FoundationVision/LlamaGen/tree/main)
- [VAR](https://arxiv.org/abs/2404.02905)(*Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction*, NIPS 2024 Best) | [\[CODE\]](https://github.com/FoundationVision/VAR/tree/main)
- [MAR](https://arxiv.org/abs/2406.11838)(*Autoregressive Image Generation without Vector Quantization*, NIPS2024) | [\[CODE\]](https://github.com/LTH14/mar)
- [RAR](https://arxiv.org/abs/2411.00776)(*Randomized Autoregressive Visual Generation*, ArXiv-version) | [\[CODE\]](https://yucornetto.github.io/projects/rar.html)
- [Next-patch](https://arxiv.org/abs/2412.15321)(*Next Patch Prediction for Autoregressive Visual Generation*, ArXiv-version) | [\[CODE\]](https://github.com/PKU-YuanGroup/Next-Patch-Prediction)

### Anything with GPT
- [Lumina-mGPT](https://arxiv.org/abs/2408.02657)(*Illuminate Flexible Photorealistic Text-to-Image Generation with Multimodal Generative Pretraining*) | [\[CODE\]](https://github.com/Alpha-VLLM/Lumina-mGPT)
- [Fluid](https://arxiv.org/abs/2410.13863)(*Fluid: Scaling Autoregressive Text-to-image Generative Models with Continuous Tokens*)
- [1D-Tokenizer](https://arxiv.org/abs/2406.07550)(*An Image is Worth 32 Tokens for Reconstruction and Generation*)

### Faster
- [MaskGiT](https://arxiv.org/abs/2202.04200)(*MaskGIT: Masked Generative Image Transformer*, ArXiv-version) | [\[CODE\]](https://github.com/google-research/maskgit)
- [Bitwise-Infinity](https://arxiv.org/abs/2412.04431)(*Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis*, ArXiv-version) | [\[CODE\]](https://github.com/FoundationVision/Infinity)
- [ZipAR](https://arxiv.org/abs/2412.04062)(*ZipAR: Accelerating Auto-regressive Image Generation through Spatial Locality*, ArXiv-version) | [\[CODE\]](https://github.com/ThisisBillhe/ZipAR)

### Attention
- [Attention Sink 0](https://arxiv.org/abs/2309.17453)(*Efficient Streaming Language Models with Attention Sinks*, ICLR 2024) | [\[CODE\]](https://github.com/sail-sg/Attention-Sink)
- [Attention Sink 1](https://arxiv.org/abs/2410.10781)(*When Attention Sink Emerges in Language Models: An Empirical View*, ArXiv-version) | [\[CODE\]](https://github.com/sail-sg/Attention-Sink)
- [Attention Register](https://arxiv.org/abs/2309.16588)(*Vision Transformers Need Registers*, ICLR 2024) | [\[CODE\]](https://github.com/kyegomez/Vit-RGTS)



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

## VI. Free to Contact =w=/
If you have any questions or want to further understand the relevant situation of the project, you can contact us in the following ways :

- **Email**: isaacpfino [at] gmail [dot] com
