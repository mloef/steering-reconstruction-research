# LLM Steering Vector Research (Ongoing)

This repository contains research on analyzing and manipulating steering vectors in Large Language Models (LLMs), specifically focusing on the Meta-Llama-3.1-8B-Instruct model and the Goodfire Layer 19 16x Residual Stream SAE.

## Overview

Across five notebooks, I first set out to improve CAA steering vectors, then dig into the nature and implications of poor SAE reconstruction (but good interpretability!) of steering vectors. I include an extra notebook - the base_notebook - that all the others are based on.

In the first notebook, I develop a method for creating CAA steering vectors more quickly, with less data, and with similar or superior quality. In the second notebook, I compare mean and PCA aggregation methods for creating CAA steering vectors, and surprisingly find that they extract nearly identical steering vectors.

In the third notebook, I investigate the poor SAE reconstruction of CAA steering vectors. I hypothesize that it is due to the SAE's inability to represent the steering vector. I test this hypothesis by creating a CAA steering vector from an SAE feature, and I reject it. The fourth notebook tracks differences in steering vector latents across multiple SAE decodings, search for a pattern to explain the poor reconstruction, and doesn't find anything interesting; I include it as an example of negative results.

In the fifth notebook, I hypothesize that the SAE is confused by the steering vector, as it has only been trained on the residual stream. I test this hypothesis by extracting latents from steered activations instead of the vector itself. The results are far superior, supporting my hypothesis. I notice a pattern in the direction of the error from the steering vector SAE reconstruction, and I hypothesize that the steering vector is contextually activated by residual stream directions corresponding to different types of text, such as assistant output, and that the SAE reconstruction is interrupting this by trying to reconstruct a full residual stream. I investigate this by tracking the directions of the activations across a sequence, as compared to a steering vector, its SAE reconstruction, and the reconstruction error. I find that the steering vector tracks the previous pattern, but so does the combination of the reconstruction and error directions. This indicates that the reconstruction is capturing both the steering vector and an unknown additional component, represented by the error. This additional component is broadly similar to activations across the entire sequence. I hypothesize that this additional component is the SAE's malformed attempt to replicate a residual stream when decoding the steering vector, and that this attempt is the source of the high reconstruction error.

## Notebooks

### Extracting CAA Steering Vectors using All Sequence Positions (`notebooks/all_position_caa.ipynb`)
Typically, CAA steering vectors are created by extracting and diffing the activations at the last token position for two contrasting prompts. With just one contrasting pair, the activations are noisy, containing many irrelevant features pertaining to the specific context of the two prompts. Therefore, we use many pairs with the same contrasting concepts, diff each pair, and average the diffs - the noise mostly cancels out.

However, I developed a different method: position the contrasting concept in the system prompt, pad to length match the two prompts, diff the activations at each token position, and then average them. I found this method to provide steering vectors of similar or superior quality to last-position CAA. It is also much faster and more data efficient, as we only need a single contrasting pair, and we only have as many comparisons as there are tokens in the longer of the two prompts.

Intuitively, this method was designed to overcome the noise introduced by having non-matching contexts in the contrasting pair. By placing the contrast in the system prompt and padding it to length, every token in the sequence matches except for the intentionally contrasting ones. Therefore, the diff is entirely due to the contrasting concept, which appears in the activations at the system prompt and every token position after.

Initial results indicate that this method provides similar or superior quality steering vectors to last-position CAA, while being much faster and more data efficient.

### CAA Mean vs PCA (`notebooks/caa_mean_vs_pca.ipynb`)
CAA steering vectors are typically created by diffing many pairs of activations, then aggregating the diffs. Two common aggregation methods are mean and PCA. Here, I compare the two methods.

Surprisingly, both methods provide nearly identical steering vectors! The PCA method extracts only a single component that accounts for 32% of the variance. It is very surprising that this single component is so similar to the mean; it suggests both are interacting with some underlying latent structure. I'd like to see further investigation here.

### CAA via SAE Steering (`notebooks/caa_via_sae_steering.ipynb`)
In the above experiments, I noticed that passing a steering vector through the SAE provides interpretable latents, but reconstruction completely breaks the vector. Accordingly, the norm of the reconstruction error is extremely large, much larger than the norm of the original steering vector.

I hypothesized that the SAE was struggling to fully represent the concepts in the steering vector. To test this, I set out to create a CAA steering vector from an SAE feature. This guarantees that the concept is well-represented by the SAE. If it can reconstruct the vector, this supports the hypothesis that the problem with prior vectors was the subject matter. If it cannot, then the issue is something else.

Rather than using pairs of activations generated from contrasting prompts, I started with activations from a neutral prompt. I encoded these activations with the SAE, upweighted a single latent, then decoded them and added the original activation residuals back in. I then diffed this against the unmodified neutral activations to get a set of steering directions, then averaged these to get a steering vector for the SAE feature.

When I passed this steering vector through the SAE, it showed extremely strong activation of the targeted SAE feature; much stronger than the previous CAA vectors targeting the same feature! However, the reconstruction error was still extremely high. Therefore, it seems that the issue with SAE reconstruction of steering vectors is not due to the SAE being unable to represent the concepts, but something else.

### Failed Double Decode CAA (`notebooks/failed_double_decode_CAA.ipynb`)
I had many failed experiments and explorations, but this one was the most convenient to package into a notebook. Here, I search for clues to the steering vector reconstruction issue by reconstructing a reconstructed steering vector, then investigating the change in latents across the reconstruction process. As expected, it gets worse, but no clues emerge from the changed latents. At the end of the next notebook, I also use an SAE to encode the reconstruction error itself, then examine the latents, but find nothing interesting.

### CAA Steering Latents (`notebooks/caa_steering_latents.ipynb`)
This notebook is the most interesting result to me. I try to understand the SAE decomposition of the CAA steering vectors, given the apparent contradiction between the interpretable latents and the high reconstruction error. I hypothesize that the SAE is confused by the steering vector, as it has only been trained on the residual stream.

I test this hypothesis by finding a way to extract latents from the residual stream that represent the steering vector, then comparing them to the original CAA steering vector latents. If they show improvement and the reconstruction error norm is lower, then the problem is with the form factor of the steering vector, not its contents. For each token position in a normal assistant/user interaction sequence, I encode steered and unsteered activations with the SAE. Then, I diff the two sets of latents. I also reconstruct both sets of activations, and compare the reconstruction error norm and direction over the sequence for steered and unsteered activations.

I find that the steered and unsteered activations have broadly similar error norm and direction. Steered error norm is, surprisingly, slightly smaller, ranging over the sequence from ~0.7 to ~0.9 relative to the unsteered error norm. However, the steered and unsteered error directions suddenly diverge at the token position in the sequence where the assistant starts speaking. Previously, cos sim between the original and steered error directions ranges from ~0.7 to ~0.9, but it suddenly drops to ~0.25 and stays at ~0.25 to ~0.55 until the end of the sequence. This is very interesting, and I'm not sure what to make of it.

I investigate the extracted latents. I find that they are substantially similar to the latents obtained by passing the original CAA steering vector through the SAE. In fact, they are a superior representation of the steering vector, as the top-activating extracted latents better represent the core steering concept, and the distribution of latent activation values is more clustered, with a clearer peak of interpretable, highly activating latents.

Clearly, the SAE is capable of well-representing the content of the steering vector; these are excellent latents, and steering  decreased the error norm of SAE activation reconstruction. Therefore, I find another way to test my hypothesis of SAE reconstruction confusion.

Based on the divergence in the direction of the steered and unsteered error across the previous sequence, I hypothesize that the steering vector is contextually activated by residual stream directions corresponding to different types of text, such as assistant output. This would explain why the steered and unsteered error directions diverged after the assistant began speaking. I hypothesize that the SAE's confused reconstruction may not be contextually activating properly.

I run another sequence through the model, extracting activations at each token position. This sequence contains a normal user question, and a steered assistant response. I compare the cos sim of the activations and three different vectors: the original CAA steering vector, the reconstructed steering vector, and the reconstruction error. I found that the original steering vector had ~0 cos sim until the steered assistant response started, then it rapidly rose, stabilizing in a ~0.15 to ~0.6 range. The cos sim of the reconstructed steering vector and the error were consistently in a middle range of ~0.2 to ~0.8, with the error direction being negative. Their values were consistently nearly identical, yet had no clear relationship to the content of the sequence. However, I noticed that the cos sim of the reconstruction was increasingly different than that of the error over the sequence. When I diffed them, I found that the difference in cos sim between the reconstruction and error tracked the cos sim of the steering vector across the sequence!

To restate, the difference in activation cos sims between the reconstruction of the steering vector and the reconstruction error closely tracks the cos sim of that steering vector across the sequence. It implies that the reconstruction /is/ capturing the steering direction as well as the original steering vector, but there is an additional component that has been added to it, as represented by the reconstruction error. This component and its negative are similar to normal activations. This is very interesting! It supports the hypothesis that the SAE is reconstructing the steering vector to resemble a normal residual stream, and additionally suggests that the reconstruction is otherwise accurate to the content of the original steering vector.

## Running the Notebooks

### Environment Variables

Create a `.env` file with:
```
HF_TOKEN=your_huggingface_token
GOODFIRE_API_KEY=your_goodfire_api_key
```

### Compute Requirements

24GB of VRAM works great. I used a 4090 for my research, as it's quite cheap and speedy.

### Credits

Thank you to Goodfire for open sourcing their SAE, and for their free latent labels API!
