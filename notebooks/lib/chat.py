import torch
from nnsight.intervention import InterventionProxy
from functools import partial

from .sae import create_sae_steering_vector, latents_to_feature_map


def activation_space_caa_intervention(activations: InterventionProxy, steering_vector=None, coeff=None):
    return activations + (coeff * steering_vector)

def feature_space_caa_intervention(activations: InterventionProxy, sae=None, steering_vector=None, coeff=None):
    encoded_caa = sae.encode(steering_vector)
    encoded_acts = sae.encode(activations)

    reconstructed_acts = sae.decode(encoded_acts)
    error = activations - reconstructed_acts
    
    steered_acts = encoded_acts + (coeff * encoded_caa)
    return sae.decode(steered_acts) + error

def feature_space_sae_intervention(activations: InterventionProxy, sae=None, coeff=None, sae_features=None, k = 0):
    features = sae.encode(activations).detach()
    reconstructed_acts = sae.decode(features).detach()
    error = activations - reconstructed_acts

    if k:
        sae_features = sorted(sae_features, key=lambda k,v:v, reverse=True)

    summed_values = sum(sae_features.values())
    
    for index, value in sae_features.items():
        features[:, :, [index]] += coeff * value / summed_values

    # Very important to add the error term back in!
    return sae.decode(features) + error

def activation_space_sae_intervention(activations: InterventionProxy, sae=None, coeff=None, sae_features=None, k = 0):
    if k:
        sae_features = sorted(sae_features, key=lambda k,v:v, reverse=True)
    
    steering_vector = create_sae_steering_vector(sae, sae_features)
    
    steering_vector *= coeff

    return activations + steering_vector

def generate_tokens(model, intervention, max_length, layer, input_tokens=None):
  if input_tokens is None:
    input_tokens = model.tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Hello, how are you?"},
        ],
        add_generation_prompt=True,
        return_tensors="pt",
    )

  for _ in range(max_length):
    logits, _, _ = model.forward(
        input_tokens,
        interventions={
          layer: intervention
        },
    )

    new_token = logits[-1].argmax(-1)
    input_tokens = torch.cat([input_tokens[0], new_token.unsqueeze(0).cpu()]).unsqueeze(0)
    if new_token == 128009:
      print("\n<EOT reached>")
      break

    decoded_new_token = model.tokenizer.decode(new_token)

    print(decoded_new_token, end="")

def test_all_interventions(model, aggregate_caa_vector, caa_features,sae, coeff, max_length, layer):
    original_input_tokens = model.tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Hello, how are you?"},
        ],
        add_generation_prompt=True,
        return_tensors="pt",
    )

    print('Activation space CAA intervention')

    intervention = partial(
        activation_space_caa_intervention, 
        steering_vector=aggregate_caa_vector, 
        coeff=coeff
    )

    generate_tokens(model, original_input_tokens, intervention, max_length, layer)

    print('\nFeature space CAA intervention')

    intervention = partial(
        feature_space_caa_intervention,
        sae=sae,
        steering_vector=aggregate_caa_vector,
        coeff=coeff
    )

    generate_tokens(model, original_input_tokens, intervention, max_length, layer)

    print('\nFeature space SAE intervention')

    extracted_pirate_features = latents_to_feature_map(caa_features[0])
    intervention = partial(
        feature_space_sae_intervention,
        sae=sae,
        sae_features=extracted_pirate_features,
        coeff=coeff
    )

    generate_tokens(model, original_input_tokens, intervention, max_length, layer)

    print('\nActivation space SAE intervention')

    intervention = partial(
        activation_space_sae_intervention,
        sae=sae,
        sae_features=extracted_pirate_features,
        coeff=coeff
    )

    generate_tokens(model, original_input_tokens, intervention, max_length, layer)