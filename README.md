# Gray Swan Technical Take Home Interview

Good work getting through the Light Screen interview – we know it was a difficult set of problems to get through, and we appreciate you taking the time to do so. 

Due to the difficulty of sitting in a virtual testing environment for only 45 minutes, we wanted to give you another chance to demonstrate some of the skills you gave us a sneak peek of during your last interview, this time a bit more tailored to things we'd do on a day-to-day basis.

## Rules

This is a completely **asynchronous** evaluation of your skillset – something that you can complete in the comfort of your own IDE and normal working environment. You may use **any** internet resources and LLM coding assistance that you would normally use. We do ask that if you explicitly copy and paste code from an LLM into your code, you place it in between comments:

```
## LLM-GENERATED CODE START ##

...

## LLM-GENERATED CODE END ##
```

Similarly, if you copy and paste code from the internet (or a paper), we ask that you provide proper attribution. This will not be used against you in any form, it will simply serve as a marker for things that we might want to ask you questions about to assess your understanding.

We **encourage** you to look at outside sources if you are feeling stuck, and we encourage discussing the project at hand and implementation strategies with LLMs. They can frequently help guide us down new directions we weren't considering before. For the remainder of this README, we will include links where appropriate to materials that we think you may find helpful. 

### TIMELINE

This README will have been included in an email that asks you to schedule a time to interview with one of our Engineers. This stage is a 2-step process: **prior** to your video-interview with our Engineers, you should complete the project provided and **reply to the email with a copy of your code within 24hr of your scheduled time** such that we can look through it in time before talking with you live. This gives us a chance to find specific points we want to discuss, and save time so that you will be able to tell us a little more about yourself. We anticipate this project should take no more than 3 hours end-to-end, given that you will have had some idea of what to do from your LightScreen evaluation. 

## Project Overview

Since we are an AI safety and security company, and have nothing against birds, you will be demonstrating some of your accumen for AI safety in this project. You may choose any aspect of safety that you feel is most interesting and/or personally important. Your overall goal should be to better-align an instruction-tuned language model of your choosing, along your desired axis, and then evaluating how you did.

The project is divided into four (really three) parts:

0. **Finding a safety-themed preference dataset**: There are plenty of great resources (read: datasets) for preference alignment on HuggingFace. You should pick one that has (or can be formatted into) samples with `prompt`, `chosen`, and `rejected` to be used with DPO. 
1. **Building a synthetic dataset**: You will use vLLM to create a synthetic evaluation dataset that will be tested against both your DPO model and the original model, to see how much your chosen dataset has improved alignment on your target axis.
3. **DPO**: You'll implement DPO (Direct Preference Optimization) with your chosen HuggingFace dataset on a model of your choosing.
4. **Interpretability and Evaluation**: You'll implement token attribution, and evaluate both models on your synthetic evaluation set.

### Codebase Structure

There is **no** starter code provided for this assignment. In the four sections below we will outline rough requirements, specifications, and suggestions for you to follow when completing these tasks. However, the vast majority of the implementation details will be left up to you. We are interested in seeing what tools/techniques you favor, and how you think through problems, so this is designed to be as free and open-ended as possible, so as to not constrain your abilities, creativity, or technical know-how. 


## Part 0: Choosing a preference datset 

For many people, "safety" means a lot of different things. You might think that x-risk is the most important axis along which we should focus our efforts, while others may view ethical alignment as being more important. We encourage you to think about what *you* find important, and use that to guide your selection of a preference dataset. As mentioned above, preference datasets usually have the following format (for singleturn):

`prompt`: The user prompt to the AI assistant
`chosen`: The preferred assistant response
`rejected`: A response that you want the assistant to avoid outputting 

For multiturn, it might look a little different. However, the core idea behind multiturn preference datasets is that **all messages in the conversation before the last assistant message are the same**. 

If you're curious what an example of this dataset looks like, [here's](https://huggingface.co/datasets/Anthropic/hh-rlhf/viewer/default/train?row=0&views%5B%5D=train) a popular one. 

### Deliverables

A link to your chosen dataset.

## Part 1: Building a synthetic eval dataset

A large part of safety work is evaluations – it's important to be able to accurately assess the safety of a model you are red-teaming or training yourself. To that end, for this part you will generate a synthetic evaluation dataset. Here's a rough outline of what we expect:

- Around 1000-2000 benign examples, 1000-2000 harmful samples, where 'benign' and 'harmful' are relative to your particular safety topic / dataset. 
- You'll probably want to use [vLLM](https://github.com/vllm-project/vllm) for this for fast and efficient generation. You may pick any model you would like to use to generate this dataset.
- Larger models and/or reasoning models are likely to be better at this task than smaller, non-reasoners. If you cannot find a model that you like that you can use via vLLM, we suggest using the free tier of [OpenRouter](https://openrouter.ai/models), which can be used like the OpenAI api. 

The rest of the design choices are up to you! We want to see how you think about evaluating models, and as such, feel that we can get the best sense by letting you interpret the rest of this part as you choose. As a friendly hint, we suggest the benign samples be questions/prompts that the model *should not* refuse to discuss, and the harmful samples be questions/prompts that the model *should* refuse to discuss. 

### Deliverables

- Your evaluation dataset as a `.json` file, or a link to a dataset that has been pushed to the HuggingFace Hub. 
- All code used to create the dataset

## Part 2: Post-Training 

For this part, you will be again using DPO to post-train an instruction-tuned model of your choosing. If you are particularly compute-constrained, we suggest [this](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) 135M parameter instruct model, however feel free to use any model size that you can work with. An overview of DPO is below:

### DPO Loss Function

If you need a refresher, here's the paper: https://arxiv.org/pdf/2305.18290

The DPO loss function is defined as:

$$L_{\text{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{p_\theta(y_w|x)}{p_{\text{ref}}(y_w|x)} - \beta \log \frac{p_\theta(y_l|x)}{p_{\text{ref}}(y_l|x)} \right) \right]$$

Where:

-   $\theta$ represents the model parameters
-   $x$ is the prompt
-   $y_w$ is the preferred (good) completion
-   $y_l$ is the dispreferred (bad) completion
-   $\beta$ is a hyperparameter controlling the strength of the preference
-   $p_\theta$ is the model's probability distribution
-   $p_{\text{ref}}$ is the reference model's probability distribution

To implement DPO posttraining, we will give you the following two options:

1. Implement a basic pytorch training loop that implements DPO loss at a low level. 
2. Implement a training setup using HuggingFace's DPOTrainer. 

You may choose whichever is more suitable. Keep in mind that if you choose to work with a high-level abstraction of DPO and pytorch training by using HuggingFace's Trainer, you will be asked more technical questions about your understanding of DPO and other topics during the interview. 

### Deliverables

- A log file that contains logs from your post-training run (any text format acceptable). At the very least this should contain loss metrics, along with any other metrics you would find useful to see during training.
- All code used to implement this part.

Note that we do not require a final model path, but we will assume that your code is plug-and-play and can be ran by us if need be.


## Part 3: Interpretability and Evaluation

There are two steps for this final part: step one is implementing token attribution, and step two is comparing your post-trained model to the original on your evaluation dataset. 

As a refresher, here's what we asked you to do during the LightScreen:

```
import torch
import multiprocessing
from colorama import Fore, Style
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from params import TEST_PROMPTS
from utils import (
    device,
    load_models,
    print_model_comparison,
)


# ------------------------------------------------------------------------------------------------
# Token attribution. You need to implement this
# ------------------------------------------------------------------------------------------------


def token_attribution(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    completion: str,
) -> list[tuple[str, float]]:
    # Implement this function. It should take a model, tokenizer, prompt, and completion, and return a list
    # of tuples representing some token attributions for the prompt tokens.
    #
    # The list should be in the same order as the prompt tokens.
    # The scores can be any finite float value where higher numbers indicate higher importance of the
    # token, ideally with linear scale.
    #
    # Example:
    #   - Input prompt: "The capital of France"
    #   - Input completion: "is Paris"
    #   - Output: [("The", 0.1), ("capital", 0.9), ("of", 0.05), ("France", 1.8)]
    return []


# ------------------------------------------------------------------------------------------------
# Helpers to visualize attribution scores. You shouldn't need to change anything below here.
# ------------------------------------------------------------------------------------------------


# Uses the attribution scores to colorize high-scoring tokens for visualization
def visualize_attribution(
    tokens_with_attributions: list[tuple[str, float]],
    max_attr: float,
) -> str:
    colored_text = ""
    for token, attribution in tokens_with_attributions:
        # Skip special tokens
        if token.startswith("<") and token.endswith(">"):
            continue

        # Normalize attribution value (0-1 scale)
        norm_attribution = min(1.0, attribution / max_attr)
        if norm_attribution > 0.7:
            color = Fore.RED  # High importance
        elif norm_attribution > 0.4:
            color = Fore.YELLOW  # Medium importance
        else:
            color = Fore.GREEN  # Low importance

        # Add space before token if needed
        if token.startswith("Ġ"):
            token = " " + token[1:]
        colored_text += f"{color}{token}{Style.RESET_ALL}"
    return colored_text


# Wrapper that adds attribution visualization to a model's completion
def create_attribution_display(model, tokenizer):
    def attribution_modifier(prompt, completion):
        # Colorize prompt tokens based on attribution scores
        completion_text = completion[len(prompt) :]
        attribution = token_attribution(model, tokenizer, prompt, completion_text)
        max_attr = max(attr for _, attr in attribution)
        colored_text = visualize_attribution(attribution, max_attr)
        return colored_text + completion_text

    return attribution_modifier


if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()
    torch.set_num_threads(num_cores)
    torch.set_num_interop_threads(num_cores)
    model, base_model, tokenizer = load_models()
    print(
        f"{Fore.RED}Red = high impact{Style.RESET_ALL}, {Fore.YELLOW}Yellow = medium impact{Style.RESET_ALL}, {Fore.GREEN}Green = low impact{Style.RESET_ALL}"
    )
    base_attribution_modifier = create_attribution_display(base_model, tokenizer)
    tuned_attribution_modifier = create_attribution_display(model, tokenizer)
    for prompt in TEST_PROMPTS:
        print_model_comparison(
            model,
            base_model,
            tokenizer,
            prompt,
            base_modifier=base_attribution_modifier,
            tuned_modifier=tuned_attribution_modifier,
        )
```

You may take inspiration from this -- we are looking for something similar. In particular, you should:
- Implement the token attribution function
- Obtain token attributions for 5 samples from your eval set for both the original and post-trained model 
- You can implement your own visualization, although we like the color setup. At minimum, we want to see each token with it's attribution score. 

Finally, run your synthetic evaluation on your model. This likely requires:
- Generating responses for the original model
- Generating responses for your post trained model
- Judging the responses

### Deliverables
- All code used to complete this section.
- A visualization of your token attribution function in action on 5 samples from your eval set.
- A 10-sample subset of the original model and post trained model responses on the evaluation dataset.
- A brief discussion of what you found during evaluation. 

## Wrapping up

That's it! This project goes both ways – it gives us a better understanding of how you think and what you know, and it should also hopefully give you a better idea of what it is you might be doing at Gray Swan. 

**Please remember to submit all deliverables in an email to the hiring team at least 24hr before your live video interview** 