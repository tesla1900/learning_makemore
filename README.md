# Learning makemore 
In this series of projects following the lectures of [Andrej Kaparthy](http://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ), I work towards building a Generative Pre-training Transformer (GPT) starting with the [makemore](https://github.com/karpathy/makemore) project. makemore takes one text file as input, where each line is assumed to be one training thing, and generates more things like it. In the first part of the project, we will feed it 32,000 names in then `names.txt` file and produce more names like it. 

## Part 1 
We start by introducing bigram character level language model and train and sample from the model. We evaluate the quality of the model using negative log likelihood loss. We build the models using two different approaches and find the that result for both of them is the same. In the first approach, we count the frequency of all the bigrams and normalize it. In the second approach, we build a linear neural network using `PyTorch` which takes 27 inputs and has 27 output neurons. We do a forward pass to produce the output, backward pass (backpropagation), computing the gradients of the loss function with respect to the model’s parameters by working backward through the network. We use gradient descent algorithm to minimize the loss function. 

In the first part, the names produced by the model are 

```
cexze.
momasurailezityha.
konimittain.
llayn.
ka.
da.
staiyaubrtthrigotai.
moliellavo.
ke.
teda.
ka.
emimmsade.
enkaviyny.
fobspehinivenvtahlasu.
dsor.
br.
jol.
pyawaisan.
ja.
fdinee.
zka.
deru.
firit.
gaikajahahbevare.
kiysthelenaririenah.
```

## Part 2 

In the previous part, we were limited by bigram language model which was using counts and a basic neural network with a single linear layer. In the second part, we follow the technique developed in the [following paper](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), which uses this neural architecture: 

![neural](images/neural_part2.png)

We will use a more sophisticated multi-layer perceptron (MLP) character-level language model. Training involves optimizing embeddings, hidden layer weights and biases through backpropagation while minimizing negative log-likelihood loss. To ensure scalability and efficiency, we implemented mini-batch training, learning rate schedules and dataset splits (train, validation, and test). By experimenting with larger embedding dimensions, hidden layers and context lengths, we achieved a validation loss of 2.17, surpassing the bigram model’s best performance of 2.45. Visualization of learned embeddings revealed meaningful clustering of characters based on their behavior in sequences. The names produced by this model are: 

```
carmah.
americkifrimy.
xilita.
halayson.
rah.
bradherric.
kaeli.
ner.
kia.
chaihvik.
leggyn.
bron.
cateegan.
shoine.
livabi.
wazelo.
dearynix.
kaelynn.
deme.
deco.
```

## Part 3 

In this part, we build on the previous part by improving neural network performance and stability. [Kaiming Initialization](https://arxiv.org/abs/1502.01852) method was implemented, in which we scale weights based on the number of input connections to ensure balanced activations and gradient magnitudes across layers, making training more efficient. This is done in order to address the issue of vanishing or exploding gradients. For example, they find for ReLU, scaling the weights by $\sqrt{2 / n_l}$ gives the best results. The factor of 2 appears due to ReLU essentially removing the left half of the input. To further stabilize training, [batch normalization (BN)](https://arxiv.org/abs/1502.03167) was introduced. BN normalizes activations dynamically during training and maintains running statistics for inference, reducing sensitivity to initialization and improving the convergence of deeper networks.

Visualization methods were used to gain insights on activation, gradient, weight gradient and update to data ratio distributions, which can be seen below.

### Forward-Pass Activation Statistics
![activation](images/activation.png) 
- **Observed**:
  - Without proper initialization, activations in the hidden layers showed undesirable behavior:
    - Activations either shrank toward zero or saturated at extreme values (e.g., ±1 for `Tanh` layers).
    - Saturated activations were more prevalent in deeper layers, reducing gradient flow.
- **Ideal**:
  - Activations should display a roughly Gaussian distribution with:
    - A mean close to zero.
    - A consistent standard deviation across layers (e.g., ~0.65 for `Tanh`).
    - Minimal saturation (ideally &lt; 5% of neurons with values near activation limits like ±1).


### Backward-Pass Gradient Statistics
![gradient](images/gradient.png) 
- **Observed**:
  - Improper initialization led to:
    - **Vanishing gradients**: Gradients became too small in earlier layers, rendering them untrainable.
    - **Exploding gradients**: Gradients became excessively large, causing instability.
- **Ideal**:
  - Gradients should maintain a stable distribution across layers:
    - No layer should have disproportionately large or small gradient magnitudes.
    - Gradients should not diminish to zero or grow excessively as they propagate backward.

### Weight and Parameter Distributions
![weightsgrad](images/weightgradient.png) 
- **Observed**:
  - Certain layers, particularly the output layer, had overly large or small weight magnitudes.
  - Imbalances in weight scales led to uneven update rates during training.
  - Gradient-to-data ratio and update-to-data ratio plots highlighted these disparities, showing some layers trained faster or slower than others.
- **Ideal**:
  - Weight distributions should:
    - Have consistent standard deviations across layers, reflecting balanced initialization.
    - Exhibit stable updates relative to their magnitude, ensuring proportional training.

### Update-to-Data Ratios
![updata](images/updata.png) 
- **Observed**:
  - Improper learning rates or unbalanced initialization caused:
    - Updates that were too small, leading to undertraining.
    - Updates that were too large, causing instability.
- **Ideal**:
  - The update-to-data ratio for parameters should stabilize around `10^-3` (on a log scale):
    - Ensures meaningful updates without destabilizing parameter values.

The results at the end of part 3 are: 

```
carmahfari.
harlin.
mili.
tatyah.
cassie.
rah.
bradheryncia.
qui.
nellara.
chaiivia.
legyn.
ham.
joce.
quintis.
lilah.
jadilyn.
jero.
dearyn.
kai.
eveiluan.
```

## Part 4

Instead of doing something new, we focus on understanding the fundamentals of backpropagation in this part. We do this by manually backpropagating through each layer--cross entropy loss, 2nd linear layer, tanh, batchnorm, 1st linear layer and the embedding table without using PyTorch autograd's `loss.backward()`. We compare our result with for each output with PyTorch's results and find them to be in agreement. The result at the end of part 4 remains the same as part 3. 

## Part 5 

In the final part of the project, we transition from a simple multi-layer perceptron model with a single hidden layer to a more advanced hierarchical architecture inspired by the [WaveNet paper](https://arxiv.org/abs/1609.03499). Instead of directly squashing all the context information in one step by flattening sequence of input characters into a single dimension, which led to a loss of contextual information between characters, we processes longer sequences through progressive fusion. The architecture starts by embedding input characters, processes pairs of embeddings through deeper layers and fuses information hierarchically in a tree-like structure (as shown below). This avoids the early squashing of information and enhances the model’s ability to capture context. We also increase the context size from 3 to 8 for better performance. 

![fusion](images/fusion.png) 

We also fixed a bug in batch normalization which incorrectly shaped the tensors for 3 dimensional inputs to account for the fusion process. When we started, our loss in part 1 was 2.48 and at the end of part 5 is 1.98. While reducing the loss was not the main aim of the project, it was to understand how something like a GPT works, it is certainly interesting to see. 

The result at the end of this project are the following name-like sounding names: 

```
apoljoe.
otta.
heagon.
javye.
meela.
waiden.
airahro.
claris.
brennen.
blayta.
arah.
astacia.
annayveon.
zayely.
cassedin.
letmit.
dawdeng.
gian.
nataliya.
carmela.
```
