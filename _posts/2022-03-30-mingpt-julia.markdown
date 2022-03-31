---
layout: post
title:  "minGPT in Julia using Flux!"
date:   2022-03-30 11:58:45 +0300
categories: julia flux machine-learning
---

As a learning exercise I tried to [port](https://github.com/cancandan/mingpt-julia/blob/main/mingpt.jl) Anrey Karpathy's [minGPT](https://github.com/karpathy/minGPT), which is based on Python and PyTorch to Julia and Flux. He exercises the code on three distinct domains; language, vision and math. Here I concentrate on the [math problem](https://github.com/karpathy/minGPT/blob/master/play_math.ipynb), in which we are interested in seeing whether the model can learn to do addition given two numbers. So we create a dataset where the input to the model is a pair of two digit numbers along with the two digits of the output. For example, the addition of 85 and 50 which results in 135 is encoded as the sequence [8, 5, 5, 0, 1, 3]. The model should predict the next digit, ie. 5, and this is encoded as [-100, -100, -100, 1, 3, 5]. The -100 is used to mask the loss to 0, because we want to train on the output locations only. A dataset created as such is split to train and test datasets and the train dataset is used to train the model.

It was very straightforward to port all of the components. For example below on the left you see the Python class definition for the `CausalSelfAttention` component, and on the right is the struct definition for Julia.

<style>
  /* .title_box {
  border: #3c5a86 1px solid;   
   border-radius: 1px;
   border-bottom: none;
   border-left: none;
   border-right: none;
} */

#title {
  position: relative;
  top: -0.5em;
  margin-left: 1em;
  display: inline;
  background-color: white;  
}

pre, code {
  border: #3c5a86 1px solid;   
  font-size: 11px;
}

img {
  border: 1px solid;
}

</style>

<div style="width: 100%;">
  <div class="title_box" style="width: 49%; float: left">
    <div id="title">Python</div>        
      <pre data-start="44" data-end="59" data-lang="python"
      data-src="https://raw.githubusercontent.com/karpathy/minGPT/master/mingpt/model.py"
      data-view="https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L44-L59"></pre>          
  </div>
  <div class="title_box" style="margin-left: 51%;"> 
    <div id="title">Julia</div>      
      <pre data-start="14" data-end="33" data-lang="julia"
      data-src="https://raw.githubusercontent.com/cancandan/mingpt-julia/main/mingpt.jl"
      data-view="https://github.com/cancandan/mingpt-julia/blob/main/mingpt.jl#L14-L33"></pre>      
    </div>
</div>


<div style="clear: both;">
</div>

The meat of this component follows next. One thing that tripped me here is the application of the mask. As you can see, on the left below, the `att` variable is modified in-place by using the `masked_fill` function of PyTorch. Doing the same thing with Flux lead to an error saying `Mutating arrays is not supported`. I guess this is not possible in the current AD component of Flux, ie. Zygote. To work around that I added the upper triangular mask to the output `att` of the batch matrix multiplication operation, which I do using Flux functions `batched_mul` and `batched_transpose`. Note that here Flux requires the batch dimesion to be the last, as evidenced by the difference in the order of `B, T, C`. 


<div style="width: 100%;">
  <div class="title_box" style="width: 49%; float: left">
    <div id="title">Python</div>        
      <pre data-start="61" data-end="79" data-lang="python"
      data-src="https://raw.githubusercontent.com/karpathy/minGPT/master/mingpt/model.py"
      data-view="https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L44-L59"></pre>          
  </div>
  <div class="title_box" style="margin-left: 51%;"> 
    <div id="title">Julia</div>      
      <pre data-start="39" data-end="76" data-lang="julia"
      data-src="https://raw.githubusercontent.com/cancandan/mingpt-julia/main/mingpt.jl"
      data-view="https://github.com/cancandan/mingpt-julia/blob/main/mingpt.jl#L39-L76"></pre>      
    </div>
</div>

<div style="clear: both;">
</div>

An interesting bit in Karpathy's code is how he had to select the parameters of the model to apply weight decay to. He carefully selects which parameters of the model will be decayed. This is shown in the lengthy function below:

<pre data-start="136" data-end="180" data-lang="python"
      data-src="https://raw.githubusercontent.com/karpathy/minGPT/master/mingpt/model.py"
      data-view="https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L136-L180"></pre>          


In Flux one can implement the `trainable` function for this, as described in the [docs](https://fluxml.ai/Flux.jl/stable/models/advanced/#Customising-Parameter-Collection-for-a-Model). Getting inspiration from that, I added a `decayed_trainable`. So how I specify the parameters looks like this:

<pre data-start="80" data-end="91" data-lang="julia"
      data-src="https://raw.githubusercontent.com/cancandan/mingpt-julia/main/mingpt.jl"
      data-view="https://github.com/cancandan/mingpt-julia/blob/main/mingpt.jl#L80-L91"></pre>          


Flux docs mention the weight decayed version of ADAM, the `ADAMW`. But as far as I understand, this is not quite what Karpathy and Pytorch's ADAMW works here, so I grabbed the code of basic ADAM and added the bag of tricks used in deep learning, like norm clipping the gradients and decoupled weight decay of selected parameters. To be precise I tried to implement the algorithm in the [paper](https://arxiv.org/pdf/1711.05101.pdf).

![ADAMW](/assets/static/adamw.png)

So our optimiser looks like this:

<pre data-start="255" data-end="295" data-lang="julia"
      data-src="https://raw.githubusercontent.com/cancandan/mingpt-julia/main/mingpt.jl"
      data-view="https://github.com/cancandan/mingpt-julia/blob/main/mingpt.jl#L255-L295"></pre>          


For training, we need a loss function and its gradient computed on batches of data. So we get the ouput from the model apply our cross entropy / softmax loss function via the `Zygote.pullback` and hit to the optimiser with its output via `Flux.Optimise.update!` as shown:

<pre data-start="297" data-end="301" data-lang="julia"
      data-src="https://raw.githubusercontent.com/cancandan/mingpt-julia/main/mingpt.jl"
      data-view="https://github.com/cancandan/mingpt-julia/blob/main/mingpt.jl#L297-L301"></pre>          

<pre data-start="336" data-end="340" data-lang="julia"
      data-src="https://raw.githubusercontent.com/cancandan/mingpt-julia/main/mingpt.jl"
      data-view="https://github.com/cancandan/mingpt-julia/blob/main/mingpt.jl#L336-L340"></pre>          

  
My model was training well at this point but it was about 10x slower than the Python version on the GPU. Having no idea what could possible make it run so slowly, I found about [Transformer.jl](https://github.com/chengchingwen/Transformers.jl), a Julia library for transformers. Here we see a custom implementation of the batched matrix multiplication AND how to efficiently differentiate it, like this:

<pre data-start="25" data-end="48" data-lang="julia"
      data-src="https://raw.githubusercontent.com/chengchingwen/Transformers.jl/master/src/fix/batchedmul.jl"
      data-view="https://github.com/chengchingwen/Transformers.jl/blob/master/src/fix/batchedmul.jl#L25-L48"></pre>

The `batched_gemm!` here is also hitting a CUDA version implemented in the Transformers.jl. And indeed bringing those in to my code, it started running fast. However, thanks to the wonderful people at [Julia Slack](https://julialang.org/slack/) (Michael Abbott, Andrew Dinhobl, Julian Samaroo), I learned all of this is already integrated to the Flux library. For example the efficient differentiation is now here in the form of a `rrule` of [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl):

<pre data-start="85" data-end="99" data-lang="julia"
      data-src="https://raw.githubusercontent.com/FluxML/NNlib.jl/d8b9b41c8977b18ab4adcc2f288ffcd9c4c43c3f/src/batched/batchedmul.jl"
      data-view="https://github.com/FluxML/NNlib.jl/blob/d8b9b41c8977b18ab4adcc2f288ffcd9c4c43c3f/src/batched/batchedmul.jl#L85#L85-L99"></pre>

It turns out that what made my code run extremely slowly was NOT casting the output of the `sqrt` below to `Float32`. The function `sqrt` outputs here a `Float64` and makes the whole chain afterwards very inefficient. So, number one thing to look out for when tracking down inefficiencies is making sure you are using the correct types.

<pre data-start="56" data-end="60" data-lang="julia"
      data-src="https://raw.githubusercontent.com/cancandan/mingpt-julia/main/mingpt.jl"
      data-view="https://github.com/cancandan/mingpt-julia/blob/main/mingpt.jl#L56-L60"></pre>          


One can run this as I did in [this notebook](https://github.com/cancandan/mingpt-julia/blob/main/run.ipynb) copied below:

```julia
include("minGPT.jl")

using Random
Random.seed!(123)

ndigit=2

(trnx,trny),(tstx,tsty)=makeData(ndigit)    

map(addOneForJulia, [trnx, trny, tstx, tsty])

config = Dict("vocab_size"=>10, "n_embed"=>128, "attn_pdrop"=>0.1f0, "resid_pdrop"=>0.1f0, "embd_pdrop"=>0.1f0, "block_size"=>6, "n_layer"=>2, "n_head"=>4,
"max_epochs"=>110, "batch_size"=>512, "learning_rate"=>6f-4, "lr_decay"=>true, "warmup_tokens"=>1024, "final_tokens"=>50*size(trnx)[2]*(ndigit+1), "betas"=>(0.9f0, 0.95f0));

model = mytraining(trnx, trny, tstx, tsty, config)
```

    Epoch: 1 Iter: 1 Train Loss: 2.95 lr_mult: 1.00 tokens: 1536
    Epoch: 1 Iter: 11 Train Loss: 2.07 lr_mult: 1.00 tokens: 16896
    Test Loss: 1.90209
    Epoch: 2 Iter: 1 Train Loss: 1.98 lr_mult: 1.00 tokens: 25536
    Epoch: 2 Iter: 11 Train Loss: 1.91 lr_mult: 1.00 tokens: 40896
    Test Loss: 1.7956433
    Epoch: 3 Iter: 1 Train Loss: 1.86 lr_mult: 1.00 tokens: 49536
    Epoch: 3 Iter: 11 Train Loss: 1.78 lr_mult: 0.99 tokens: 64896
    Test Loss: 1.7278897
    Epoch: 4 Iter: 1 Train Loss: 1.76 lr_mult: 0.99 tokens: 73536
    Epoch: 4 Iter: 11 Train Loss: 1.73 lr_mult: 0.99 tokens: 88896
    Test Loss: 1.6655266
    Epoch: 5 Iter: 1 Train Loss: 1.69 lr_mult: 0.98 tokens: 97536
    Epoch: 5 Iter: 11 Train Loss: 1.69 lr_mult: 0.98 tokens: 112896
    Test Loss: 1.6301216
    Epoch: 6 Iter: 1 Train Loss: 1.63 lr_mult: 0.98 tokens: 121536
    Epoch: 6 Iter: 11 Train Loss: 1.63 lr_mult: 0.97 tokens: 136896
    Test Loss: 1.5736698
    ...    
    Epoch: 109 Iter: 1 Train Loss: 0.01 lr_mult: 0.94 tokens: 2593536
    Epoch: 109 Iter: 11 Train Loss: 0.00 lr_mult: 0.93 tokens: 2608896
    Test Loss: 0.00010189927
    Epoch: 110 Iter: 1 Train Loss: 0.01 lr_mult: 0.92 tokens: 2617536
    Epoch: 110 Iter: 11 Train Loss: 0.01 lr_mult: 0.91 tokens: 2632896
    Test Loss: 0.0002310586





```julia
give_exam(model, trnx, trny, config)
```

    tot: 8000 tot_correct: 7999
    


```julia
give_exam(model, tstx, tsty, config)
```

    tot: 2000 tot_correct: 2000

