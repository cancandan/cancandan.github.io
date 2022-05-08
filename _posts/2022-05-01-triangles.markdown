---
layout: post
title:  "Some CUDA programming fun in Julia"
date:   2022-05-07 17:29:00 +0300
categories: julia graphics cuda 
tags:
    - julia
---


Suppose we want to draw a batch of images, where each image is made up of randomly positioned and colored triangles, that are blending. It will look like this:

![triangles](/assets/static/avnicompare.png){: .center-image}

and then find the euclidean distance of each such image to a given target image. 

Now why on earth am I doing this? Well, this turns into an interesting optimization problem of finding the closest triangle image and also an excuse to practise Julia. The inspiration is from the [this repo](https://github.com/google/brain-tokyo-workshop/tree/master/es-clip) based on [EvoJax](https://github.com/google/evojax).

Towards framing this as an optimization problem, we will represent a triangle as a vector of size 10, made up of floating point numbers between 0 and 1. Four numbers of this vector are for the color of the triangle; r,g,b and alpha, and for the three vertices of the triangle we need 6 numbers, (x1,y1), (x2,y2), (x3,y3). Hence, if we want to draw M images, each image having N triangles, we need a matrix of size (10,N,M), which will be our parameters matrix. I want to randomly create such a matrix and [min-max scale](https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_\(min-max_normalization\)) it along the triangle dimension, by which I mean, for each image, I first find the minimum and maximum of a triangle parameter among the N triangles, and then subtract from the parameter this minimum and then divide the result by the difference between the maximum and the minimum. I want to end up with an array of size (3,w,h,M) for the images, where w is width and h is height, and an array of size M for the distances. Let's see how fast we can do this.

First order of business is setting this up, note that I am scaling the numbers to the given width and height:

{% highlight julia %}
using Images
using Statistics
using Cairo
using BenchmarkTools
using Random: seed!

function prepare()
    seed!(123)
    w,h=128,128
    num_params=10
    num_triangles=50
    num_images=256
    target = rand(Float32, 3, h, w)
    prms = rand(Float32, num_params, num_triangles, num_images);
    prms .= (prms .- minimum(prms, dims=2)) ./ (maximum(prms, dims=2) .- minimum(prms, dims=2))   
    prms[collect(1:2:6),:,:] .*= w
    prms[collect(2:2:6),:,:] .*= h         
    return prms, target, num_images, num_triangles, w, h
end
{% endhighlight %}

## With a 2d library

The first thing that comes to mind is to use a 2d graphics library for drawing, and since the [Cairo lib](https://github.com/JuliaGraphics/Cairo.jl) is available, let's try that. The function below is drawing the triangles on a blank white Cairo canvas, and copying it to the img array at the end:

{% highlight julia %}
@views function renderCairo(img, prms, num_triangles, w, h)                  
    # blank white canvas
    buffer = ones(UInt32, w, h) * typemax(UInt32)    
    c = Cairo.CairoImageSurface(buffer, Cairo.FORMAT_ARGB32, flipxy=false)
    cr = CairoContext(c);        
    
    for i in 1:num_triangles
        q = prms[:,i]
        set_source_rgba(cr,q[7], q[8], q[9], q[10])        
        move_to(cr, q[1],q[2]);
        line_to(cr, q[3],q[4]);
        line_to(cr, q[5],q[6]);
        close_path(cr);
        fill(cr);            
    end        
            
    resultimg = reinterpret(ARGB32, permutedims(c.data, (2, 1)))
    resultchn = Float32.(channelview(RGB.(resultimg)))                
    img .= resultchn
    Cairo.finish(c)
    Cairo.destroy(c)        
end
{% endhighlight %}

Now let's draw each image in this fashion:

{% highlight julia %}
@views function withCairo() 
    prms, target, num_images, num_triangles, w, h = prepare() 

    imgs = Array{Float32}(undef, 3, h, w, num_images)    
    for i in 1:num_images
        img = imgs[:,:,:,i]        
        renderCairo(img, prms[:,:,i], num_triangles, w, h)        
    end
    dists = reshape(mean((imgs .- target) .^2, dims=(1,2,3)), num_images)
    return imgs, 1 .- dists
end
{% endhighlight %}

Benchmarking this with `@btime withCairo();` 

I see `428.101 ms (3157 allocations: 205.09 MiB)`.


## The cross product method


Now the cool part. Move your mouse inside and outside of the triangle below. You will see a bar chart, depicting the magnitude and direction of the 3rd components of the cross products, AB with AP (reds), BC with BP (greens) and CA with CP (blues). Observe that all those bars point to the same direction ONLY inside the triangle!

<div id='container' style="text-align: center;"></div>

<script>

let sketch = function(p) {

p.setup = function(){
    c = p.createCanvas(400, 400);
    c.style("touch-action", "none");
    // c.style.touchAction="none";
    p.background(1);
};

// p.draw = function() {
//         p.background(220);
//         p.triangle(30, 75, 58, 20, 86, 75);
//     };
// };

p.drawArrow = function(base, vec, myColor) {
  p.push();
  p.stroke(myColor);
  p.strokeWeight(3);
  p.fill(myColor);
  p.translate(base.x, base.y);
  p.line(0, 0, vec.x, vec.y);
  p.rotate(vec.heading());
  let arrowSize = 7;
  p.translate(vec.mag() - arrowSize, 0);
  p.triangle(0, arrowSize / 2, 0, -arrowSize / 2, arrowSize, 0);
  p.pop();
};

p.draw = function() {
        p.background(220);
        [x1,y1,x2,y2,x3,y3]=[60, 330, 70, 50, 360, 75];
        // triangle(x1,y1,x2,y2,x3,y3);  
        r = p.color(255, 0, 0)
        g = p.color(0, 255, 0)
        b = p.color(0, 0, 255)
        
        p.strokeWeight(4);
        p.stroke(r);
        // p.line(x1, y1, x2, y2);
        p.drawArrow(p.createVector(x1,y1), p.createVector(x2-x1,y2-y1), r);
        
        p.stroke(g);  
        // p.line(x2, y2, x3, y3);
        p.drawArrow(p.createVector(x2,y2), p.createVector(x3-x2,y3-y2), g);
        
        p.stroke(b);  
        // p.line(x3, y3, x1, y1);
        p.drawArrow(p.createVector(x3,y3), p.createVector(x1-x3,y1-y3), b);
        
        p.stroke(p.color(0, 0, 0));  
        
        
        p.strokeWeight(1);

        p.text("A", x1-10, y1+10);
        p.text("B", x2-20, y2);
        p.text("C", x3, y3-10);
        p.text("P", p.mouseX-20, p.mouseY);

        if (p.mouseX <= 400 && p.mouseX >= 0 && p.mouseY <= 400 && p.mouseY >= 0) {
            
            // p.line(x1, y1, p.mouseX, p.mouseY);
            // p.line(x2, y2, p.mouseX, p.mouseY);
            // p.line(x3, y3, p.mouseX, p.mouseY);
            p.drawArrow(p.createVector(x1,y1), p.createVector(p.mouseX-x1,p.mouseY-y1), r);    
            p.drawArrow(p.createVector(x2,y2), p.createVector(p.mouseX-x2,p.mouseY-y2), g);
            p.drawArrow(p.createVector(x3,y3), p.createVector(p.mouseX-x3,p.mouseY-y3), b);
            
            
            // p.push();
            cr1=(x2-x1)*(p.mouseY-y1)-(y2-y1)*(p.mouseX-x1)
            cr2=(x3-x2)*(p.mouseY-y2)-(y3-y2)*(p.mouseX-x2)
            cr3=(x1-x3)*(p.mouseY-y3)-(y1-y3)*(p.mouseX-x3)
            p.fill(r);
            p.rect(300, 300, 20, -cr1/1000);  
            
            p.translate(30,0);
            p.fill(g);
            p.rect(300, 300, 20, -cr2/1000);  
            
            p.translate(30,0);
            p.fill(b);
            p.rect(300, 300, 20, -cr3/1000);  
            
        
            p.translate(-60,0);
            p.text("3rd component of \n Cross Products =>", 180, 280);
            // p.pop();
        }
        
    };
};

new p5(sketch, 'container');
</script>

Whats great about this is that, cross products are just multiplications and subtractions, perfect job to parallelize with a GPU.

So, what needs to be done is clear. For each of the M images, and for each of the N triangles, our operation is to update a pixel color to blend with the current triangle's color, if that pixel is inside the triangle. We will parallelize this operation with a CUDA kernel, shown below:

{% highlight julia %}
using CUDA

function puttri(prms, imgs, tri, ins)    
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    abx = prms[3,tri,ins] - prms[1,tri,ins]
    aby = prms[4,tri,ins] - prms[2,tri,ins]
    apx = idx - prms[1,tri,ins]
    apy = idy - prms[2,tri,ins]
    cr1 = apx * aby - apy * abx

    bcx = prms[5,tri,ins] - prms[3,tri,ins]
    bcy = prms[6,tri,ins] - prms[4,tri,ins]
    bpx = idx - prms[3,tri,ins]
    bpy = idy - prms[4,tri,ins]
    cr2 = bpx * bcy - bpy * bcx

    cax = prms[1,tri,ins] - prms[5,tri,ins]
    cay = prms[2,tri,ins] - prms[6,tri,ins]
    cpx = idx - prms[5,tri,ins]
    cpy = idy - prms[6,tri,ins]
    cr3 = cpx * cay - cpy * cax

    if ((cr1>=0) & (cr2>=0) & (cr3>=0)) | ((cr1<=0) & (cr2<=0) & (cr3<=0))
        oneMinusAlpha = (1.0f0-prms[10,tri,ins])        
        imgs[1,idx,idy,ins] = imgs[1,idx,idy,ins] * oneMinusAlpha + prms[7,tri,ins] * prms[10,tri,ins]
        imgs[2,idx,idy,ins] = imgs[2,idx,idy,ins] * oneMinusAlpha + prms[8,tri,ins] * prms[10,tri,ins]
        imgs[3,idx,idy,ins] = imgs[3,idx,idy,ins] * oneMinusAlpha + prms[9,tri,ins] * prms[10,tri,ins]
    end
    return
end
{% endhighlight %}

We will need to pass our parameters and target array to the GPU, and then call the kernel with `@cuda`. We can create white canvases with `CUDA.ones` here, so no need to pass it.

{%highlight julia %}
function withGpu()
    prms, target, num_images, num_triangles, w, h = prepare()     

    prms = CuArray(prms)    
    imgs = CUDA.ones(3, h, w, num_images)
    target = CuArray(target)
    for tri in 1:num_triangles
        for i in 1:num_images
            @cuda threads=(32,32) blocks=(8,8) puttri(prms, imgs, tri, i)        
        end                                
    end
    gpufitnesses = 1.0f0 .- reshape(mean((imgs .- target) .^ 2, dims=(1,2,3)),num_images)
    return Array(imgs), Array(gpufitnesses)
end
{% endhighlight %}

Benchmarking this I see `120.315 ms (38689 allocations: 52.53 MiB)`

That's about 4x speedup, not really impressive, but perhaps not too bad considering I have an old GPU. Note that I benchmarked with `--check-bounds=no`, which is a startup option that you pass to Julia, when launching, that disables the performance killer "bounds checking". 

In the next post, I will talk about the very cool and general [PGPE](https://people.idsia.ch/~juergen/icann2008sehnke.pdf) algorithm used in [EvoJax](https://github.com/google/evojax) to steer these images towards a target image. You can see one example of this [here](https://cancandan.github.io/about/).

Please let me know if you have any comments, suggestions, improvements. 