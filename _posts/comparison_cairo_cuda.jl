using CUDA
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

@views function renderCairo(img, prms, num_triangles, w, h)                  
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

imgs_gpu, fitnesses_gpu = withGpu();
colorview(RGB, permutedims(imgs_gpu[:,:,:,1], (1,3,2)))

imgs_cairo, fitnesses_cairo = withCairo();
colorview(RGB, imgs_cairo[:,:,:,1])

fitnesses_gpu ≈ fitnesses_cairo

isapprox(fitnesses_cairo, fitnesses_gpu, rtol=0.005)

@btime withCairo();
@btime withGpu();