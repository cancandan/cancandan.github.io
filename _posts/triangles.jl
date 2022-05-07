using CUDA
using Images
using Statistics
using Cairo
using BenchmarkTools

#### VECTORIZED CROSS PRODUCT
@views function putTriangle(imgs, prms, mesh, w, h)            
    A = prms[1:2,:]
    B = prms[3:4,:]
    C = prms[5:6,:]
    color = prms[7:9,:]
    alpha = prms[10,:]
        
    v1 = B .- A
    v2 = mesh .- reshape(A, (2,1,ni))
    
    cr1 = ((v2[1,:,:] .* reshape(v1[2,:], 1, ni)) .- (v2[2,:,:] .* reshape(v1[1,:], 1, ni)))

    v1 = C .- B
    v2 = mesh .- reshape(B, (2,1,ni))

    cr2 = ((v2[1,:,:] .* reshape(v1[2,:], 1,ni)) .- (v2[2,:,:] .* reshape(v1[1,:], 1, ni))) 

    v1 = A .- C
    v2 = mesh .- reshape(C, (2,1,ni))

    cr3 = ((v2[1,:,:] .* reshape(v1[2,:], 1,ni)) .- (v2[2,:,:] .* reshape(v1[1,:], 1, ni))) 
      
    condition = ((cr1 .>= 0) .& (cr2 .>= 0) .& (cr3 .>= 0)) .| ((cr1 .<= 0) .& (cr2 .<= 0) .& (cr3 .<= 0))        

    condition = reshape(condition, 1, h, w, ni)
    imgs .= (imgs .* (.!condition)) .+ ( imgs .* condition .* (1.0 .- reshape(alpha,1,1,1,ni)) ) .+ (reshape(color, (3,1,1,ni)) .* condition .* reshape(alpha,1,1,1,ni))                
end

@views function renderAndComputeFitnessesCrossProductVectorized(inputParams, target, w=200, h=200)
    _,nt,ni = size(inputParams)    

    prms = copy(inputParams)

    prms .= (prms .- minimum(prms, dims=2)) ./ (maximum(prms, dims=2) .- minimum(prms, dims=2))    

    prms[collect(1:2:6),:,:] .*= w
    prms[collect(2:2:6),:,:] .*= h

    imgs = ones(Float32,3,h,w,ni);
    
    xv = ((1:w)' .* ones(Int32,h))
    yv = (ones(Int32,w)' .* (1:h))
    mesh = hcat(reshape(xv,:),reshape(yv,:))'
    mesh = reshape(mesh, (2,w*h,1))
    mesh = repeat(mesh, outer=(1,1,ni))        

    @inbounds for i in 1:nt                                                                            
        putTriangle(imgs, prms[:,i,:], mesh, w, h)
    end
    
    fitnesses = 1.0 .- dropdims(mean((imgs.-target).^2,dims=(1,2,3)), dims=(1,2,3))
    return imgs, fitnesses
end
#### VECTORIZED CROSS PRODUCT

###### CAIRO
@views function renderCairo(img, prms, nt, w, h)                  
    buffer = ones(UInt32, w, h) * typemax(UInt32)
    c = Cairo.CairoImageSurface(buffer, Cairo.FORMAT_ARGB32, flipxy=false)
    cr = CairoContext(c);        
    
    for i in 1:nt
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

@views function renderAndComputeFitnessesCairo(inputParams, target, w=200, h=200)    
    _,nt,ni = size(inputParams)    

    prms = copy(inputParams)

    prms .= (prms .- minimum(prms, dims=2)) ./ (maximum(prms, dims=2) .- minimum(prms, dims=2))    

    prms[collect(1:2:6),:,:] .*= w
    prms[collect(2:2:6),:,:] .*= h

    imgs = ones(Float32,3,h,w,ni);
            
    @inbounds for i in 1:ni
        img = imgs[:,:,:,i]        
        renderCairo(img, prms[:,:,i], nt, w, h)        
    end
    fitnesses = 1.0 .- dropdims(mean((imgs.-target).^2,dims=(1,2,3)), dims=(1,2,3))
    return imgs, fitnesses
end
###### END CAIRO


#### GPU VECTORIZED CROSS PRODUCT
@views function putTriangleGpu(i, ni, imgs, prms, mesh, w, h, A, B, C, AB, BC, CA)
    
    A = A[:,i,:]
    B = B[:,i,:]
    C = C[:,i,:]
    color = prms[7:9,i,:]
    alpha = prms[10,i,:]
        
    v2 = mesh .- reshape(A, (2,1,ni))
    
    cr1 = ((v2[1,:,:] .* reshape(  AB[2,i,:]   , 1, ni)) .- (v2[2,:,:] .* reshape( AB[1,i,:] , 1, ni)))
        
    v2 = mesh .- reshape(B, (2,1,ni))

    cr2 = ((v2[1,:,:] .* reshape(  BC[2,i,:] , 1,ni)) .- (v2[2,:,:] .* reshape( BC[1,i,:] , 1, ni))) 
    
    v2 = mesh .- reshape(C, (2,1,ni))

    cr3 = ((v2[1,:,:] .* reshape(  CA[2,i,:] , 1,ni)) .- (v2[2,:,:] .* reshape( CA[1,i,:], 1, ni))) 
      
    condition = reshape(((cr1 .>= 0) .& (cr2 .>= 0) .& (cr3 .>= 0)) .| ((cr1 .<= 0) .& (cr2 .<= 0) .& (cr3 .<= 0)), 1, h, w, ni)
        
    alpha = reshape(alpha,1,1,1,ni)
    imgs .= (imgs .* (.!condition)) .+ ( imgs .* condition .* (1.0 .- alpha) ) .+ (reshape(color, (3,1,1,ni)) .* condition .* alpha)                
    
end


@views function renderAndComputeFitnessesCrossProductVectorizedGPU(inputParams, target, w=200, h=200)
    _,nt,ni = size(inputParams)    

    # prms = copy(inputParams)

    prms .= (prms .- minimum(prms, dims=2)) ./ (maximum(prms, dims=2) .- minimum(prms, dims=2))    

    prms[collect(1:2:6),:,:] .*= w
    prms[collect(2:2:6),:,:] .*= h

    imgs = ones(Float32,3,h,w,ni);
    imgs = CuArray(imgs)
    
    xv = ((1:w)' .* ones(Int32,h))
    yv = (ones(Int32,w)' .* (1:h))
    mesh = hcat(reshape(xv,:),reshape(yv,:))'
    mesh = reshape(mesh, (2,w*h,1))
    mesh = repeat(mesh, outer=(1,1,ni))        
    
    mesh = CuArray(mesh)
    
    A = prms[1:2,:,:]
    B = prms[3:4,:,:]
    C = prms[5:6,:,:]

    AB = B .- A
    BC = C .- B
    CA = A .- C

    @inbounds for i in 1:nt                                                                            
        putTriangleGpu(i, ni, imgs, prms, mesh, w, h, A, B, C, AB, BC, CA)
    end
    
    fitnesses = 1.0 .- dropdims(mean((imgs.-target).^2,dims=(1,2,3)), dims=(1,2,3))
    return imgs, fitnesses
end
#### END GPU VECTORIZED CROSS PRODUCT


#### SETUP
img_path = "assets/static/avni200.jpg";
imj = load(img_path);
target = Float32.(channelview(RGB.(imj)))
np,nt,ni=10,50,256
prms = rand(10,nt,ni);
#### END SETUP

#### MEASURE PERFORMANCE
imgs, fitnesses = renderAndComputeFitnessesCairo(prms, target);
# 790.891 ms (3151 allocations: 499.17 MiB)
@btime imgs, fitnesses = renderAndComputeFitnessesCairo(prms, target);

imgs, fitnesses = renderAndComputeFitnessesCrossProductVectorized(prms, target);
# 371.469 ms (1611 allocations: 291.78 MiB)
@btime imgs, fitnesses = renderAndComputeFitnessesCrossProductVectorized(prms, target);

target = CuArray(target)
prms = CuArray(prms)
imgs, fitnesses = renderAndComputeFitnessesCrossProductVectorizedGPU(prms, target);

# 2.777 s (33488 allocations: 279.25 MiB)
@btime imgs, fitnesses = renderAndComputeFitnessesCrossProductVectorizedGPU(prms, target);
imgs = Array(imgs);
#### END MEASURE PERFORMANCE

#### CHECK
colorview(RGB, imgs[:,:,:,1])
#### END CHECK


### KERNEL
function puttri(prms, imgs, tri, ins)    
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    # gidx = gridDim().x    
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
        oneMinusAlpha = (1-prms[10,tri,ins])
        imgs[1,idx,idy,ins] = imgs[1,idx,idy,ins] * oneMinusAlpha + prms[7,tri,ins] * prms[10,tri,ins]
        imgs[2,idx,idy,ins] = imgs[1,idx,idy,ins] * oneMinusAlpha + prms[8,tri,ins] * prms[10,tri,ins]
        imgs[3,idx,idy,ins] = imgs[1,idx,idy,ins] * oneMinusAlpha + prms[9,tri,ins] * prms[10,tri,ins]
    end
    return
end

# function kernelFitnesses(imgs, target, fit, ins)
#     cache = @cuDynamicSharedMem(Int64, 1024)

#     idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  
#     idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    
#     cacheIndex = threadIdx().x - 1 + (threadIdx().y - 1)*32
#     i::Int = blockDim().x ÷ 2
#     j::Int = blockDim().y ÷ 2
#     while i!=0
#         if cacheIndex < i
#             # cache[cacheIndex + 1] += cache[cacheIndex + i + 1]
#             fit[ins] += (imgs[1,idx,idy,ins] - target[1,idx+i,idy+i,ins])^2
#         end
#         sync_threads()
#         i = i ÷ 2
#     end
#     if cacheIndex == 0
#         c[blockIdx().x] = cache[1]
#     end
# end

w,h=256,256
totins=256
numtri=50
prms = rand(Float32, 10, numtri, totins)
prms[collect(1:2:6),:,:] .*= w
prms[collect(2:2:6),:,:] .*= h
imgs = ones(Float32, 3,h,w, totins);
prms = CuArray(prms)
imgs = CuArray(imgs)


target = rand(Float32, 3, h, w)
target = CuArray(target)
function bench()
    for tri in 1:numtri
        for i in 1:totins
            @cuda threads=(32,32) blocks=(8,8) puttri(prms, imgs, tri, i)        
        end                        
    end
    gpufitnesses = mean((imgs[:,:,:,:] .- target) .^ 2, dims=(1,2,3))
    return
end
@btime bench()





colorview(RGB, Array(imgs[:,:,:,1]))

# fit = zeros(Float32, totins)
# fit = CuArray(fit)
# @cuda kernelFitnesses(imgs, target, fit)


# KERNEL