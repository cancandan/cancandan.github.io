using Images
using Statistics
using Cairo

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

@views function render(img, prms, mesh, nt, w, h)    
    @inbounds for j in 1:nt        
        @inbounds for k in 1:w*h
            row, col = mesh[:,k]
            AP = mesh[:,k] .- prms[1:2,j]
            AB = prms[3:4,j] .- prms[1:2,j]            
            cr1 = AB[1] * AP[2] - AB[2] * AP[1]

            BP = mesh[:,k] .- prms[3:4,j]
            BC = prms[5:6,j] .- prms[3:4,j]
            cr2 = BC[1] * BP[2] - BC[2] * BP[1]

            CP = mesh[:,k] .- prms[5:6,j]
            CA = prms[1:2,j] .- prms[5:6,j]
            cr3 = CA[1] * CP[2] - CA[2] * CP[1]
                        
            if (cr1>=0 && cr2>=0 && cr3>=0) || (cr1<=0 && cr2<=0 && cr3<=0)
                img[:,col,row] .= (img[:,col,row] .* (1 .- prms[10,j])) .+ prms[7:9,j] .* prms[10,j]                
            end                
        end
    end
end

@views function renderAndComputeFitnesses(inputParams, target, w=200, h=200)    
    _,nt,ni = size(inputParams)    

    prms = copy(inputParams)

    (prms .- minimum(prms, dims=2)) ./ (maximum(prms, dims=2) .- minimum(prms, dims=2))    

    prms[collect(1:2:6),:,:] .*= w
    prms[collect(2:2:6),:,:] .*= h

    imgs = ones(Float32,3,h,w,ni);

    xv = ((1:w)' .* ones(Int32,h))
    yv = (ones(Int32,w)' .* (1:h))
    mesh = hcat(reshape(xv,:),reshape(yv,:))'
    
    dists = zeros(Float32, ni)
    @inbounds for i in 1:ni
        img = imgs[:,:,:,i]
        renderBetterVec(img, prms[:,:,i], mesh, nt, w, h)
        # renderCairo(img, prms[:,:,i], nt, w, h)
        dists[i] = mean((img .- target) .^2)
    end
    return imgs, 1 .- dists
end


@views function renderAndComputeFitnessesCairo(inputParams, target, w=200, h=200)    
    _,nt,ni = size(inputParams)    

    prms = copy(inputParams)

    (prms .- minimum(prms, dims=2)) ./ (maximum(prms, dims=2) .- minimum(prms, dims=2))    

    prms[collect(1:2:6),:,:] .*= w
    prms[collect(2:2:6),:,:] .*= h

    imgs = ones(Float32,3,h,w,ni);
        
    dists = zeros(Float32, ni)
    @inbounds for i in 1:ni
        img = imgs[:,:,:,i]
        # render(img, prms[:,:,i], mesh, nt, w, h)
        renderCairo(img, prms[:,:,i], nt, w, h)
        dists[i] = mean((img .- target) .^2)
    end
    return imgs, 1 .- dists
end

img_path = "assets/static/avni200.jpg";
imj = load(img_path);
target = Float32.(channelview(RGB.(imj)))

np=10
nt=50
ni=256

prms = rand(10,nt,ni);

using BenchmarkTools

# 898.475 ms (24000089 allocations: 1.80 GiB)
imgs, fitnesses = renderAndComputeFitnesses(prms, target);
@btime renderAndComputeFitnesses(prms, target);

# 9.230 ms (197 allocations: 19.48 MiB) => 100 times faster
imgs2, fitnesses2 = renderAndComputeFitnessesCairo(prms, target);
@btime renderAndComputeFitnessesCairo(prms, target);

colorview(RGB, imgs[:,:,:,1])
colorview(RGB, imgs2[:,:,:,1])



#### MASKING WORKING
function exampleMasking()
    h,w=60,120
    img = zeros(3,h,w) .+ 0.3
    colorview(RGB, img)


    xv = ((1:w)' .* ones(Int32,h))
    yv = (ones(Int32,w)' .* (1:h))
    mesh = hcat(reshape(xv,:),reshape(yv,:))'

    # condition = (mesh[1,:].>8) .& (mesh[1,:].<11) .& (mesh[2,:] .> 4)
    A = [110,0]
    B = [60,60] 
    v1 = B .- A
    v2 = mesh .- A

    condition = ((v2[1,:] .* v1[2]) .- (v2[2,:] .* v1[1])) .> 0
        
    img .+= reshape(condition, 1, h, w)
    colorview(RGB, img)
end

exampleMasking()

## TRIANGLE WITH MASKING
@views function putTriangle()
    h,w=60,120
    img = zeros(3,h,w) .+ 0.3
    # colorview(RGB, img)

    xv = ((1:w)' .* ones(Int32,h))
    yv = (ones(Int32,w)' .* (1:h))
    mesh = hcat(reshape(xv,:),reshape(yv,:))'
    
    A = [50,10]
    B = [110,45] 
    C = [10,50]

    ### 
    v1 = B .- A
    v2 = mesh .- A

    condition1 = ((v2[1,:] .* v1[2]) .- (v2[2,:] .* v1[1])) 

    v1 = C .- B
    v2 = mesh .- B

    condition2 = ((v2[1,:] .* v1[2]) .- (v2[2,:] .* v1[1])) 

    v1 = A .- C
    v2 = mesh .- C

    condition3 = ((v2[1,:] .* v1[2]) .- (v2[2,:] .* v1[1])) 
      
    condition = ((condition1 .>= 0) .& (condition2 .>= 0) .& (condition3 .>= 0)) .| ((condition1 .<= 0) .& (condition2 .<= 0) .& (condition3 .<= 0))
    
    #### 
    
    img .+= reshape(condition, 1, h, w)
    colorview(RGB, img)
end

putTriangle()


# VECTORIZED PUT SINGLE TRIANGLE ON MULTIPLE IMAGES
@views function putTriangle()#i, imgs, prms, mesh)    
    nt=50
    ni=10
    i = 1
    h,w=60,80
    xv = ((1:w)' .* ones(Int32,h))
    yv = (ones(Int32,w)' .* (1:h))
    mesh = hcat(reshape(xv,:),reshape(yv,:))'
    mesh = reshape(mesh, (2,w*h,1))
    mesh = repeat(mesh, outer=(1,1,ni))
    imgs = zeros(Float32,3,h,w,10)
    prms = rand(10,nt,ni)
    prms[collect(1:2:6),:,:] .*= w
    prms[collect(2:2:6),:,:] .*= h
        
    # func inside
    prms = prms[:,1,:]
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
    # colorview(RGB, imgs[:,:,:,1])
    return imgs
        
end
imgs = putTriangle()
colorview(RGB, imgs[:,:,:,1])





#### GPU VECTORIZED CROSS PRODUCT
@views function putTriangleGpu(imgs, prms, mesh, w, h)            
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
    @inbounds for i in 1:nt                                                                            
        putTriangleGpu(imgs, prms[:,i,:], mesh, w, h)
    end
    
    fitnesses = 1.0 .- dropdims(mean((imgs.-target).^2,dims=(1,2,3)), dims=(1,2,3))
    return imgs, fitnesses
end

img_path = "assets/static/avni200.jpg";
imj = load(img_path);
target = Float32.(channelview(RGB.(imj)))
np,nt,ni=10,50,256
prms = rand(10,nt,ni);

target = CuArray(target)
prms = CuArray(prms)
imgs, fitnesses = renderAndComputeFitnessesCrossProductVectorizedGPU(prms, target);

#### END GPU VECTORIZED CROSS PRODUCT