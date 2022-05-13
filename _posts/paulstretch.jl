using FileIO: load, save, loadstreaming, savestreaming
using LibSndFile
using FFTW

function optimize_windowsize(n)
    orig_n=n
    while true
        n=orig_n
        while mod(n,2)==0
            n/=2
        end
        while mod(n,3)==0
            n/=3
        end
        while mod(n,5)==0
            n/=5
        end
        if n<2
            break
        end
        orig_n+=1
    end
    return orig_n
end

function doit()
    x, sr = load("assets/static/oxp.wav")
    
    stretch=8.0
    windowsize_seconds = 0.5

    x = Float32.(x)
    windowsize=Int(trunc(windowsize_seconds*sr))    
    if windowsize<16
        windowsize=16
    end
    windowsize=optimize_windowsize(windowsize)    
    windowsize=(windowsize÷2)*2
    half_windowsize=Int(windowsize/2)
    nsamples=size(x,1)
    end_size=Int(floor(sr*0.05))
    if end_size<16
        end_size=16
    end

    x[1+nsamples-end_size:nsamples,1] .*= LinRange(1., 0., end_size)
    
    displace_pos=(windowsize*0.5)/stretch
    
    window = (1.0 .- (LinRange{Float32}(-1., 1., windowsize).^2)).^1.25
    old_windowed_buf=zeros(windowsize,2)

    start_pos=1
    savestreaming("paulout.wav", 2, Int(sr), Float32) do dest
        while true
            istart_pos=Int(floor(start_pos))
            buf=x[istart_pos:istart_pos+windowsize-1,:]
            if size(buf,1)<windowsize
                append!(buf,zeros(2,windowsize-size(buf,1),2))
            end
            buf.=buf.*window
                
            freqs=abs.(rfft(buf,1))

            ph = (rand(Float32, size(freqs,1),2)*2pi)im        
            freqs = freqs .* exp.(ph)

            buf = irfft(freqs,size(buf,1))

            buf .= buf .* window
        
            output=buf[1:half_windowsize,:].+old_windowed_buf[half_windowsize:windowsize-1,:]
            old_windowed_buf=buf
                    
            clamp!(output, -1.0, 1.0)
                                
            write(dest, output)
            
            start_pos+=displace_pos
            if start_pos>=nsamples                    
                break
            end
        end
    end
end  



    x, sr = load("assets/static/oxp.wav")
    
    stretch=8.0
    windowsize_seconds = 0.5

    x = Float32.(x)
    windowsize=Int(trunc(windowsize_seconds*sr))    
    if windowsize<16
        windowsize=16
    end
    windowsize=optimize_windowsize(windowsize)    
    windowsize=(windowsize÷2)*2
    half_windowsize=Int(windowsize/2)
    nsamples=size(x,1)
    end_size=Int(floor(sr*0.05))
    if end_size<16
        end_size=16
    end

    x[1+nsamples-end_size:nsamples,1] .*= LinRange(1., 0., end_size)
    
    displace_pos=(windowsize*0.5)/stretch
    
    window = (1.0 .- (LinRange{Float32}(-1., 1., windowsize).^2)).^1.25
    # plot lines(range(0,windowsize,windowsize), window)
    old_windowed_buf=zeros(windowsize,2)

    start_pos=1
    # savestreaming("paulout.wav", 2, Int(sr), Float32) do dest
        # while true
            istart_pos=Int(floor(start_pos))

            
            buf=x[istart_pos:istart_pos+windowsize-1,:]
            lines(range(1,size(buf,1),size(buf,1)), buf[:,1])

            if size(buf,1)<windowsize
                append!(buf,zeros(2,windowsize-size(buf,1),2))
            end
            
            buf.=buf.*window
            lines(range(1,size(buf,1),size(buf,1)), buf[:,1])
                
            
            freqs=abs.(rfft(buf,1))
            lines(range(1,size(freqs,1),size(freqs,1)), freqs[:,1])

            ph = (rand(Float32, size(freqs,1),2)*2pi)im        
            freqs = freqs .* exp.(ph)            

            buf = irfft(freqs,size(buf,1))
            lines(range(1,size(buf,1),size(buf,1)), buf[:,1])

            buf .= buf .* window
            lines(range(1,size(buf,1),size(buf,1)), buf[:,1])
        
            output=buf[1:half_windowsize,:].+old_windowed_buf[half_windowsize:windowsize-1,:]
            lines(range(1,size(output,1),size(output,1)), output[:,1])

            old_windowed_buf=buf
                    
            clamp!(output, -1.0, 1.0)
                                
            # write(dest, output)
            
            start_pos+=displace_pos
            if start_pos>=nsamples                    
                break
            end
        # end
    # end


doit()
