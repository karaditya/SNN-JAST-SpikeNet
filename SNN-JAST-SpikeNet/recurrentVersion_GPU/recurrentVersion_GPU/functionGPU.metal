//
//  functionGPU.metal
//  recurrentVersion_GPU
//
//  Created by Aditya on 26/08/2022.
//

#include <metal_stdlib>
using namespace metal;

kernel void actInp(constant int *inpArr [[buffer(0)]],
                   device atomic_uint *contrArr [[buffer(1)]],
                   uint id  [[thread_position_in_grid]]){

    
    int outInd = inpArr[id];
    uint incr = uint(pow( 2, float(4*(outInd%8))));
    atomic_fetch_add_explicit(contrArr + int(outInd / 8) , incr, memory_order_relaxed);
}




kernel void procssInp(volatile device atomic_uint *contrArr [[buffer(0)]],
                      constant int *neurWipe [[buffer(1)]],
                      volatile device atomic_int *spikeList [[buffer(2)]],
                      uint id  [[thread_position_in_grid]]){

    int wipe = neurWipe[0];
    int wipeTh = neurWipe[1];
    int spikeTh = neurWipe[2];
    int maxSpikeIndx = neurWipe[3];
    
    uint count = atomic_load_explicit(contrArr + int(id / 8), memory_order_relaxed);
    int bits =  extract_bits(count, 4*(id%8), 4);

    if (bits >= spikeTh){
        
        uint decr = uint(pow( 2, float(4*(id%8))));
        atomic_fetch_add_explicit(contrArr + int(id / 8), -decr*bits, memory_order_relaxed);
        int posSpike = atomic_fetch_add_explicit(spikeList, 1, memory_order_relaxed) + 1;
        if (posSpike <= maxSpikeIndx) {
            
            atomic_fetch_add_explicit(spikeList + posSpike, id, memory_order_relaxed);
        }
    }
    

    else {
        
        if ((wipe == 1)  & (bits > 0)) {
            
            uint decr = uint(pow( 2, float(4*(id%8))));
            atomic_fetch_add_explicit(contrArr + int(id / 8), -decr*bits, memory_order_relaxed);
            
            
        }
        
        if ((wipe == 2)  & (bits > 0)) {
            
            uint decr = uint(pow( 2, float(4*(id%8))));
            atomic_fetch_add_explicit(contrArr + int(id / 8), -decr*1, memory_order_relaxed);
            
        }
        
        if ((wipe == 3) & (bits < wipeTh) & (bits > 0)){
            
            uint decr = uint(pow( 2, float(4*(id%8))));
            atomic_fetch_add_explicit(contrArr + int(id / 8), -decr*bits, memory_order_relaxed);
            
        }
    }
}
