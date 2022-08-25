//
//  functionGPU.metal
//  GPU_connection_conversion
//
//  Created by Aditya on 25/08/2022.
//

#include <metal_stdlib>
using namespace metal;

kernel void fillInpConnx(constant int *outBuff  [[buffer(0)]],
                    device atomic_int *inpBuff  [[buffer(1)]],
                         constant int *sizeArr  [[buffer(2)]],
                         constant int *batchArr [[buffer(3)]],
                         uint id  [[thread_position_in_grid]]){

    
    int outInd = id;
    int numConnx = sizeArr[0];
    int maxInpConnx = sizeArr[1];
    int end = numConnx*(id + 1);
    int inpBatchNum = batchArr[0];
    int inpBatchSize = batchArr[1];
    int outBatchNum = batchArr[2];
    int outBatchSize = batchArr[3];
    
    for (int i = numConnx*outInd; i < end; i ++){
        
        int inpInd = outBuff[i];
        if ((inpInd < (inpBatchNum + 1)*inpBatchSize) & (inpInd >= inpBatchNum*inpBatchSize)) {
            
            int pos = atomic_fetch_add_explicit(inpBuff + maxInpConnx*(inpInd - inpBatchNum*inpBatchSize), 1, memory_order_relaxed);
            if ((pos+1) < maxInpConnx) {
                
                atomic_fetch_add_explicit(inpBuff + 1 + (pos + maxInpConnx*(inpInd - inpBatchNum*inpBatchSize)), outInd + outBatchNum*outBatchSize , memory_order_relaxed);
                
            }
        }
    }
}
