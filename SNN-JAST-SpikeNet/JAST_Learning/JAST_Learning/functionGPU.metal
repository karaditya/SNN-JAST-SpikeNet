//
//  functionGPU.metal
//  JAST_Learning
//
//  Created by Aditya on 26/08/2022.
//

#include <metal_stdlib>
using namespace metal;


kernel void actInp(constant int *outArr [[buffer(0)]],
                   constant int *actVect [[buffer(1)]],
                   volatile device atomic_uint *contrArr [[buffer(2)]],
                   constant int *sizeArr [[buffer(3)]],
                   uint2 id  [[thread_position_in_grid]]){
    
    int xPos = id.x;
    int yPos = id.y;

    int numOut = sizeArr[0];
    int sizeact = sizeArr[1];
    int numConnx = sizeArr[2];
    
    if ((xPos >= numOut) || (yPos >= sizeact)) {
        
        return;
    }
    
    
    for (int j = 0; j < numConnx; j++) {
        
        int inpInd = outArr[numConnx*xPos + j];
        if (inpInd == actVect[yPos]) {
            uint incr = uint(pow( 2, float(4*(xPos%8))));
            atomic_fetch_add_explicit(contrArr + int(xPos / 8) , incr, memory_order_relaxed);
            
        }
    }
    
}





kernel void procssInp(volatile device atomic_uint *contrArr [[buffer(0)]],
                      constant int *neurWipe [[buffer(1)]],
                      volatile device atomic_int *spikeList [[buffer(2)]],
                      volatile device atomic_int *learnList [[buffer(3)]],
                      uint id  [[thread_position_in_grid]]){
    

    int wipe = neurWipe[0];
    int wipeTh = neurWipe[1];
    int spikeTh = neurWipe[2];
    int maxSpikeIndx = neurWipe[3];
    int learnTh = neurWipe[4];
    int maxLearn = neurWipe[5];
    
    uint count = atomic_load_explicit(contrArr + int(id / 8), memory_order_relaxed);
    int bits =  extract_bits(count, 4*(id%8), 4);
    
    if (bits == learnTh){
        
        int posLearn = atomic_fetch_add_explicit(learnList, 1, memory_order_relaxed) + 1;
        if (posLearn <= maxLearn){
            
            atomic_fetch_add_explicit(learnList + posLearn, id, memory_order_relaxed);
        }
    }
    
    if (bits >= spikeTh){
        
        int decr = -int(pow( 2, float(4*(id%8))));
        atomic_fetch_add_explicit(contrArr + int(id / 8), decr*bits, memory_order_relaxed);
        int posSpike = atomic_fetch_add_explicit(spikeList, 1, memory_order_relaxed) + 1;
        if (posSpike <= maxSpikeIndx){
            
            atomic_fetch_add_explicit(spikeList + posSpike, id, memory_order_relaxed);
        }
    }
    

    else {
        
        if ((wipe == 1)  & (bits > 0)) {
            
            int decr = -int(pow( 2, float(4*(id%8))));
            atomic_fetch_add_explicit(contrArr + int(id / 8), decr*bits, memory_order_relaxed);
            
            
        }
        
        if ((wipe == 2)  & (bits > 0)) {
            
            int decr = -int(pow( 2, float(4*(id%8))));
            atomic_fetch_add_explicit(contrArr + int(id / 8), decr*1, memory_order_relaxed);
            
        }
        
        if ((wipe == 3) & (bits < wipeTh) & (bits > 0)){
            
            int decr = -int(pow( 2, float(4*(id%8))));
            atomic_fetch_add_explicit(contrArr + int(id / 8), decr*bits, memory_order_relaxed);
            
        }
    }
}




kernel void JAST(constant int *learnArr [[buffer(0)]],
            volatile device atomic_int *learnOut [[buffer(1)]],
            constant int *learnAct [[buffer(2)]],
            uint2 id  [[thread_position_in_grid]]){


    int learnTh = learnArr[0];
    int slideWindow = learnArr[1];
    int xSize = learnArr[2];
    int ySize = learnArr[3];
    int numConnx = learnArr[4];
    int xPos = id.x;
    int yPos = id.y;

    if ((xPos >= xSize) || (yPos >= ySize-slideWindow)) {

        return;
    }
    

    int arr1[20];
    
    for (int i = 0; i < numConnx; i ++){
        
        arr1[i] = atomic_load_explicit(learnOut + (4*xPos + i), memory_order_relaxed);

    }
    
    int arr2[20];
    
    for (int i = 0; i < slideWindow; i ++){
        
        arr2[i] = learnAct[yPos+i];

    }
    
    int arr3[20];
    arr3[0] = 0;
    int count = 0;
    
    for (int i = 0; i < slideWindow; i ++){

        int num = arr2[i];
        bool flag = false;
        
        for (int j = 0; j < numConnx; j ++){

            if (num == arr1[j]){

                count ++;
                flag = true;
                arr1[j] = -3;

            }
        }

        if (flag == false){
            
            arr3[0]++;
            int ind = arr3[0];
            arr3[ind] = num;

        }

    }

    threadgroup_barrier(mem_flags::mem_device);

    if (count == learnTh){

        for (int exInd = 0; exInd < numConnx; exInd ++){

            int num = arr1[exInd];
            if (num != -3) {
                
                int ind = arr3[0];
                for (int i = 1; i < ind; i ++){
                    
                    if (arr3[i] != -3){
                       
                        atomic_exchange_explicit(learnOut + (4*xPos + exInd) ,  arr3[i] , memory_order_relaxed);
                        arr3[i] = -3;
                        break;
                    }
                }
            }
        }
    }
}
