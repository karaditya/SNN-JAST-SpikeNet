//
//  main.swift
//  GPU_connection_conversion
//
//  Created by Aditya on 25/08/2022.
//

import Foundation
import MetalKit

typealias DType = Int32

let numIn:    DType = 10000000
let numOut:   DType = 100000000
let numConnx: DType = 4
let maxInpConnx: DType = 200

// Batch parameters

let inpBatchSize: DType = 100000 // numIn : for not doing it in batches
let outBatchSize: DType = 10000000 // numOut: for not doing it in batches

let inpBatchArr: [DType] = Array<DType>(0..<(numIn/inpBatchSize))
let outBatchArr: [DType] = Array<DType>(0..<(numOut/outBatchSize))


// Ranges for sub batches (Use only for generating billion file: use file write mode "wb" for range1 and mode "ab" for the subsequent ranges)

let range1 = 0 ..< (inpBatchArr.count/8)
let range2 = (inpBatchArr.count/8) ..< (inpBatchArr.count/4)
let range3 = (inpBatchArr.count/4) ..< (3*inpBatchArr.count/8)
let range4 = (3*inpBatchArr.count/8) ..< (inpBatchArr.count/2)
let range5 = (inpBatchArr.count/2) ..< (5*inpBatchArr.count/8)
let range6 = (5*inpBatchArr.count/8) ..< (6*inpBatchArr.count/8)
let range7 = (6*inpBatchArr.count/8) ..< (7*inpBatchArr.count/8)
let range8 = (7*inpBatchArr.count/8) ..< inpBatchArr.count

let sizeArr: [DType] = [numConnx, maxInpConnx]


let dirPath = "/Users/aditya/Desktop/SNN-JAST-SpikeNet/benchmark"

// Generate output connections file
let fileName = dirPath + "/Connections_7.bin"
let fileNameInp = dirPath + "/InpConnections_7.bin"
let fileUrl = URL(fileURLWithPath: fileName)


// Generate input connections file
var fpInp: UnsafeMutablePointer<FILE>?

let checkFile = false

if (checkFile == false) {

    fpInp = fopen(fileNameInp, "wb")

    // Make output file handle
    var fpOut: UnsafeMutablePointer<FILE>?

    fpOut = fopen(fileName, "rb")

    
    // Make device state
    let device = MTLCreateSystemDefaultDevice()


    let sizeBuff = device?.makeBuffer(bytes: sizeArr,
                                      length: MemoryLayout<DType>.stride*sizeArr.count,
                                      options: .storageModeShared)



    // Start timer
    let start = CFAbsoluteTimeGetCurrent()

   
    for inpBatch in inpBatchArr {
        
        
//        print("Input Batch: \(inpBatch)")
        
        let inpConnx: [DType] = Array<DType>(repeating: 0, count: Int(inpBatchSize)*Int(maxInpConnx))
        
        let inpSize = Int(inpBatchSize) * Int(maxInpConnx) * MemoryLayout<DType>.stride
        
        let inpConnxBuff = device?.makeBuffer(bytes: inpConnx,
                                              length: inpSize,
                                              options: .storageModeShared)
        
        
        
        // Start Kernel 3 for connection conversion

        let commandQueue = device?.makeCommandQueue()

        let getFunctionLibrary = device?.makeDefaultLibrary()

        let fillInpConnx = getFunctionLibrary?.makeFunction(name: "fillInpConnx")

        var computePipelineState: MTLComputePipelineState!

        do {
            computePipelineState = try device?.makeComputePipelineState(function: fillInpConnx!)
        } catch {
            print(error)
        }
        
        
        
        
        for outBatch in outBatchArr {
         
            
//            print("Output Batch: \(outBatch)")

            let batchArr: [DType] = [inpBatch, inpBatchSize, outBatch, outBatchSize]

            let buffer = UnsafeMutablePointer<DType>.allocate(capacity: Int(outBatchSize)*Int(numConnx))
            buffer.initialize(repeating: 0, count: Int(outBatchSize)*Int(numConnx))
            let size = Int(outBatchSize)*Int(numConnx) * MemoryLayout<DType>.stride
            fseek(fpOut, Int(outBatch) * size, SEEK_SET)
            fread(buffer, size, 1, fpOut)
           

            let outConnxBuff = device?.makeBuffer(bytes: buffer,
                                                  length: size,
                                                  options: .storageModeShared)


            let batchBuff = device?.makeBuffer(bytes: batchArr,
                                               length: MemoryLayout<DType>.stride*batchArr.count,
                                               options: .storageModeShared)

            
            
            // Encode command buffer
            let commandBuff = commandQueue?.makeCommandBuffer()

            let commandEncoder = commandBuff?.makeComputeCommandEncoder()
            commandEncoder?.setComputePipelineState(computePipelineState)
            
            
            commandEncoder?.setBuffer(inpConnxBuff, offset: 0, index: 1)
            commandEncoder?.setBuffer(sizeBuff, offset: 0, index: 2)
            

            // Set buffers
            commandEncoder?.setBuffer(outConnxBuff, offset: 0, index: 0)
            commandEncoder?.setBuffer(batchBuff, offset: 0, index: 3)

            let w = computePipelineState.maxTotalThreadsPerThreadgroup
            let thGRW = Int(outBatchSize)
            let threadsPerThreadgroup = MTLSizeMake(w, 1, 1)
            let threadsPerGrid = MTLSize(width: thGRW,
                                         height: 1,
                                         depth: 1)


            commandEncoder?.dispatchThreads(threadsPerGrid,
                                            threadsPerThreadgroup: threadsPerThreadgroup)

            



            buffer.deinitialize(count: size)
            buffer.deallocate()
            
            commandEncoder?.endEncoding()
            commandBuff?.commit()

            commandBuff?.waitUntilCompleted()
            

        }
        
        


    let inpConnxPointer = inpConnxBuff?.contents().bindMemory(to: DType.self, capacity: inpSize)
    fwrite(inpConnxPointer, inpSize, 1, fpInp)



 }


    fclose(fpInp)
    fclose(fpOut)



    // End timer
    let end = CFAbsoluteTimeGetCurrent()

    // Print out the time elapsed
    print("Time elapsed in GPU load = \(end - start)")
    

}


else {

    //  Read file to verify
    var inpConnx1: [DType] = Array<DType>(repeating: 0, count: Int(maxInpConnx))
    let inpNum = 0 // Set a number to view the corresponding input neuron
    print("Range of inpBatch: \(inpBatchArr[range2])")



    fpInp = fopen(fileNameInp, "rb")
    let inpBuffer = UnsafeMutablePointer<DType>.allocate(capacity: inpConnx1.count)
    inpBuffer.initialize(repeating: 0, count: inpConnx1.count)
    let size =  MemoryLayout<DType>.stride*inpConnx1.count
    fseek(fpInp, inpNum * size, SEEK_SET)
    fread(inpBuffer, size, 1, fpInp)


    // Print out the connections
    for ind in 0..<inpConnx1.count {

        let tmp = inpBuffer.advanced(by: ind)
        inpConnx1[ind] = tmp.pointee
    }

    inpBuffer.deinitialize(count: size) //`size` and `buffer` may be re-written at this point
    free(inpBuffer)
    fclose(fpInp)

    print("Connections after reading: \(inpConnx1)")

}
