//
//  main.swift
//  binaryFileVersion
//
//  Created by Aditya on 25/08/2022.
//

import MetalKit
typealias DType = Int32


// Architecture Size parameters
let numIn:    DType = 100000
let numOut:   DType = 1000000
let numAct:   DType = 1000
let numConnx: DType = 4
let maxInpConnx: DType = 200

/** EXPERIMENTS PERFORMED **/

// 10, 10, 7 :: 10 : 0.03822 // 24
// 100, 1000, 10 :: 60 : 0.032052 // 394
// 10000, 100000, 100 :: 100 : 0.031723 // 4042

// 100000, 1000000, 1000 :: 200 : : apple: 0.02669, 10: 0.03419, 100: 0.03952, 1000: 0.04379, 10000: 0.03808, 100000: 0.042938, 1000000: 0.063506 // 40179


// 1000000, 10000000, 10000 :: 200 : apple: 0.030565, 10: 0.03001, 100: 0.03104, 1000: 0.03080, 10000: 0.03390, 100000: 0.034021, 1000000: 0.050273 // 399276

// 10000000, 100000000, 100000 :: 200 : apple: 0.07396, 10: 0.07469, 100: 0.08088, 1000: 0.081058, 10000: 0.0758290, 100000: 0.07533, 1000000: 0.086197 // 4027548

// 100000000, 1000000000, 1000000 :: 560 :  apple: 0.72009, 10: 0.70561, 100: 0.76063, 1000: 0.69757, 10000: 0.667329, 100000: 0.65321, 1000000: 0.66840 // 56510250


// Neuron Wipe parameters
let spikeTh: DType = 4
let neurWipe: DType = 2
let wipeTh: DType = 2
let maxSpikes: DType = 10

// Other parameters
let verbose: DType = 0  // no print for verbose zero
let copyRes: Bool  = false  // copy results from buffer to array bool (set true when verbose > 0)
let sample: Bool = false // sample read
let exitGPU: Bool = sample // for exiting GPU compute while sampling; user can set explicitly true if required
let appleRec = false // Setting this to true will ignore the given gridSize and opt for Apple's recommended size

// Input and Output Filenames
let pathDir: String = "/Users/aditya/Desktop/SNN-JAST-SpikeNet" // Change this directory according to user
let outFileName: String = pathDir + "/benchmark/Connections_6.bin"
let inpFileName: String = pathDir + "/benchmark/InpConnections_6.bin"



// Kernel 1 Thread parameters
let thrGroup = 1024 // Apple recommend: 1024 (max possible threadgroups)

let blockSize = thrGroup
var gridSize = 1000000 // Apple recommend: (inpNeurArr.count+blockSize-1) / blockSize



// Kernel 2 Thread parameters
let prcssThrGroup = 1024 // Apple recommend: 1024 (max possible threadgroups)

let prcssBlockSize = prcssThrGroup
var prcssGridSize =  1000 // Apple recommend: (Int(numOut)+prcssBlockSize-1) / prcssBlockSize

// will set apple recommendation for neuron wipe
if (appleRec == true){
    
    prcssGridSize = (Int(numOut)+prcssBlockSize-1) / prcssBlockSize
    
}

/** FUNCTIONS FOR SAMPLING **/

// Randomly show a few reads
func showOutput(_ outFileName: String,_ numOut: DType,_ numConnx: DType ,_ ind: Int? = nil) {
    
    let outFp: UnsafeMutablePointer<FILE>?
    outFp = fopen(outFileName, "rb")
    var outInd: Int
    if ind == nil {
        
        outInd =  Int.random(in: 0..<Int(numOut)) // 23233909
        
    }
    else {
        
        outInd = ind!
    }
    
    let outBuffer = UnsafeMutablePointer<DType>.allocate(capacity: Int(numConnx))
    outBuffer.initialize(repeating: 0, count: Int(numConnx))
    let size = Int(numConnx) * MemoryLayout<DType>.stride
    
    fseek(outFp, outInd*size, SEEK_SET)
    fread(outBuffer, size, 1, outFp)
    
    print("Printing Output Connections of : \(outInd)")
    for i in 0..<numConnx {
    
        print(outBuffer.advanced(by: Int(i)).pointee)
    
    }
    fclose(outFp)
    outBuffer.deinitialize(count: size) //`size` and `buffer` may be re-written at this point
    free(outBuffer)
}

func showInput(_ inpFileName: String,_ numIn: DType,_ maxInpConnx: DType ,_ ind: Int? = nil) {
    
    let inpFp: UnsafeMutablePointer<FILE>?
    inpFp = fopen(inpFileName, "rb")
    var inpInd: Int
    if ind == nil {
        
        inpInd =  Int.random(in: 0..<Int(numIn))
        
    }
    else {
        
        inpInd = ind!
    }
    
    let inpBuffer = UnsafeMutablePointer<DType>.allocate(capacity: Int(maxInpConnx))
    inpBuffer.initialize(repeating: 0, count: Int(maxInpConnx))
    let size = Int(maxInpConnx) * MemoryLayout<DType>.stride
    
    fseek(inpFp, inpInd*size, SEEK_SET)
    fread(inpBuffer, size, 1, inpFp)
    
    print("Printing Input Connections of : \(inpInd)")
    for i in 0..<maxInpConnx {
    
        print(inpBuffer.advanced(by: Int(i)).pointee)
    
    }
    fclose(inpFp)
    inpBuffer.deinitialize(count: size) //`size` and `buffer` may be re-written at this point
    free(inpBuffer)
}

if sample != false {
    
    // Call functions to sample a few connections for viewing
    showOutput(outFileName, numOut, numConnx)
    showInput(inpFileName, numIn, maxInpConnx)
    print("end_of_sample")
}

// Exit GPU while sampling
if exitGPU != false {
    
    exit(1)
    
}





/** SPIKE PROPAGATION PHASE**/

// Function for reading only the activated input connection from binary file
func actInpConnx(_ inpFileName: String,_ actVect: [DType]) -> [DType] {
    
    var inpFp: UnsafeMutablePointer<FILE>?
    inpFp = fopen(inpFileName, "rb")
    
    var inpNeurArr: [DType] = []
    for i in 0..<actVect.count {

        let inpInd = Int(actVect[i])
        let inpBuffer = UnsafeMutablePointer<DType>.allocate(capacity: Int(maxInpConnx))
        inpBuffer.initialize(repeating: 0, count: Int(maxInpConnx))
        let inpSize = Int(maxInpConnx) * MemoryLayout<DType>.stride

        fseek(inpFp, inpInd*inpSize, SEEK_SET)
        fread(inpBuffer, inpSize, 1, inpFp)


        var count = inpBuffer.pointee
        if (count >= maxInpConnx) {
            
            count = maxInpConnx-1
            
        }
        
        for j in 0..<count {

            inpNeurArr.append(inpBuffer.advanced(by: Int(j) + 1).pointee)
        }

        inpBuffer.deinitialize(count: inpSize) //`size` and `buffer` may be re-written at this point
        free(inpBuffer)


    }
    fclose(inpFp)
    
    
    return inpNeurArr
}


// Make seed generator (optional for comparing)
struct RandomNumberGeneratorWithSeed: RandomNumberGenerator {
    init(seed: Int) {
        // Set the random seed
        srand48(seed)
    }

    func next() -> UInt64 {
        // drand48() returns a Double, transform to UInt64
        return withUnsafeBytes(of: drand48()) { bytes in
            bytes.load(as: UInt64.self)
        }
    }
}
var generator = RandomNumberGeneratorWithSeed(seed: 708)





// Make activation function to create random activation vector
func makeActVect(_ numAct: DType,_ numIn: DType) -> [DType] {

    return (1...numAct).map( {_ in DType.random(in: 0..<numIn, using: &generator)} )

}



// Randomly activate inputs
let actVect: [DType] = makeActVect(numAct, numIn)

// Fill in the activated input connections in a 1D-array
let inpNeurArr: [DType] = actInpConnx(inpFileName, actVect)



if verbose != 0 {

print("Activated input indices: \(actVect)")
print("Connection array of activated inputs: \(inpNeurArr)")

}

print("Count of events processed: \(inpNeurArr.count)")

// Create zero counter list and size array to be sent to the GPU
var contrArr: [UInt32] = Array<UInt32>(repeating: 0, count: Int(numOut / 8) + 1 )
let sizeArr: [DType] = [DType(inpNeurArr.count), numOut]



// Start total timer
let start = CFAbsoluteTimeGetCurrent()

// Create GPU device state for compute kernels
let device = MTLCreateSystemDefaultDevice()


// Run the first kernel to register spike counts from the activated inputs //
let commandQueue = device?.makeCommandQueue()

let getFunctionLibrary = device?.makeDefaultLibrary()

let actInp = getFunctionLibrary?.makeFunction(name: "actInp")

var computePipelineState: MTLComputePipelineState!

do {
    computePipelineState = try device?.makeComputePipelineState(function: actInp!)
} catch {
    print(error)
}


// Make all memory buffers
let inpArrBuff = device?.makeBuffer(bytes: inpNeurArr,
                                    length: MemoryLayout<DType>.stride*inpNeurArr.count,
                                    options: .storageModeShared)

// will set apple recommendation for spike propagation
if (appleRec == true){
    
    gridSize = (inpNeurArr.count+blockSize-1) / blockSize

    
}

let contrArrBuff = device?.makeBuffer(bytes: contrArr,
                                      length: MemoryLayout<UInt32>.stride*contrArr.count,
                                      options: .storageModeShared)

let sizeArrBuff = device?.makeBuffer(bytes: sizeArr,
                                      length: MemoryLayout<DType>.stride*sizeArr.count,
                                      options: .storageModeShared)



// Encode command buffer
let commandBuff = commandQueue?.makeCommandBuffer()

let commandEncoder = commandBuff?.makeComputeCommandEncoder()
commandEncoder?.setComputePipelineState(computePipelineState)

// Set buffers
commandEncoder?.setBuffer(inpArrBuff, offset: 0, index: 0)
commandEncoder?.setBuffer(contrArrBuff, offset: 0, index: 1)
commandEncoder?.setBuffer(sizeArrBuff, offset: 0, index: 2)

// Dispatch threads and thread groups
let threadsPerThreadgroup = MTLSizeMake(blockSize, 1, 1)
let threadgroupsPerGrid = MTLSize(width: gridSize,
                                  height: 1,
                                  depth: 1)


commandEncoder?.dispatchThreadgroups(threadgroupsPerGrid,
                                           threadsPerThreadgroup: threadsPerThreadgroup)

commandEncoder?.endEncoding()


// Start Kernel 1 timer
let start1 = CFAbsoluteTimeGetCurrent()

commandBuff?.commit()
commandBuff?.waitUntilCompleted()

// End Kernel 1 timer
let end1 = CFAbsoluteTimeGetCurrent()


if copyRes == true {
    var contrArrPointer = contrArrBuff?.contents().bindMemory(to: UInt32.self, capacity: MemoryLayout<UInt32>.stride*contrArr.count)

    // Print out the output neurons after spiking
    for ind in 0..<contrArr.count {

        contrArr[ind] = contrArrPointer!.pointee
        contrArrPointer = contrArrPointer?.advanced(by: 1)

    }
}
    
if verbose != 0 {
    print("Counter Array: \(contrArr)")
}



/** NEURON WIPE PHASE **/

// Run the second kernel to process spikes and generate spike list
let commandQueue1 = device?.makeCommandQueue()

let getFunctionLibrary1 = device?.makeDefaultLibrary()

let procssInp = getFunctionLibrary1?.makeFunction(name: "procssInp")

var computePipelineState1: MTLComputePipelineState!

do {
    computePipelineState1 = try device?.makeComputePipelineState(function: procssInp!)
} catch {
    print(error)
}


// Encode command buffer
let commandBuff1 = commandQueue1?.makeCommandBuffer()

let commandEncoder1 = commandBuff1?.makeComputeCommandEncoder()
commandEncoder1?.setComputePipelineState(computePipelineState1)


let neurWipeArr: [DType] = [neurWipe, wipeTh, spikeTh, maxSpikes]
var spikeList: [DType] = Array<DType>(repeating: 0, count:  Int(maxSpikes)+1)

let neurWipeBuff = device?.makeBuffer(bytes: neurWipeArr,
                                    length: MemoryLayout<DType>.stride*neurWipeArr.count,
                                    options: .storageModeShared)

let spikeListBuff = device?.makeBuffer(bytes: spikeList,
                                    length: MemoryLayout<DType>.stride*spikeList.count,
                                    options: .storageModeShared)

// Set buffers

commandEncoder1?.setBuffer(contrArrBuff, offset: 0, index: 0)
commandEncoder1?.setBuffer(neurWipeBuff, offset: 0, index: 1)
commandEncoder1?.setBuffer(spikeListBuff, offset: 0, index: 2)
commandEncoder1?.setBuffer(sizeArrBuff, offset: 0, index: 3)



let threadsPerThreadgroup1 = MTLSizeMake(prcssBlockSize, 1, 1)
let threadgroupsPerGrid1 = MTLSize(width: prcssGridSize,
                                  height: 1,
                                  depth: 1)

commandEncoder1?.dispatchThreadgroups(threadsPerThreadgroup1,
                                           threadsPerThreadgroup: threadsPerThreadgroup1)

commandEncoder1?.endEncoding()

// Start Kernel 2 timer
let start2 = CFAbsoluteTimeGetCurrent()

commandBuff1?.commit()
commandBuff1?.waitUntilCompleted()

// End Kernel 1 timer
let end2 = CFAbsoluteTimeGetCurrent()


if copyRes == true {
    var contrArrPointer = contrArrBuff?.contents().bindMemory(to: UInt32.self, capacity: MemoryLayout<UInt32>.stride*contrArr.count)

    // Print out the output neurons after spiking
    for ind in 0..<contrArr.count {

        contrArr[ind] = contrArrPointer!.pointee
        contrArrPointer = contrArrPointer?.advanced(by: 1)

    }

    var spikeListPointer = spikeListBuff?.contents().bindMemory(to: DType.self, capacity: MemoryLayout<DType>.stride*spikeList.count)

    // Print out the output neurons after spiking
    for ind in 0..<spikeList.count {

        spikeList[ind] = spikeListPointer!.pointee
        spikeListPointer = spikeListPointer?.advanced(by: 1)

    }
}

if verbose != 0 {
    print("Counter Array after NeurWipe: \(contrArr)")
    print("Spike List: \(spikeList)")
}



// End timer
let end = CFAbsoluteTimeGetCurrent()

// Print out the time elapsed by Kernel 1
print("Time elapsed in Kernel 1 compute = \(end1 - start1)")

// Print out the time elapsed by Kernel 2
print("Time elapsed in Kernel 2 compute = \(end2 - start2)")

// Print out the total time elapsed
print("Total Time elapsed in GPU = \(end - start)")

