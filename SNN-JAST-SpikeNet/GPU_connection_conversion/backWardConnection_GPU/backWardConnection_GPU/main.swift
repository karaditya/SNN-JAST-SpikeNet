//
//  main.swift
//  backWardConnection_GPU
//
//  Created by Aditya on 25/08/2022.
//

import MetalKit

typealias DType = Int32


// SNN architecture parameters
let numIn: DType = 1000000000
let numOut: DType = 1000000
let numAct: DType = 10000000
let numConnx: DType = 4


// NeurWipe parameters
let spikeTh: DType = 4
let neurWipe: DType = 3
let wipeTh: DType = 2
let maxSpikes: DType = 10


// Verbosity parameters
let verbose: DType = 0  // no print for verbose zero
let copyRes: Bool  = false  // copy results bool (set true when verbose > 0)


// Kernel 1 Thread parameters (max Thread group size is 1024)
let appleRec: Bool = false // when set true, ignores user settings and choose apple settings
let execWidth = 32 // Apple recommend: 32
let thrGroup = 1024 // Apple recommend: 1024 (max possible threadgroups)

let blockSizeX = execWidth
let blockSizeY = thrGroup / blockSizeX

var gridSizeX = 100 // Apple recommend: (Int(numOut)+blockSizeX-1) / blockSizeX
var gridSizeY = 10 // Apple recommend: (Int(numAct)+blockSizeY-1) / blockSizeY

// Kernel 2 Thread parameters (max Thread group size is 1024)
let prcssThrGroup = 1024 // Apple recommend: 1024 (max possible threadgroups)

let prcssBlockSize = prcssThrGroup
var prcssGridSize = 100 // Apple recommend: (Int(numOut)+prcssBlockSize-1) / prcssBlockSize

if (appleRec == true){
    
    gridSizeX = (Int(numOut)+blockSizeX-1) / blockSizeX
    gridSizeY = (Int(numAct)+blockSizeY-1) / blockSizeY
    
    prcssGridSize = (Int(numOut)+prcssBlockSize-1) / prcssBlockSize
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
var generator = RandomNumberGeneratorWithSeed(seed: 11)



// Create an empty array for connections of output Neurons
var connxTout: [[DType]] = []

// Define function for creating random connections
func makeRandConnx(_ numIn: DType,_ numConnx: DType) -> [DType] {
    
    let connxArr: [DType] =  (0..<numConnx).map({_ in DType.random(in: 0..<numIn, using: &generator)})
    
    return connxArr
}


// Generate Random connections and populate the neurons
for _ in 0..<numOut {
    
    let tmp = makeRandConnx(numIn, numConnx)
    connxTout.append(tmp)
    
    
}


// Define function for making activation
func makeActVect(_ numAct: DType,_ numIn: DType) -> [DType] {
    
    return (1...numAct).map( {_ in DType.random(in: 0..<numIn, using: &generator)} )
}

// Make activation vector and size array
let actVect: [DType] = makeActVect(numAct, numIn)
let sizeArr: [DType] = [numOut, numAct, numConnx]

// Create compressed zero counter list and spikeList
var contrArr: [UInt32] = Array<UInt32>(repeating: 0, count: Int(numOut / 8) + 1 )
var spikeLst:[DType] = []


if (verbose != 0){
    print("Activated Inputs: \(actVect)")
}


if (verbose == 2){
    print(connxTout)
}



// Start timer
let start = CFAbsoluteTimeGetCurrent()

// Create GPU device state for compute kernels
let device = MTLCreateSystemDefaultDevice()

// Run the first kernel on device to propagate spikes and update counter array //
let commandQueue = device?.makeCommandQueue()

let getFunctionLibrary = device?.makeDefaultLibrary()

let actInp = getFunctionLibrary?.makeFunction(name: "actInp")


var computePipelineState: MTLComputePipelineState!

do {
    computePipelineState = try device?.makeComputePipelineState(function: actInp!)
} catch {
    print(error)
}


// Create buffer of Neuron type data
let outNeuronsBuff = device?.makeBuffer(bytes: Array(connxTout.joined()),
                                      length: MemoryLayout<DType>.stride*Array(connxTout.joined()).count,
                                      options: .storageModeShared)

let actVectBuff = device?.makeBuffer(bytes: actVect,
                                     length: MemoryLayout<DType>.stride*actVect.count,
                                     options: .storageModeShared)

let contrArrBuff = device?.makeBuffer(bytes: contrArr,
                                      length: MemoryLayout<UInt32>.stride*contrArr.count,
                                      options: .storageModeShared)

let sizeArrBuff = device?.makeBuffer(bytes: sizeArr,
                                     length: MemoryLayout<DType>.stride*sizeArr.count,
                                     options: .storageModeShared)



// Create command buffer object
let commandBuff = commandQueue?.makeCommandBuffer()

// Encode commands in the command buffer
let commandEncoder = commandBuff?.makeComputeCommandEncoder()
commandEncoder?.setComputePipelineState(computePipelineState)


// Set Out Neurons buffer
commandEncoder?.setBuffer(outNeuronsBuff, offset: 0, index: 0)
commandEncoder?.setBuffer(actVectBuff, offset: 0, index: 1)
commandEncoder?.setBuffer(contrArrBuff, offset: 0, index: 2)
commandEncoder?.setBuffer(sizeArrBuff, offset: 0, index: 3)


// Dispatch threads and thread groups
let threadsPerThreadgroup = MTLSizeMake(blockSizeX, blockSizeY, 1)
let threadgroupsPerGrid = MTLSize(width: gridSizeX,
                                  height: gridSizeY,
                                  depth: 1)


commandEncoder?.dispatchThreadgroups(threadgroupsPerGrid,
                                           threadsPerThreadgroup: threadsPerThreadgroup)

commandEncoder?.endEncoding()
commandBuff?.commit()

commandBuff?.waitUntilCompleted()


if (copyRes != false){
    
    // Store the pointer to the result
    let contrArrPointer = contrArrBuff?.contents().bindMemory(to: UInt32.self, capacity: MemoryLayout<UInt32>.size*contrArr.count)

    // Print out the counter array after spiking
    for ind in 0..<contrArr.count {
        
        let count = contrArrPointer?.advanced(by: ind).pointee
        contrArr[ind] = count!

    }
}


if (verbose != 0) {
    print("Counter Array: \(contrArr)")
}


// Run the second kernel to process spikes and generate spike list //

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
commandBuff1?.commit()

commandBuff1?.waitUntilCompleted()






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
// Print out the time elapsed
print("Time elapsed in GPU compute = \(end - start)")

