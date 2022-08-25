//
//  main.swift
//  forwardConnection_GPU
//
//  Created by Aditya on 25/08/2022.
//

import MetalKit

typealias DType = Int32

// Network size parameters
let numIn:    DType = 10000000
let numOut:   DType = 100000000
let numAct:   DType = 100000
let numConnx: DType = 4

// Neuron wipe parameters
let spikeTh: DType = 4
let neurWipe: DType = 2
let wipeTh: DType = 2
let maxSpikes: DType = 10

// Other parameters
let verbose: DType = 0  // no print for verbose zero
let copyRes: Bool  = false  // copy results bool (set true when verbose > 0)


// Kernel 1 Thread parameters (max Thread group size is 1024)
let appleRec: Bool = false // when set true, ignores user settings and choose apple settings

let thrGroup = 1024 // Apple recommend: 1024 (max possible threadgroups)

let blockSize = thrGroup
var gridSize =  10 // Apple recommend: (Array(inpNeurArr.joined()).count+blockSize-1) / blockSize


// Kernel 2 Thread parameters (max Thread group size is 1024)
let prcssThrGroup = 1024 // Apple recommend: 1024 (max possible threadgroups)

let prcssBlockSize = prcssThrGroup
var prcssGridSize = 100 // Apple recommend: (Int(numOut)+prcssBlockSize-1) / prcssBlockSize

if (appleRec == true){
    
    prcssGridSize = (Int(numOut)+prcssBlockSize-1) / prcssBlockSize
    
}

// Define struct for Output Neuron
struct OutNeuron {

    var counter: DType = 0
    var connxArr: [DType]
}


// Create empty list of output and input neurons
var outNeurArr: [OutNeuron] = []
var inpNeurArr: [[DType]] = []


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
var generator = RandomNumberGeneratorWithSeed(seed: 0)



// Populate the output Neurons array with random connections
for _ in 0..<numOut {

    outNeurArr.append( OutNeuron( connxArr: (0..<numConnx).map({_ in DType.random(in: 0..<numIn, using: &generator)}) ) )

}

// Store the input connections list by converting the output connections
var outToIn: [[DType]] = Array(repeating: [], count: Int(numIn))

for i in 0..<numOut {
    for j in 0..<numConnx {

        let inpInd = outNeurArr[Int(i)].connxArr[Int(j)]
        outToIn[Int(inpInd)].append(i)
    }
}

// Make activation function to create random activation vector
func makeActVect(_ numAct: DType,_ numIn: DType) -> [DType] {

    return (1...numAct).map( {_ in DType.random(in: 0..<numIn, using: &generator)} )

}



// Randomly activate inputs
let actVect: [DType] = makeActVect(numAct, numIn)


if (verbose == 2) {
    print(outNeurArr)
}


for ind in actVect {

    inpNeurArr.append(outToIn[Int(ind)])

}


if verbose != 0 {

print("Activated input indices: \(actVect)")
print("Connection array of activated inputs: \(inpNeurArr)")

}


// Create zero counter list and size array to be sent to the GPU
var contrArr: [UInt32] = Array<UInt32>(repeating: 0, count: Int(numOut / 8) + 1 )
let sizeArr: [DType] = [DType(Array(inpNeurArr.joined()).count), numOut]



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
let inpArrBuff = device?.makeBuffer(bytes: Array(inpNeurArr.joined()),
                                    length: MemoryLayout<DType>.stride*Array(inpNeurArr.joined()).count,
                                    options: .storageModeShared)

print("Events propagated: \(Array(inpNeurArr.joined()).count)")

if (appleRec == true){
    
    gridSize = (Array(inpNeurArr.joined()).count+blockSize-1) / blockSize
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




// Run the second kernel to process spikes and generate spike list  //
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

