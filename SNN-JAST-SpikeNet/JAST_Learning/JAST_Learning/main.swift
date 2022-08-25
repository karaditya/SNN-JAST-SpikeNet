//
//  main.swift
//  JAST_Learning
//
//  Created by Aditya on 26/08/2022.
//

import MetalKit

typealias DType = Int32


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


var generator = RandomNumberGeneratorWithSeed(seed: 9400)



// Define function for creating random connections
func makeRandConnx(_ numIn: DType,_ numConnx: DType) -> [DType] {
    
    let connxArr: [DType] =  (0..<numConnx).map({_ in DType.random(in: 0..<numIn, using: &generator)})
    
    return connxArr
}

// Populate the outNeurons
let numIn: DType = 10000//10//1000000
let numOut: DType = 1000000// 10//10000000
let numAct: DType =  256// 7//10000
let numConnx: DType = 4

// Verbosity parameters
let verbose: DType = 0  // no print for verbose zero
let copyRes: Bool  = true  // copy results bool (set true when verbose > 0)

// NeurWipe parameters
let spikeTh: DType = 4
let neurWipe: DType = 1
let wipeTh: DType = 4
let maxSpikes: DType = 10

// Learning parameters
let learnTh: DType = 2
let maxLearn: DType = 20//10000
let slideWindow: DType = numAct//numConnx



// Create an empty array for connections of output Neurons
let sizeArr: [DType] = [numOut, numAct, numConnx]
// Create an empty array of output Neurons
var outNeurons:[DType] = Array<DType>(repeating: 0, count:  Int(numOut*numConnx))

// Generate Random connections and populate the neurons
for ind in 0..<numOut {
    
    let tmp = makeRandConnx(numIn, numConnx)
    for j in 0..<numConnx{
       
        outNeurons[Int(numConnx*ind + j)] = tmp[Int(j)]
        
    }
    
    
    
}

// Make activation vector
func makeActVect(_ numAct: DType,_ numIn: DType) -> [DType] {
    
    return (1...numAct).map( {_ in DType.random(in: 0..<numIn, using: &generator)} )
    
}


if (verbose == 2){
    print(outNeurons)
}


// Define CPU function for scheduling Spike Propagation in GPU
func feed(_ outNeurons: [DType],_ actVect: [DType],_ contrArr: [UInt32]) -> [UInt32] {
    
    let outNeurons = outNeurons
    let actVect = actVect
    var contrArr = contrArr
    
    if (verbose != 0){
        
        print("Count array received from last iter: \(contrArr)")
        
    }
    
    // Start timer
    let start = CFAbsoluteTimeGetCurrent()


    let device = MTLCreateSystemDefaultDevice()

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
    let outNeuronsBuff = device?.makeBuffer(bytes: outNeurons,
                                            length: MemoryLayout<DType>.stride*outNeurons.count,
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


    let w = computePipelineState.threadExecutionWidth
    let h = computePipelineState.maxTotalThreadsPerThreadgroup / w
    let threadsPerThreadgroup = MTLSizeMake(w, h, 1)
    let threadgroupsPerGrid = MTLSize(width: (Int(numOut)+w-1) / w,
                                      height: (Int(numAct)+h-1) / h,
                                      depth: 1)


    commandEncoder?.dispatchThreadgroups(threadgroupsPerGrid,
                                               threadsPerThreadgroup: threadsPerThreadgroup)

    commandEncoder?.endEncoding()
    commandBuff?.commit()

    commandBuff?.waitUntilCompleted()

    // Store the pointer to the result
    let contrArrPointer = contrArrBuff?.contents().bindMemory(to: UInt32.self, capacity: MemoryLayout<UInt32>.size*contrArr.count)

    // Print out the counter array after spiking
    for ind in 0..<contrArr.count {
        
        let count = contrArrPointer?.advanced(by: ind).pointee
        contrArr[ind] = count!

    }
    
    // End timer
    let end = CFAbsoluteTimeGetCurrent()
    // Print out the time elapsed
    print("Time elapsed in GPU spike propagation = \(end - start)")
    
    return contrArr

}



// Define CPU function for scheduling Spike Processing in GPU
func spikeFun(_ contrArr: [UInt32]) -> ([DType], [DType], [UInt32]) {
    
    var contrArr = contrArr
    
    // Start timer
    let start = CFAbsoluteTimeGetCurrent()
    
    let device = MTLCreateSystemDefaultDevice()
    
    // Run the second kernel to process spikes and generate spike list //

    let commandQueue = device?.makeCommandQueue()

    let getFunctionLibrary = device?.makeDefaultLibrary()

    let procssInp = getFunctionLibrary?.makeFunction(name: "procssInp")

    var computePipelineState: MTLComputePipelineState!

    do {
        computePipelineState = try device?.makeComputePipelineState(function: procssInp!)
    } catch {
        print(error)
    }


    // Encode command buffer
    let commandBuff = commandQueue?.makeCommandBuffer()

    let commandEncoder = commandBuff?.makeComputeCommandEncoder()
    commandEncoder?.setComputePipelineState(computePipelineState)


    let neurWipeArr: [DType] = [neurWipe, wipeTh, spikeTh, maxSpikes, learnTh, maxLearn]
    
    var spikeList: [DType] = Array<DType>(repeating: 0, count:  Int(maxSpikes)+1)
    
    var learnList: [DType] = Array<DType>(repeating: 0, count:  Int(maxLearn)+1)
    
    
    let neurWipeBuff = device?.makeBuffer(bytes: neurWipeArr,
                                        length: MemoryLayout<DType>.stride*neurWipeArr.count,
                                        options: .storageModeShared)

    let spikeListBuff = device?.makeBuffer(bytes: spikeList,
                                        length: MemoryLayout<DType>.stride*spikeList.count,
                                        options: .storageModeShared)
    
    let learnListBuff = device?.makeBuffer(bytes: learnList,
                                        length: MemoryLayout<DType>.stride*learnList.count,
                                        options: .storageModeShared)

    let contrArrBuff = device?.makeBuffer(bytes: contrArr,
                                          length: MemoryLayout<UInt32>.stride*contrArr.count,
                                          options: .storageModeShared)
    
    
    // Set buffers
    commandEncoder?.setBuffer(contrArrBuff,  offset: 0, index: 0)
    commandEncoder?.setBuffer(neurWipeBuff,  offset: 0, index: 1)
    commandEncoder?.setBuffer(spikeListBuff, offset: 0, index: 2)
    commandEncoder?.setBuffer(learnListBuff, offset: 0, index: 3)

    let w = computePipelineState.threadExecutionWidth
    let thGRW = Int(numOut)
    let threadsPerThreadgroup = MTLSizeMake(w, 1, 1)
    let threadsPerGrid = MTLSize(width: thGRW,
                                      height: 1,
                                      depth: 1)


    commandEncoder?.dispatchThreads(threadsPerGrid,
                                               threadsPerThreadgroup: threadsPerThreadgroup)

    commandEncoder?.endEncoding()
    commandBuff?.commit()

    commandBuff?.waitUntilCompleted()
    
    var contrArrPointer = contrArrBuff?.contents().bindMemory(to: UInt32.self, capacity: MemoryLayout<UInt32>.stride*contrArr.count)

    // Print out the output neurons after spiking
    for ind in 0..<contrArr.count {

        contrArr[ind] = contrArrPointer!.pointee
        contrArrPointer = contrArrPointer?.advanced(by: 1)

    }
    
    if (verbose != 0){
        
        print("Counter after processing: \(contrArr)")
        
    }
    
    
    var spikeListPointer = spikeListBuff?.contents().bindMemory(to: DType.self, capacity: MemoryLayout<DType>.stride*spikeList.count)

    // Print out the output neurons after spiking
    for ind in 0..<spikeList.count {

        spikeList[ind] = spikeListPointer!.pointee
        spikeListPointer = spikeListPointer?.advanced(by: 1)

    }
    
    var learnListPointer = learnListBuff?.contents().bindMemory(to: DType.self, capacity: MemoryLayout<DType>.stride*learnList.count)

    // Print out the output neurons after learning threshold is reached
    for ind in 0..<learnList.count {

        learnList[ind] = learnListPointer!.pointee
        learnListPointer = learnListPointer?.advanced(by: 1)

    }
    
    // End timer
    let end = CFAbsoluteTimeGetCurrent()
    // Print out the time elapsed
    print("Time elapsed in GPU neurWipe = \(end - start)")
    
    
    return (spikeList, learnList, contrArr)
}



// Propagate and process the spikes for a couple of iterations
var actVect: [DType] = Array<DType>(repeating: 0, count:  Int(numAct))
var spikeList: [DType]
var learnList: [DType] = Array<DType>(repeating: 0, count: Int(maxLearn) )

// Create compressed zero counter list and spikeList
var contrArr: [UInt32] = Array<UInt32>(repeating: 0, count: Int(numOut / 8) + 1 )

var num_iter: DType = 4
for _ in 0..<num_iter {
    
    actVect = makeActVect(numAct, numIn)
    
    if (verbose != 0){
        print("Activated Inputs: \(actVect)")
    }
    
    
    contrArr = feed(outNeurons, actVect, contrArr)

    
    if (verbose != 0){
        print("Counter array before processing: \(contrArr)")
    }
    
    (spikeList, learnList, contrArr) = spikeFun(contrArr)

    print("Spike List generated: \(spikeList)")
    print("Learning threshold reached for neurons: \(learnList)")
    print("\n")
    
    
}


//print(learnList)


// Define CPU function for scheduling JAST Learning in GPU
func JASTLearn(_ outNeurons: [DType],_ learnList: [DType],_ learnAct: [DType]) -> [DType] {



    let learnList = learnList
    var ind = Int(learnList[0])
    var outNeurons1: [DType] = outNeurons
    var outNeurLearn: [DType] = Array<DType>(repeating: 0, count: ind*Int(numConnx))

    if ind != 0 {
        
        if ind > Int(maxLearn) {
            
            ind = Int(maxLearn)
        }
        
        for i in 0..<ind {
            for j in 0..<Int(numConnx) {
                
                outNeurLearn[4*i + j] = outNeurons[4*Int(learnList[i+1]) + j]
                
            }
        }
    }
    
    if (verbose != 0){
        
        print("Sequence for learning: \(learnAct)")
        print("Out Neurons before learning: \(outNeurLearn)")
        
    }
    
    

    // Start timer
    let start = CFAbsoluteTimeGetCurrent()

    let device = MTLCreateSystemDefaultDevice()

    // Run the third kernel for JAST learning of neurons that reached larning threshold //

    let commandQueue = device?.makeCommandQueue()

    let getFunctionLibrary = device?.makeDefaultLibrary()

    let JAST = getFunctionLibrary?.makeFunction(name: "JAST")

    var computePipelineState: MTLComputePipelineState!

    do {
        computePipelineState = try device?.makeComputePipelineState(function: JAST!)
    } catch {
        print(error)
    }


    // Encode command buffer
    let commandBuff = commandQueue?.makeCommandBuffer()

    let commandEncoder = commandBuff?.makeComputeCommandEncoder()
    commandEncoder?.setComputePipelineState(computePipelineState)

    
    let learnArr: [DType] = [learnTh, slideWindow, DType(ind), DType(learnAct.count), numConnx]

    let learnArrBuff = device?.makeBuffer(bytes: learnArr,
                                        length: MemoryLayout<DType>.stride*learnArr.count,
                                        options: .storageModeShared)


    let learnOutBuff = device?.makeBuffer(bytes: outNeurLearn,
                                          length: MemoryLayout<DType>.stride*outNeurLearn.count,
                                          options: .storageModeShared)

    let learnActBuff = device?.makeBuffer(bytes: learnAct,
                                          length: MemoryLayout<DType>.stride*learnAct.count,
                                          options: .storageModeShared)

    // Set buffers
    commandEncoder?.setBuffer(learnArrBuff, offset: 0, index: 0)
    commandEncoder?.setBuffer(learnOutBuff, offset: 0, index: 1)
    commandEncoder?.setBuffer(learnActBuff, offset: 0, index: 2)


    let w = computePipelineState.threadExecutionWidth
    let h = computePipelineState.maxTotalThreadsPerThreadgroup / w
    let threadsPerThreadgroup = MTLSizeMake(w, h, 1)
    let threadgroupsPerGrid = MTLSize(width: (ind+w-1) / w,
                                      height: (learnAct.count+h-1) / h,
                                      depth: 1)


    commandEncoder?.dispatchThreadgroups(threadgroupsPerGrid,
                                               threadsPerThreadgroup: threadsPerThreadgroup)

    commandEncoder?.endEncoding()
    commandBuff?.commit()

    commandBuff?.waitUntilCompleted()

    var learnOutPointer = learnOutBuff?.contents().bindMemory(to: DType.self, capacity: MemoryLayout<DType>.stride*outNeurLearn.count)

    // Print out the output neurons after spiking
    
    for i in 0..<ind {
        for j in 0..<Int(numConnx){
            
            outNeurons1[4*Int(learnList[i+1]) + j] = learnOutPointer!.pointee
            outNeurLearn[4*i + j] = learnOutPointer!.pointee
            learnOutPointer = learnOutPointer?.advanced(by: 1)
        }
    }
    
    if (verbose != 0){
        
        print("Output neurons after learning: \(outNeurLearn)")
        
    }
    


    // End timer
    let end = CFAbsoluteTimeGetCurrent()
    // Print out the time elapsed
    print("Time elapsed in GPU Learning = \(end - start)")


    return outNeurons1
}

let randEvents1: [DType] = (1...numAct).map( {_ in DType.random(in: 0..<numIn, using: &generator)} )
let randEvents2: [DType] = (1...numAct).map( {_ in DType.random(in: 0..<numIn, using: &generator)} )

// Learning array with pattern
let learnAct: [DType] = randEvents1 + actVect + randEvents2

// Random array with no pattern
let learnAct1: [DType] = (1...learnAct.count).map( {_ in DType.random(in: 0..<numIn, using: &generator)} )

// Scrambled array with scattered pattern
var learnAct2: [DType] = randEvents1
for i in actVect {
    
    learnAct2 +=  [i] + randEvents1
    
}

//var learnAct: [DType] = [0, 9] + actVect + [ 2, 3]
outNeurons = JASTLearn(outNeurons, learnList, learnAct)


if (verbose == 2) {
    
    print(outNeurons)
    print("\n")
    
}


if (verbose != 0) {
    
    print("Activation received: \(actVect)")
    
    
}

contrArr = feed(outNeurons, actVect, contrArr)

if (verbose != 0){
    print("Counter array before processing: \(contrArr)")
}

(spikeList, learnList, contrArr) = spikeFun(contrArr)

print("Spike List generated: \(spikeList)")
print("Learning threshold reached for neurons: \(learnList)")
print("\n")
